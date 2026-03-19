from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if SRC_DIR.exists():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from tooth_service.config import ensure_checkpoint_exists
from tooth_service.constants import DEFAULT_BOT_Q, DEFAULT_SAM_MODEL_TYPE, DEFAULT_TOP_Q
from tooth_service.image_io import decode_uploaded_image
from tooth_service.pipeline import analyze_image
from tooth_service.sam_runner import load_sam_model
from tooth_service.visualization import (
    download_filename,
    download_payload_from_serialized,
    export_payload,
    results_to_rows,
    status_message,
    warning_lines,
)

DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'sam_vit_h_4b8939.pth'


def _get_streamlit():
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime-only guard
        raise RuntimeError(
            'Streamlit is required to run this app. Install dependencies with `pip install -r requirements.txt`.'
        ) from exc
    return st


def _resolve_optional_device(raw_device: str) -> Optional[str]:
    value = raw_device.strip()
    return value or None


def _analysis_settings(st):
    with st.sidebar:
        st.title('Settings')
        checkpoint_path = st.text_input('SAM checkpoint path', value=str(DEFAULT_CHECKPOINT))
        device = st.text_input('Device', value='')

        with st.expander('Advanced settings', expanded=False):
            top_q = st.number_input('top_q', min_value=0.0, max_value=1.0, value=float(DEFAULT_TOP_Q), step=0.01)
            bot_q = st.number_input('bot_q', min_value=0.0, max_value=1.0, value=float(DEFAULT_BOT_Q), step=0.01)
            smooth = st.number_input('smooth', min_value=1, max_value=31, value=7, step=2)
            pad = st.number_input('pad', min_value=0, max_value=64, value=10, step=1)
            points_per_side = st.number_input('points_per_side', min_value=4, max_value=64, value=16, step=1)
            points_per_batch = st.number_input('points_per_batch', min_value=1, max_value=32, value=4, step=1)

    mask_generator_kwargs = {
        'points_per_side': int(points_per_side),
        'points_per_batch': int(points_per_batch),
        'crop_n_layers': 0,
        'min_mask_region_area': 100,
    }

    return {
        'checkpoint_path': checkpoint_path,
        'device': _resolve_optional_device(device),
        'top_q': float(top_q),
        'bot_q': float(bot_q),
        'smooth': int(smooth),
        'pad': int(pad),
        'mask_generator_kwargs': mask_generator_kwargs,
    }


def _cached_model_loader(st):
    @st.cache_resource(show_spinner=False)
    def _load(checkpoint_path: str, device: Optional[str]):
        return load_sam_model(
            model_type=DEFAULT_SAM_MODEL_TYPE,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    return _load


def _run_analysis(image_rgb, settings, st):
    checkpoint = ensure_checkpoint_exists(settings['checkpoint_path'])
    cached_load = _cached_model_loader(st)
    sam_model = cached_load(str(checkpoint), settings['device'])

    return analyze_image(
        image_rgb,
        checkpoint_path=str(checkpoint),
        model_type=DEFAULT_SAM_MODEL_TYPE,
        sam_model=sam_model,
        device=settings['device'],
        mask_generator_kwargs=settings['mask_generator_kwargs'],
        top_q=settings['top_q'],
        bot_q=settings['bot_q'],
        smooth=settings['smooth'],
        pad=settings['pad'],
    )


def _clear_analysis_result(st):
    st.session_state.analysis_output = None
    st.session_state.analysis_image = None


def _render_result(st, image_rgb, output):
    serialized = export_payload(output)
    status = serialized.get('status', 'ok')

    st.subheader('Result')
    st.caption(status_message(serialized))

    if serialized.get('error'):
        stage = serialized.get('error_stage') or 'analysis'
        st.error(f"Analysis failed during {stage.replace('_', ' ')}.")
    elif status == 'empty':
        st.info('No tooth candidates were found after filtering.')
    else:
        st.success(f"Analysis complete: {serialized.get('instances_count', 0)} instances")

    for warning in warning_lines(serialized.get('warnings', [])):
        st.warning(warning)

    columns = st.columns(2)
    with columns[0]:
        st.image(image_rgb, caption='Original image', use_column_width=True)
    with columns[1]:
        st.image(output['overlay_image'], caption='Overlay', use_column_width=True)

    rows = results_to_rows(serialized.get('results', []))
    st.subheader('Per-tooth measurements')
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.write('No measurements to show.')

    st.download_button(
        label='Download JSON',
        data=download_payload_from_serialized(serialized),
        file_name=download_filename(serialized),
        mime='application/json',
    )


def main():
    st = _get_streamlit()
    st.set_page_config(page_title='Tooth Streamlit Service', layout='wide')
    st.title('Tooth Streamlit Service')
    st.write(
        'Upload a dental image, run tooth conicity analysis in the browser, and review the per-tooth output with a JSON download.'
    )

    settings = _analysis_settings(st)
    uploaded_file = st.file_uploader(
        'Upload one image',
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
        accept_multiple_files=False,
    )
    run_clicked = st.button('Run analysis')

    if 'analysis_output' not in st.session_state:
        st.session_state.analysis_output = None
        st.session_state.analysis_image = None

    if run_clicked:
        if uploaded_file is None:
            _clear_analysis_result(st)
            st.error('Upload an image before running analysis.')
        else:
            try:
                image_rgb = decode_uploaded_image(uploaded_file.getvalue())
            except Exception as exc:
                _clear_analysis_result(st)
                st.error(str(exc))
            else:
                with st.spinner('Running analysis...'):
                    try:
                        output = _run_analysis(image_rgb, settings, st)
                    except Exception as exc:
                        _clear_analysis_result(st)
                        st.error(str(exc))
                    else:
                        st.session_state.analysis_output = output
                        st.session_state.analysis_image = image_rgb

    if st.session_state.analysis_output is not None and st.session_state.analysis_image is not None:
        _render_result(st, st.session_state.analysis_image, st.session_state.analysis_output)


if __name__ == '__main__':
    main()

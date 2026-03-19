import importlib
import sys
from pathlib import Path

import numpy as np


class _DummySessionState(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value


class _DummyStreamlit:
    def __init__(self):
        self.session_state = _DummySessionState(
            analysis_output={"status": "ok"},
            analysis_image="old-image",
        )
        self.image_calls = []
        self.dataframe_calls = []
        self.download_calls = []

    def subheader(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def columns(self, count):
        return [_DummyColumn() for _ in range(count)]

    def image(
        self,
        image,
        caption=None,
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    ):
        self.image_calls.append(
            {
                "image": image,
                "caption": caption,
                "width": width,
                "use_column_width": use_column_width,
                "clamp": clamp,
                "channels": channels,
                "output_format": output_format,
            }
        )

    def dataframe(self, data, **kwargs):
        self.dataframe_calls.append({"data": data, **kwargs})

    def download_button(self, **kwargs):
        self.download_calls.append(kwargs)


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_app_module_imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module("app")
    assert mod is not None
    assert hasattr(mod, "main")


def test_clear_analysis_result_resets_session_state():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    app = importlib.import_module("app")

    st = _DummyStreamlit()

    app._clear_analysis_result(st)

    assert st.session_state.analysis_output is None
    assert st.session_state.analysis_image is None


def test_render_result_uses_streamlit_image_compatible_kwargs():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    app = importlib.import_module("app")

    st = _DummyStreamlit()
    image_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    output = {
        "status": "ok",
        "error": False,
        "error_stage": None,
        "instances_count": 1,
        "overlay_image": image_rgb.copy(),
        "results": [
            {
                "id": 1,
                "bbox_xyxy": [0, 0, 3, 3],
                "conicity_width_deg": 10.0,
                "conicity_lr_deg": 11.0,
                "w_top": 1.0,
                "w_bot": 2.0,
                "h_eff": 3.0,
            }
        ],
        "warnings": [],
    }

    app._render_result(st, image_rgb, output)

    assert len(st.image_calls) == 2
    assert st.image_calls[0]["use_column_width"] is True
    assert st.image_calls[1]["use_column_width"] is True

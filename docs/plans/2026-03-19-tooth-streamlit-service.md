# Tooth Streamlit Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the existing notebook into a local browser-based Streamlit service that accepts one uploaded image and returns tooth conicity results with visual overlays and JSON export.

**Architecture:** Extract the notebook into small Python modules with a single orchestration pipeline, then place a thin Streamlit UI on top. Keep the current SAM-based workflow for `v1`, remove classifier and Colab-only paths, and normalize the result schema so the UI and export layer consume a stable data contract.

**Tech Stack:** Python 3, Streamlit, NumPy, OpenCV, PyTorch, segment-anything, pytest

---

### Task 1: Scaffold the app layout and dependency files

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/tooth_service/__init__.py`
- Create: `src/tooth_service/constants.py`

**Step 1: Write the failing test**

Create a basic import smoke test in `tests/test_imports.py`:

```python
from tooth_service.constants import DEFAULT_TOP_Q, DEFAULT_BOT_Q


def test_default_thresholds_are_exposed():
    assert DEFAULT_TOP_Q == 0.15
    assert DEFAULT_BOT_Q == 0.65
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_imports.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing package/module file.

**Step 3: Write minimal implementation**

Add a minimal package with constants used by the notebook-derived pipeline. Include dependencies in `requirements.txt`:

```txt
streamlit
numpy
opencv-python
matplotlib
torch
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
pytest
```

Use a `.gitignore` that excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `checkpoints/`, and generated outputs.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt .gitignore README.md src/tooth_service/__init__.py src/tooth_service/constants.py tests/test_imports.py
git commit -m "chore: scaffold tooth streamlit service"
```

### Task 2: Extract and normalize taper calculation logic

**Files:**
- Create: `src/tooth_service/taper.py`
- Create: `tests/test_taper.py`
- Modify: `src/tooth_service/constants.py`

**Step 1: Write the failing test**

Create tests for the notebook's pure geometry functions:

```python
import numpy as np

from tooth_service.taper import find_closest_angle, trapezoid_taper_no_rot


def test_find_closest_angle_clamps_to_supported_range():
    assert find_closest_angle(0.01) == 5
    assert find_closest_angle(0.90) == 28


def test_trapezoid_taper_returns_normalized_result_schema():
    mask = np.zeros((40, 20), dtype=np.uint8)
    for y in range(5, 35):
        left = 6 - min((y - 5) // 10, 2)
        right = 13 + min((y - 5) // 10, 2)
        mask[y, left:right + 1] = 1

    result = trapezoid_taper_no_rot(mask, top_q=0.15, bot_q=0.65, smooth=1)

    assert result is not None
    assert "angle_from_dict" in result
    assert "conicity_width_deg" in result
    assert result["conicity_width_deg"] == result["angle_from_dict"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_taper.py -v`
Expected: FAIL because `taper.py` does not exist yet.

**Step 3: Write minimal implementation**

Move and clean these notebook functions into `taper.py`:
- `find_closest_angle`
- `widths_by_row`
- `trapezoid_taper_no_rot`

Normalize the return schema so downstream code can reliably use both:

```python
return {
    "angle_from_dict": angle_from_dict,
    "conicity_width_deg": float(angle_from_dict),
    "conicity_lr_deg": conicity_lr_deg,
    ...
}
```

Keep this module free of Streamlit and file I/O.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_taper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tooth_service/constants.py src/tooth_service/taper.py tests/test_taper.py
git commit -m "feat: extract taper measurement logic"
```

### Task 3: Extract tooth mask filtering and crop generation

**Files:**
- Create: `src/tooth_service/mask_filtering.py`
- Create: `tests/test_mask_filtering.py`

**Step 1: Write the failing test**

Cover two notebook behaviors: mask metrics and crop extraction.

```python
import numpy as np

from tooth_service.mask_filtering import crop_from_mask, mask_props


def test_mask_props_returns_bbox_metrics():
    seg = np.zeros((20, 20), dtype=np.uint8)
    seg[4:15, 6:12] = 1
    props = mask_props(seg)
    assert props is not None
    assert props["w"] > 0
    assert props["h"] > 0


def test_crop_from_mask_returns_crop_mask_and_bbox():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    seg = np.zeros((20, 20), dtype=np.uint8)
    seg[5:10, 7:11] = 1
    crop, mask_crop, bbox = crop_from_mask(img, seg, pad=0)
    assert crop.shape[:2] == mask_crop.shape
    assert bbox == (7, 5, 10, 9)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_mask_filtering.py -v`
Expected: FAIL because `mask_filtering.py` does not exist yet.

**Step 3: Write minimal implementation**

Move and clean these notebook functions into `mask_filtering.py`:
- `mask_props`
- `iou`
- `overlap_min`
- `color_stats_hsv`
- `select_instances`
- `resize_mask`
- `crop_from_mask`

Add one orchestration helper such as:

```python
def build_tooth_items(image_rgb, raw_masks, *, iou_thresh=0.50, contain_thresh=0.85, max_instances=24, pad=10):
    ...
    return tooth_items, instances
```

The helper should accept raw SAM masks and return a list of tooth records with `id`, `crop`, `mask_crop`, `bbox`, and the full-image segmentation.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_mask_filtering.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tooth_service/mask_filtering.py tests/test_mask_filtering.py
git commit -m "feat: extract tooth mask filtering"
```

### Task 4: Add image decoding and checkpoint configuration helpers

**Files:**
- Create: `src/tooth_service/image_io.py`
- Create: `src/tooth_service/config.py`
- Create: `tests/test_image_io.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
import io
import numpy as np
from PIL import Image

from tooth_service.image_io import decode_uploaded_image
from tooth_service.config import resolve_checkpoint_path


def test_decode_uploaded_image_returns_rgb_array():
    img = Image.new("RGB", (4, 3), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    arr = decode_uploaded_image(buf.getvalue())
    assert arr.shape == (3, 4, 3)


def test_resolve_checkpoint_path_prefers_existing_file(tmp_path):
    ckpt = tmp_path / "sam_vit_h_4b8939.pth"
    ckpt.write_bytes(b"x")
    assert resolve_checkpoint_path(ckpt) == ckpt
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_image_io.py tests/test_config.py -v`
Expected: FAIL because helper modules do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- `decode_uploaded_image(file_bytes) -> np.ndarray`
- `resolve_checkpoint_path(path_like) -> Path`
- optional `ensure_checkpoint_exists(...)` that raises a clear exception instead of silently failing

Do not auto-download in the first implementation unless the behavior is explicit and surfaced in the UI. Hidden download side effects make debugging harder.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_image_io.py tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tooth_service/image_io.py src/tooth_service/config.py tests/test_image_io.py tests/test_config.py
git commit -m "feat: add image and checkpoint helpers"
```

### Task 5: Wrap SAM model loading and raw mask generation

**Files:**
- Create: `src/tooth_service/sam_runner.py`
- Create: `tests/test_sam_runner.py`
- Modify: `src/tooth_service/constants.py`

**Step 1: Write the failing test**

The model itself is too heavy for a real unit test, so write a behavior test around validation and API shape.

```python
import numpy as np
import pytest

from tooth_service.sam_runner import resize_longest, validate_model_type


def test_resize_longest_leaves_small_image_unchanged():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    out = resize_longest(img, max_side=256)
    assert out.shape == img.shape


def test_validate_model_type_rejects_unknown_values():
    with pytest.raises(ValueError):
        validate_model_type("bad-model")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_sam_runner.py -v`
Expected: FAIL because `sam_runner.py` does not exist yet.

**Step 3: Write minimal implementation**

Create:
- `resize_longest`
- `validate_model_type`
- `load_sam_model(model_type, checkpoint_path, device)`
- `generate_masks(image_rgb, ...)`

Use lazy imports for `segment_anything` so pure unit tests do not require a loaded model.

If `torch.cuda.is_available()` is false, fall back to CPU without crashing. Only use `torch.autocast("cuda", dtype=torch.float16)` when the selected device is CUDA.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_sam_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tooth_service/constants.py src/tooth_service/sam_runner.py tests/test_sam_runner.py
git commit -m "feat: wrap sam loading and inference"
```

### Task 6: Build the orchestration pipeline and result serializer

**Files:**
- Create: `src/tooth_service/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Use stubs instead of a real SAM run.

```python
import numpy as np

from tooth_service.pipeline import analyze_image


def test_analyze_image_returns_structured_results(monkeypatch):
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    def fake_generate_masks(*args, **kwargs):
        return [{"segmentation": np.pad(np.ones((8, 6), dtype=np.uint8), ((10, 14), (12, 14)))}]

    monkeypatch.setattr("tooth_service.pipeline.generate_masks", fake_generate_masks)

    result = analyze_image(image, checkpoint_path="dummy.pth", model_type="vit_h")

    assert "results" in result
    assert "overlay_image" in result
    assert isinstance(result["results"], list)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_pipeline.py -v`
Expected: FAIL because `pipeline.py` does not exist yet.

**Step 3: Write minimal implementation**

In `pipeline.py`, compose:
- image validation
- mask generation
- tooth item extraction
- taper calculation
- normalization to a stable result schema
- overlay image generation

Return a dictionary like:

```python
{
    "results": [...],
    "overlay_image": overlay_rgb,
    "instances_count": 3,
    "warnings": [],
}
```

Partial failures should append a warning instead of aborting the whole analysis.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tooth_service/pipeline.py tests/test_pipeline.py
git commit -m "feat: add end-to-end analysis pipeline"
```

### Task 7: Build the Streamlit UI

**Files:**
- Create: `app.py`
- Create: `src/tooth_service/visualization.py`
- Modify: `README.md`

**Step 1: Write the failing test**

For the UI layer, use a lightweight import smoke test.

```python
import importlib


def test_app_module_imports():
    mod = importlib.import_module("app")
    assert mod is not None
```

Save it in `tests/test_app_import.py`.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_app_import.py -v`
Expected: FAIL because `app.py` does not exist yet.

**Step 3: Write minimal implementation**

Build `app.py` with this flow:
- page title and short description
- checkpoint path input or config default
- `st.file_uploader` for one image
- `st.button("Run analysis")`
- spinner during processing
- on success: show original image, overlay image, results table, JSON download button
- on failure: show `st.error(...)`
- on warnings: show `st.warning(...)`

Cache the loaded SAM model where possible to avoid reloading it on every rerun.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_app_import.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py src/tooth_service/visualization.py README.md tests/test_app_import.py
git commit -m "feat: add streamlit browser interface"
```

### Task 8: Add execution docs and run verification

**Files:**
- Modify: `README.md`
- Create: `tests/conftest.py`
- Optional create: `scripts/smoke_run.py`

**Step 1: Write the failing test**

Add one final smoke check around result serialization or CLI-free import flow, for example:

```python
from tooth_service.pipeline import serialize_results


def test_serialize_results_returns_json_safe_objects():
    payload = serialize_results([
        {"id": 1, "bbox_xyxy": (1, 2, 3, 4), "conicity_width_deg": 8.0}
    ])
    assert payload[0]["id"] == 1
    assert payload[0]["bbox_xyxy"] == [1, 2, 3, 4]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_pipeline.py::test_serialize_results_returns_json_safe_objects -v`
Expected: FAIL because the serializer helper does not exist yet.

**Step 3: Write minimal implementation**

Add or finish:
- JSON-safe serialization helpers
- `README.md` instructions for:
  - creating a virtualenv
  - installing dependencies
  - placing the SAM checkpoint under `checkpoints/`
  - running `streamlit run app.py`
- optional smoke script if it meaningfully reduces manual setup cost

**Step 4: Run test to verify it passes**

Run the full verification set:

```bash
PYTHONPATH=src pytest -v
```

Then, if dependencies and checkpoint are available, run the app manually:

```bash
streamlit run app.py
```

Expected:
- all tests PASS
- the app opens in a browser
- uploading one image produces a result table and JSON download

**Step 5: Commit**

```bash
git add README.md tests/conftest.py scripts/smoke_run.py src/tooth_service/pipeline.py
git commit -m "docs: document local streamlit workflow"
```

# Tooth Streamlit Service Design

## Goal

Turn the existing notebook-based tooth analysis workflow into a browser-accessible web service that runs locally on a user's computer and returns tooth conicity results from a single uploaded image.

## Current Context

The workspace currently contains a single notebook, `tooth_sam_taper_classifier.ipynb`, with this pipeline:
- load image from a hard-coded local/Colab path
- run `SAM` automatic mask generation
- filter masks to tooth-like instances
- crop each detected tooth
- compute taper/conicity metrics from each tooth mask
- optionally train and run a classifier
- export JSON results

For `v1`, only the upload-and-measure path is in scope. The classifier and manual labeling flow are explicitly out of scope.

## Requirements Confirmed

- The service must run locally on a computer.
- The user must interact with it through a browser.
- Input is a single uploaded image.
- Output must include conicity results.
- The long-term target is a public link for any user, but the first implementation is local.
- The solution should remain free to use in the first public deployment iteration.

## Approaches Considered

### 1. Streamlit with the existing SAM pipeline

Pros:
- closest to the current notebook logic
- fastest path to a working browser UI
- minimal algorithmic rewrite

Cons:
- the current `SAM vit_h` setup is heavy for free public hosting
- notebook code must be modularized before it is maintainable

### 2. Streamlit with a lighter segmentation model

Pros:
- easier future public deployment on free hosting
- smaller resource footprint

Cons:
- not a faithful transfer of the current notebook
- would require revalidation of segmentation quality and conicity outputs

### 3. Static site via GitHub Pages

Pros:
- free and easy to publish

Cons:
- not viable for this workflow because GitHub Pages does not run a Python backend or ML inference

## Recommendation

Use Approach 1 for `v1`: build a local `Streamlit` application around the existing `SAM`-based workflow. Once the local version is stable, prepare a separate deployment adaptation for free public hosting, most likely with infrastructure or model changes as needed.

## Architecture

The service will be a single-page `Streamlit` app with the notebook logic extracted into importable Python modules.

Proposed structure:
- `app.py`: Streamlit entrypoint and UI
- `src/tooth_service/config.py`: runtime configuration and checkpoint resolution
- `src/tooth_service/image_io.py`: uploaded file decoding and image conversion helpers
- `src/tooth_service/sam_runner.py`: `SAM` model loading and mask generation
- `src/tooth_service/mask_filtering.py`: tooth-like mask filtering, deduplication, crop extraction
- `src/tooth_service/taper.py`: conicity/taper calculation logic
- `src/tooth_service/visualization.py`: overlays and preview rendering for UI
- `src/tooth_service/pipeline.py`: orchestration layer from image input to structured results
- `tests/...`: unit and smoke tests
- `checkpoints/`: model weights

## Data Flow

1. User opens the local app in a browser.
2. User uploads one image through `Streamlit`.
3. The app decodes the file into an RGB image array.
4. The pipeline loads or reuses the configured `SAM` checkpoint.
5. `SAM` generates raw masks.
6. Mask filtering keeps only likely tooth instances and removes duplicates.
7. Each retained instance is cropped.
8. The taper module computes conicity metrics per tooth.
9. The app renders visual overlays and a results table.
10. The app offers JSON export.

## UI Scope for V1

Single page with four sections:
- upload area
- minimal run controls
- visual output
- results table and JSON download

Visible controls:
- image upload
- run analysis button
- optional advanced settings expander for technical thresholds

Visible outputs:
- original image
- image with mask/contour overlays
- per-tooth preview or summary where useful
- results table with fields such as `tooth_id`, `bbox`, `angle_from_dict`, `conicity_lr_deg`, `w_top`, `w_bot`, `h_eff`
- downloadable `results.json`

## Error Handling

The app must fail explicitly and locally for these cases:
- uploaded file is not a valid image
- checkpoint is missing
- checkpoint download fails
- model initialization fails
- segmentation returns no masks
- filtering leaves no tooth candidates
- taper cannot be computed for some teeth

Behavior rules:
- total pipeline failure produces a clear UI error
- partial failures do not crash the app; they are surfaced in the results table and warnings

## Non-Goals for V1

- classifier training
- manual labeling workflow
- multi-image batch mode
- persistence/database storage
- authentication
- true cloud deployment in the first implementation step

## Testing Strategy

Three levels are sufficient for the first iteration:
- unit tests for pure geometry and mask-processing functions
- smoke test for the end-to-end pipeline on a small fixture image or stubbed masks
- manual browser verification of the Streamlit UI

## Deployment Path

Phase 1:
- local browser app via `streamlit run app.py`

Phase 2:
- move the code to GitHub
- adapt for free public deployment, likely `Hugging Face Spaces`
- if current `SAM` weights are too heavy, evaluate a lighter model as a separate deployment concern rather than changing the local `v1` algorithm immediately

## Known Refactor Notes

The notebook contains notebook-only and Colab-only code that must be removed from the service implementation:
- `!pip ...`
- `google.colab.drive`
- hard-coded image paths
- direct `matplotlib` notebook display flow

There is also a data-model inconsistency in the notebook: later cells expect `conicity_width_deg`, while the taper function currently returns `angle_from_dict`. The service refactor must normalize this output schema before UI and JSON export are built on top of it.

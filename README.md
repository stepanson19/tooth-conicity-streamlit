---
title: Tooth Conicity App
sdk: docker
app_port: 8501
short_description: Tooth conicity measurement from uploaded dental images
---

# Tooth Streamlit Service

Local Streamlit wrapper for tooth conicity measurement.

The dependency pins in `requirements.txt` assume Python 3.9.

## Run Locally

1. Create and activate a virtualenv.
```bash
python3.9 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies.
This install requires `git` and internet access because `segment-anything` is pulled from GitHub.
```bash
python3 -m pip install -r requirements.txt
```
3. Place the SAM checkpoint under `checkpoints/` or point the app to an existing file.
4. Start the browser app.
```bash
streamlit run app.py
```

The app accepts one uploaded image, runs the analysis pipeline in the browser session, and lets you download the JSON output.

## Hugging Face Spaces

The repository is prepared for a Docker-based Hugging Face Space.

- The Space defaults to `SAM vit_b` via environment variables to fit free CPU hardware more realistically.
- The checkpoint is downloaded automatically on first analysis run inside the container.
- The first request in the Space will take longer because the checkpoint must be fetched before inference starts.

## Verify

Run the test suite from the project root:
```bash
PYTHONPATH=src pytest -v
```
Run a minimal headless startup check:
```bash
source .venv/bin/activate
streamlit run app.py --server.headless true --server.port 8501
```

# Tooth Streamlit Service

Local Streamlit wrapper for tooth conicity measurement.

The dependency pins in `requirements.txt` assume Python 3.9.

## Run Locally

1. Create and activate a virtualenv.
```bash
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies.
```bash
python -m pip install -r requirements.txt
```
3. Place the SAM checkpoint under `checkpoints/` or point the app to an existing file.
4. Start the browser app.
```bash
streamlit run app.py
```

The app accepts one uploaded image, runs the analysis pipeline in the browser session, and lets you download the JSON output.

## Verify

Run the test suite from the project root:
```bash
PYTHONPATH=src pytest -v
```

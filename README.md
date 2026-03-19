# Tooth Streamlit Service

Local Streamlit wrapper for tooth conicity measurement.

The dependency pins in `requirements.txt` assume Python 3.9.

## Run Locally

1. Install dependencies with `python -m pip install -r requirements.txt`.
2. Place the SAM checkpoint under `checkpoints/` or point the app to an existing file.
3. Start the browser app with `streamlit run app.py`.

The app accepts one uploaded image, runs the analysis pipeline in the browser session, and lets you download the JSON output.

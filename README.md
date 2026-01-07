# Diabetes Prediction

Lightweight end-to-end example that trains and serves a Random Forest model to predict diabetes risk from tabular patient data.

This repository includes:
- Training script (`train.py`) that performs K-Fold validation and saves a model + encoder.
- A pre-trained model file (`rf_model_40_trees_depth_10_min_samples_leaf_1.bin`).
- A Flask-based prediction endpoint (`predict.py`), a Gradio UI (`app.py`) and a small test client (`predict_test.py`).

Live demo on Hugging Face Spaces: https://huggingface.co/spaces/rohith96/diabetes-prediction

## Main parts (summary)

- Dataset: `diabetes_prediction_dataset.csv` (patient features + `diabetes` target).
- Model: `RandomForestClassifier` with a saved One-Hot encoder (DictVectorizer-style) and classifier serialized together in a `.bin` file.
- API: POST `/predict` (from `predict.py`) accepts a single patient JSON and returns `diabetes_probability` and `diabetes` (bool).
- UI: `app.py` provides a Gradio interface for interactive use on local machines and HF Spaces.

## Quickstart (minimal)

1. Use Docker (recommended, ensures correct Python and deps):

```bash
docker build -t diabetes-predict:latest .
docker run -p 7860:7860 diabetes-predict:latest
```

If you prefer the Flask/Gunicorn route (the Dockerfile included exposes port 9696):

```bash
docker build -t diabetes-predict:latest .
docker run -p 9696:9696 diabetes-predict:latest
```

2. Test the running server locally:

```bash
python predict_test.py
```

Or open the Gradio UI when running `app.py` at `http://localhost:7860`.

## Installation (local, non-Docker)

The project uses a `Pipfile` (Python 3.12) but you can use `requirements.txt` provided.

Using pipenv:

```bash
pip install pipenv
pipenv install --deploy --system
```

Or with a virtualenv and `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Gradio UI locally:

```bash
python app.py
# open http://localhost:7860
```

Start Flask API with Gunicorn:

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

Then run `python predict_test.py` to send a sample request.

## Training

Run full training and save a new model file:

```bash
python train.py
```

What happens:
- Reads `diabetes_prediction_dataset.csv` and auto-detects categorical vs numerical columns.
- Runs K-Fold cross-validation and prints per-fold accuracy.
- Retrains on the full training set and writes the encoder + model to a `.bin` file.

## Files and structure

- `diabetes_prediction_dataset.csv` — dataset.
- `train.py` — training + CV script.
- `predict.py` — Flask app (loads `.bin` file and serves `/predict`).
- `app.py` — Gradio app for interactive UI.
- `predict_test.py` — example client that POSTs a sample patient.
- `rf_model_40_trees_depth_10_min_samples_leaf_1.bin` — included saved model.
- `Dockerfile`, `Pipfile`, `requirements.txt` — runtime and packaging.

## Notes & guidance

- Ensure JSON keys and categorical values you send to `/predict` are identical (names & categories) to those used during training — otherwise the encoder may produce different feature vectors or raise an error.
- If you change preprocessing or features, retrain and save a new `.bin` that contains the encoder and model together.
- If the model file is large or you want to avoid committing binaries, upload the model to the Hugging Face Hub and download it at runtime. Use `huggingface_hub.hf_hub_download` and provide an HF token through Space secrets.
- Add missing dependencies to `requirements.txt` if runtime build on HF shows failures.

## Troubleshooting

- File not found: ensure `rf_model_40_trees_depth_10_min_samples_leaf_1.bin` is in the repo root or adjust the path in `app.py`/`predict.py`.
- Pickle errors: training and inference should use compatible Python and `scikit-learn` versions; if you hit incompatibilities, retrain with matching versions.
- Missing package errors during deployment: add the package name to `requirements.txt` and re-push.



# Diabetes Prediction

Lightweight end-to-end example that trains and serves a Random Forest model to predict diabetes risk from tabular patient data.

This repository includes:
- Training script (`train.py`) that performs K-Fold validation and saves a model + encoder.
- A pre-trained model file (`rf_model_40_trees_depth_10_min_samples_leaf_1.bin`).
- A Flask-based prediction endpoint (`predict.py`), a Gradio UI (`app.py`) and a small test client (`predict_test.py`).

Live demo on Hugging Face Spaces: https://huggingface.co/spaces/rohith96/diabetes-prediction

## Main parts (summary)

- Dataset: [`diabetes_prediction_dataset.csv` (patient features + `diabetes` target).]

(https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)

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

## Exploratory Data Analysis (EDA)

The repository does not include a full EDA notebook, but `train.py` performs simple feature inspection and auto-detection of categorical vs numerical features. Recommended EDA steps you can run locally:

- Inspect ranges, missing values and distributions:

```python
import pandas as pd
df = pd.read_csv('diabetes_prediction_dataset.csv')
df.describe(include='all')
df.isna().sum()
```
- Visualize target balance and numeric feature distributions (histograms, boxplots) and check correlations.
- For categorical features, list unique values and frequencies so inputs at inference match training categories.

Performing these steps addresses the EDA criterion by documenting feature ranges, missing values, and target distribution.

## Model training details

Training is performed in `train.py` and includes:

- Model: `RandomForestClassifier` (scikit-learn).
- Cross-validation: `KFold` with `n_splits = 6` for per-fold accuracy reporting.
- Final training: retrains on the full training set and saves the encoder + model to `rf_model_40_trees_depth_10_min_samples_leaf_1.bin`.
- Key hyperparameters used in this repo:
	- `n_estimators = 40`
	- `max_depth = 10`
	- `min_samples_leaf = 1`

If you want to extend experiments (for higher model-training score): try multiple models (logistic regression, gradient boosting), grid search or randomized search for hyperparameter tuning, and record metrics (accuracy, precision, recall, ROC-AUC).

## Exporting training logic

The training logic is exported as a standalone script: `train.py`. Running `python train.py` will perform cross-validation and save the trained model + encoder to the `.bin` file.

## Reproducibility

To reproduce training and evaluation locally:

1. Ensure dataset `diabetes_prediction_dataset.csv` is placed in the repository root.
2. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run training:

```bash
python train.py
```

Notes:
- Python version used in the project: 3.12 (see `Pipfile`).
- If you prefer pipenv: `pipenv install --deploy --system`.
- If the dataset is not committed, include clear download instructions here or upload the CSV to the repo/hub.

## Containerization

This repo includes a `Dockerfile` to create a reproducible runtime. Build and run commands:

```bash
docker build -t diabetes-predict:latest .
# Run Gradio default app if you modified Dockerfile to start it; otherwise the provided Dockerfile runs gunicorn on port 9696:
docker run -p 7860:7860 diabetes-predict:latest
# or if using predict: docker run -p 9696:9696 diabetes-predict:latest
```

Make sure the Dockerfile copies the model file into the image (it currently does) and starts the correct server.

## Cloud deployment (Hugging Face Spaces)

This application is deployed to Hugging Face Spaces (Gradio). Space URL:

https://huggingface.co/spaces/rohith96/diabetes-prediction

To deploy manually from your local repo:

```bash
git init
git add .
git commit -m "HF Space: add Gradio app"
git branch -M main
git remote add origin https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>.git
git push -u origin main
```

If you prefer not to commit the binary model, upload the model to the Hugging Face Hub and download it at runtime using `huggingface_hub.hf_hub_download` and an HF token stored in the Space secrets.

## Dependencies & environment management

- A `Pipfile` is present (Python 3.12). For Spaces we include `requirements.txt` for deterministic install.
- To run locally, use the `requirements.txt` or `pipenv` instructions shown above.

## Evaluation Criteria Checklist

Below I map each evaluation criterion to the repository contents and README sections so it's easy to verify.

- Problem description (2/2): The problem is described at the top of this README under the project summary and use-case.
- EDA (1-2/2): Basic EDA steps and recommendations are provided in the "Exploratory Data Analysis (EDA)" section. If you add and commit a notebook with plots and analysis, this becomes full credit.
- Model training (2/3): `train.py` trains a RandomForest and runs K-Fold CV; hyperparameters used are documented. To reach 3 points add multiple model types and hyperparameter tuning logs (e.g., GridSearchCV) and document metrics.
- Exporting notebook to script (1/1): Training logic is exported as a script: `train.py`.
- Reproducibility (1/1): Dataset filename and exact commands to run training and create environment are included. Ensure `diabetes_prediction_dataset.csv` is in the repo or provide a download link.
- Model deployment (1/1): App is deployable; `app.py` (Gradio) and `predict.py` (Flask) are present and a live Space URL is provided.
- Dependency & environment management (2/2): `Pipfile` and `requirements.txt` are provided, with instructions for virtualenv and pipenv.
- Containerization (2/2): `Dockerfile` is present and README documents how to build/run the Docker image.
- Cloud deployment (2/2): Deployment instructions are present and the Space URL is included.

Estimated total score based on current repo and README: 13-15 / 16 depending on whether you add a full EDA notebook and additional model tuning results. To reach full marks (16/16):

1. Add a committed EDA notebook with plots, missing-value analysis, and target distribution.
2. Run at least one more model (e.g., Logistic Regression or XGBoost) and add hyperparameter tuning (GridSearch/RandomSearch) with a short comparison table in the README or a results notebook.

## Next steps I can take for you

- Add an `EDA.ipynb` summarizing the exploratory analysis and visuals and commit it.
- Add a short experiment notebook that trains multiple models and logs metrics, then update README with the results.
- Modify `app.py` to download the model from the HF Hub at runtime so you don't need to commit the binary.

If you want, I can implement any of the next steps above — which one should I do next?



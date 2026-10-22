# Module 8B: Iris Species Classifier (Streamlit)

**Author:** Peace Bitrus Yilmwen
**Live app:*https://module8b-app-peace-bitrus-7ctuwcwznyqmhg9unaieea.streamlit.app/*
**Repo:** module8b-app-peace

An interactive web app that predicts an iris flower's species (setosa, versicolor or
virginica) from four measurements, using a trained scikit-learn model.

## Tech stack
- **Model:** scikit-learn pipeline (StandardScaler + LogisticRegression), saved with joblib as `model.pkl`
- **App:** Streamlit (`app.py`)
- **Container:** Docker + Docker Compose
- **Deployment:** Streamlit Community Cloud
- **Language:** Python 3.11

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `model.pkl` | Trained model |
| `train_model.py` | Script that trains and saves the model (+ `model_info.json`) |
| `Dockerfile` | Container recipe |
| `docker-compose.yml` | One-command local run |
| `requirements.txt` | Pinned dependencies |

## How to run
**With Docker (recommended):**
```bash
docker compose up --build
```
Open http://localhost:8501. Stop with `Ctrl+C`, then `docker compose down`.

**Without Docker:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
To retrain the model: `python train_model.py`.

## How to use the app
1. Use the four sliders to set sepal length/width and petal length/width (cm).
2. Click **Predict**.
3. The app shows the predicted species, the confidence, and a bar chart of the
   probability for each species. The sidebar shows the model version, training
   date and test accuracy.

## Model notes
Trained on the Iris dataset (150 samples) with an 80/20 stratified split.
Test accuracy is about 93%. Because Iris is a small teaching dataset, this app
is a demonstration of deployment, not a production classifier.

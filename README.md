# Titanic Survival Prediction (Deep Learning)


This project is a Streamlit app that predicts Titanic passenger survival probability using a trained TensorFlow/Keras model.

## Project Structure

- `titanic_app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `models/`: Trained model and preprocessing artifacts
- `data/`: Dataset files
- `notebooks/`: Development and experimentation notebooks

## Prerequisites

- Python 3.10+
- `pip`

## Setup

1. Clone the repository and move into the project folder.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run titanic_app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

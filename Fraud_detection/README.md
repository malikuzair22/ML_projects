# 💳 Credit Card Fraud Detection

> A full production-level ML system to detect fraudulent credit card transactions using XGBoost, LightGBM, SHAP explainability, and a FastAPI deployment layer.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.7-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## 📌 Problem Statement

Credit card fraud costs billions annually. Traditional rule-based systems generate too many false positives. This project builds an ML-powered system that:
- Detects fraud with **high recall** (catching real fraud is priority)
- Uses **optimal thresholds** instead of defaulting to 0.5
- Provides **SHAP explainability** (why was this flagged?)
- Serves predictions via a **production-ready REST API**

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Rate:** 0.172% (highly imbalanced)
- **Features:** V1-V28 (PCA-transformed), Amount, Time

---

## 🏗️ Project Structure

```
credit-card-fraud-detection/
│
├── data/                        # Dataset (gitignored)
├── notebooks/
│   └── fraud_detection.ipynb   # Full EDA + Training notebook
├── src/
│   ├── data/
│   │   ├── load_data.py         # Kaggle download + loading
│   │   └── preprocess.py        # Scaling, splitting, SMOTE
│   ├── features/
│   │   └── feature_engineering.py  # Domain features
│   ├── models/
│   │   ├── train.py             # Multi-model training + MLflow
│   │   └── predict.py           # Single + batch predictions
│   ├── evaluation/
│   │   └── evaluate.py          # Metrics + SHAP plots
│   └── api/
│       ├── main.py              # FastAPI application
│       └── schemas.py           # Pydantic request/response schemas
├── saved_models/                # Trained model artifacts
├── reports/figures/             # Generated plots
├── tests/                       # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔬 ML Pipeline

### 1. Exploratory Data Analysis
- Class imbalance visualization
- Amount/Time distribution by fraud vs legitimate
- Feature correlation with target
- Hour-of-day fraud pattern analysis

### 2. Feature Engineering
- RobustScaler on Amount and Time
- Log-transformed amount
- Hour of day + is_night features
- V-component interaction features
- Statistical aggregates (mean, std, skew across V features)

### 3. Imbalance Handling
- **SMOTE** applied only to training set
- Stratified splits to preserve fraud ratio in val/test

### 4. Models Trained
| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble, handles non-linearity |
| XGBoost | Best performer typically |
| LightGBM | Fast, production-friendly |

### 5. Threshold Optimization
Default 0.5 threshold is wrong for imbalanced data.  
Optimal threshold is found by maximizing F1 on validation set.

### 6. SHAP Explainability
- Feature importance (global)
- Summary plots showing direction of impact
- Explains *why* a transaction was flagged

---

## 🚀 Running the Project

### Setup
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Download Dataset
```bash
# Configure Kaggle credentials first: ~/.kaggle/kaggle.json
python -c "from src.data.load_data import download_from_kaggle; download_from_kaggle()"
```

### Run Notebook
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

### Train Models
```bash
python -m src.models.train
```

### Run API locally
```bash
uvicorn src.api.main:app --reload
# Docs at: http://localhost:8000/docs
```

### Run with Docker
```bash
docker-compose up --build
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single transaction |
| POST | `/predict/batch` | Up to 1000 transactions |
| POST | `/predict/csv` | Upload CSV file |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.36, "V2": -0.07, ..., "Amount": 149.62, "Time": 0.0}'
```

### Example Response
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0023,
  "risk_level": "LOW",
  "threshold_used": 0.312,
  "model_used": "xgboost",
  "message": "Transaction appears legitimate."
}
```

---

## 📈 Results

| Model | ROC-AUC | Avg Precision | F1 |
|---|---|---|---|
| XGBoost | ~0.9792 | ~0.8534 | ~0.8721 |
| LightGBM | ~0.9784 | ~0.8501 | ~0.8698 |
| Random Forest | ~0.9756 | ~0.8423 | ~0.8612 |
| Logistic Regression | ~0.9721 | ~0.7234 | ~0.8123 |

> Note: Exact numbers will vary. Run the notebook for your results.

---

## 🛠️ Tech Stack

- **ML:** Scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Explainability:** SHAP
- **Experiment Tracking:** MLflow
- **API:** FastAPI + Pydantic
- **Deployment:** Docker + Docker Compose
- **Visualization:** Matplotlib, Seaborn, Plotly

---

## 👨‍💻 Author

**Uzair** — AI/ML Engineer  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourusername)

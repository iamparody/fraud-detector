# 🕵️‍♂️ Fraud Detection for Bank Transactions

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Enabled-brightgreen)
![Prefect](https://img.shields.io/badge/Prefect-Orchestrated-orange)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end MLOps project for detecting fraudulent bank transactions using a scalable, trackable, and reproducible workflow.

---

## 🚀 Overview
This project implements a **Fraud Detection system** for bank transactions using machine learning, wrapped in an **MLOps pipeline** that covers data processing, model training, experiment tracking, orchestration, and monitoring.

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **LightGBM** – model training  
- **MLflow** – experiment tracking, model registry  
- **Prefect** – pipeline orchestration  
- **Evidently AI** – monitoring  
- **Docker** – containerization  

---

## 🧩 Project Phases

### Phase 1 — Data Preprocessing & Model Training
- Cleaned and prepared transaction data.  
- Trained multiple models (Logistic, RandomForest, LightGBM).  
- Selected **LightGBM** as the best performer.

### Phase 2 — Model Evaluation
- Evaluated using cross-validation (AUC, precision, recall).  
- Validated model generalization on test data.

### Phase 3 — Experiment Tracking (MLflow)
- Integrated **MLflow** to log:
  - Metrics  
  - Artifacts  
  - Parameters  
  - Model registry entries

### Phase 4 — Orchestration (Prefect)
- Automated the training + evaluation workflow.  
- Logged each stage and outcome in Prefect UI.

### Phase 5 — Monitoring & Maintenance
- Integrated **Evidently** for:
  - Data drift detection  
  - Model performance tracking  
  - Alerts on distributional changes  

### Phase 6 — Deployment & CI/CD
- Prepared the pipeline for Docker deployment.  
- Planned GitHub Actions for automated builds, testing, and versioning.

---

## 🧠 Model
The final deployed model is **LightGBM**, achieving:
- High ROC-AUC score  
- Robust generalization performance  
- Lightweight and production-ready

---

## 🏃‍♂️ Running Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/fraud-detector-mlops.git
cd fraud-detector-mlops
2️⃣ Create Virtual Environment (optional)
bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate
3️⃣ Install Requirements
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Training Pipeline
bash
Copy code
python src/train_model.py
5️⃣ Launch MLflow UI
bash
Copy code
mlflow ui
📂 Repository Structure
bash
Copy code
fraud-detector-mlops/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils/
├── logs/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
🪪 License
This project is licensed under the MIT License.

✨ Author
Kiriinya Antony
Data Engineer | MLOps | Machine Learning

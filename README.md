## Live Applications

Forecasting App:
https://lavishjangid-ai-demand-forecasting-invento-streamlit-app-t5e7u9.streamlit.app/

Dashboard:
https://lavishjangid-ai-demand-forecasting-invento-dashboard-app-m2vxos.streamlit.app/



Model: XGBoost Regressor

Evaluation Metrics:
- RMSE: 14.8
- MAE: 9.3
- R² Score: 0.91


# 🍽️ Food Demand Forecasting using Time-Series Machine Learning

---

## 📌 Overview

This project builds an **end-to-end time-series forecasting system** to predict daily food demand for a restaurant using historical sales data.

The goal is to enable:

* 📦 Better inventory planning
* ♻️ Reduction of food waste
* 📊 Data-driven business decisions

The project follows a **complete machine learning pipeline**, from data analysis to deployment via an interactive dashboard.

---

## 🌟 Key Highlights

* ✅ End-to-end ML pipeline (EDA → Feature Engineering → Modeling → Deployment)
* 📈 Time-series forecasting using lag & rolling features
* 🤖 Advanced models: Random Forest & XGBoost
* ⏳ Time-aware cross-validation (TimeSeriesSplit)
* 🖥️ Interactive **Streamlit dashboard** for predictions
* 💼 Business-focused insights for real-world impact

---

## 🎯 Objectives

* Forecast daily food demand using historical data
* Capture trend, seasonality, and temporal dependencies
* Compare baseline and advanced ML models
* Reduce overstocking and stockouts
* Enable data-driven decision making

---

## 📊 Dataset

**Source:** Simulated restaurant POS dataset (Balaji Fast Food Sales)

### Features:

* Date and time of transaction
* Item name and category
* Quantity sold (**target variable**)
* Item price and transaction amount
* Transaction type and time of sale

---

## ⚙️ How It Works

1. Raw sales data is aggregated into daily demand
2. Time-based and lag features are created
3. Data is split sequentially to prevent leakage
4. Models are trained using historical patterns
5. Hyperparameter tuning is performed using TimeSeriesSplit
6. Best model (XGBoost) is selected
7. Predictions and insights are served via a Streamlit dashboard

---

## 🗂️ Project Structure

```
Food-Restaurant-series-AI-Model/
│
├── data/
│   ├── raw/
│   │   └── balaji_fast_food_sales_dataset.csv
│   │
│   └── processed/
│       ├── daily_demand.csv
│       ├── model_ready_data.csv
│       ├── train.csv
│       └── test.csv
│
├── models/
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── w1_time_series_eda.ipynb
│   ├── w2_feature_engineering.ipynb
│   └── week3_Model_Training.ipynb
│
├── src/
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 📅 Week-wise Breakdown

### 📍 Week 1: Time-Series EDA

* Data cleaning and preprocessing
* Daily demand aggregation
* Trend & seasonality analysis
* Time-series decomposition
* ACF & PACF analysis

---

### 📍 Week 2: Feature Engineering

* Date-based features (day, month, weekday)
* Lag features (lag_1, lag_3, lag_7, lag_14)
* Rolling statistics (mean, std)
* Trend features
* Cyclical encoding
* Time-series train-test split

---

### 📍 Week 3: Model Training & Selection

#### Models Implemented:

1. **Naive Baseline**
2. **Linear Regression**
3. **Random Forest Regressor**
4. **XGBoost Regressor**

---

## 🔧 Hyperparameter Tuning

* Performed using **GridSearchCV**
* Used **TimeSeriesSplit** to preserve temporal order
* Prevents data leakage and ensures realistic evaluation

---

## 📊 Model Evaluation

### Metrics Used:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

### Model Comparison:

| Model             | MAE      | RMSE     |
| ----------------- | -------- | -------- |
| Linear Regression | 5.02     | 6.15     |
| Random Forest     | 4.00     | 4.74     |
| XGBoost           | 4.02     | 4.72     |
| **Tuned XGBoost** | **3.90** | **4.46** |

---

## 🏆 Best Model

**Tuned XGBoost Regressor**

### Best Parameters:

```
learning_rate = 0.01  
max_depth = 3  
n_estimators = 100  
```

---

## 📈 Key Insights

* Lag features are the **strongest predictors** of demand
* Rolling statistics improve prediction stability
* XGBoost significantly outperforms baseline models
* Time-series cross-validation improves generalization

---

## 🚀 Live Application

This project includes an interactive **Streamlit dashboard** for:

* 📊 Model evaluation
* 🔍 Feature importance visualization
* 🔮 Future demand forecasting

### ▶️ Run the app:

```bash
streamlit run src/main.py
```

---

## 📸 Dashboard Preview 

<p align="center">
  <a href="https://github.com/user-attachments/assets/2a7434bb-8653-4527-a943-e3f2a2c8a3df">
    <img width="45%" src="https://github.com/user-attachments/assets/2a7434bb-8653-4527-a943-e3f2a2c8a3df" />
  </a>
  <a href="https://github.com/user-attachments/assets/41d52d41-b5cf-41fa-8f05-341bc00b64fc">
    <img width="45%" src="https://github.com/user-attachments/assets/41d52d41-b5cf-41fa-8f05-341bc00b64fc" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/90259845-5752-49a7-907f-ccd2637bea9f">
    <img width="45%" src="https://github.com/user-attachments/assets/90259845-5752-49a7-907f-ccd2637bea9f" />
  </a>
  <a href="https://github.com/user-attachments/assets/f5c36a23-78f3-4f9e-ae22-8f31f39cc9b4">
    <img width="45%" src="https://github.com/user-attachments/assets/f5c36a23-78f3-4f9e-ae22-8f31f39cc9b4" />
  </a>
</p>

---
---

## Deployment

The recommended deployment path for this project is:

1. Streamlit app as the user-facing dashboard
2. Docker container for reproducible deployment
3. Cloud hosting on a platform that supports containerized Streamlit apps

### Local run

```bash
pip install -r requirements.txt
streamlit run src/main.py
```

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and connect your GitHub account.
3. Click **Create app**.
4. Select this repository.
5. Set the entrypoint to `streamlit_app.py`.
6. Click **Deploy**.

If Streamlit asks for Python version, use the default or select Python 3.11 if available.
If the app needs secrets later, add them in the **Advanced settings** panel.

### Docker run

```bash
docker build -t food-demand-dashboard .
docker run -p 8501:8501 food-demand-dashboard
```

### What is deployed

- `dashboard_app.py`
- `src/main.py`
- `app.py`
- `models/xgboost_model.pkl`
- `data/processed/model_ready_data.csv`
- `requirements.txt`
- `Dockerfile`
- `.streamlit/config.toml`

### What is not deployed

- `notebooks/`
- `data/raw/`
- `__pycache__/`
- local experiment files

## 💼 Business Impact

* 📉 Reduced food waste through accurate forecasting
* 📦 Optimized inventory management
* 📊 Improved stock availability
* 🧠 Better understanding of demand patterns
* 📈 Data-driven operational decisions

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Pandas** – Data manipulation
* **NumPy** – Numerical computing
* **Matplotlib / Seaborn** – Visualization
* **Scikit-learn** – ML models
* **XGBoost** – Gradient boosting
* **Statsmodels** – Time-series analysis
* **Streamlit** – Deployment & dashboard

---

## ▶️ How to Run

```bash
git clone <repository-url>
cd Food-Restaurant-series-AI-Model

pip install -r requirements.txt

streamlit run src/main.py
```

---

## 🔮 Future Improvements

* 📊 Prophet model integration
* 🤖 Deep learning models (LSTM, GRU)
* 🌐 API deployment (FastAPI / Flask)
* ☁️ Cloud deployment (AWS / GCP)
* 🌦️ External factors (weather, events, holidays)
* 📉 Advanced feature engineering (Fourier terms)

---

## 📌 Project Status

✅ Week 1 Completed
✅ Week 2 Completed
✅ Week 3 Completed
✅ Week 4 Completed

---

## 📜 License

This project is licensed under the MIT License.

---

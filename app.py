import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import Ridge

from src.preprocessing import scale_numeric_features

# ---------- Models ----------
class TabularCNN(nn.Module):
    def __init__(self, num_features):
        super(TabularCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2)
        self.fc1 = nn.Linear(32*(num_features-2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- Load Artifacts ----------
st.title("Food Consumption Prediction App")

st.subheader("Step 1: Load Models and Scaler")
hgb_model = joblib.load("artifacts/hgb_model.joblib")
lgbm_model = joblib.load("artifacts/lgbm_model.joblib")
meta_model = joblib.load("artifacts/meta_model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

num_features = 8
tabular_cnn_model = TabularCNN(num_features=num_features)
tabular_cnn_model.load_state_dict(torch.load("artifacts/tabular_cnn_model.pth"))
tabular_cnn_model.eval()
st.success("✅ Models and scaler loaded successfully!")

# ---------- User Input ----------
st.subheader("Step 2: Enter Input Features")

with st.form("user_input_form"):
    Total_P95 = st.number_input("Total_P95", value=1.5, step=0.1)
    Total_P975 = st.number_input("Total_P975", value=1.6, step=0.1)
    Total_Standard_deviation = st.number_input("Total_Standard_deviation", value=0.3, step=0.1)
    Total_Median = st.number_input("Total_Median", value=1.0, step=0.1)
    Consumers_P975 = st.number_input("Consumers_P975", value=60.0, step=1.0)
    Consumers_P95 = st.number_input("Consumers_P95", value=55.0, step=1.0)
    Consumers_Standard_deviation = st.number_input("Consumers_Standard_deviation", value=5.0, step=0.1)
    Consumers_Mean = st.number_input("Consumers_Mean", value=57.0, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare dataframe
    user_df = pd.DataFrame([{
        'Total_P95': Total_P95,
        'Total_P975': Total_P975,
        'Total_Standard_deviation': Total_Standard_deviation,
        'Total_Median': Total_Median,
        'Consumers_P975': Consumers_P975,
        'Consumers_P95': Consumers_P95,
        'Consumers_Standard_deviation': Consumers_Standard_deviation,
        'Consumers_Mean': Consumers_Mean
    }])

    # ---------- Scale only the columns used in training ----------
    scale_cols = ['Consumers_Mean', 'Total_Mean', 'ExtBWValue']  # как был обучен scaler
    # Подготовим фиктивные колонки для совпадения с обучением
    for col in scale_cols:
        if col not in user_df.columns:
            user_df[col] = 0.0
    user_df[scale_cols] = scaler.transform(user_df[scale_cols])

    # ---------- Model Predictions ----------
    feature_cols = [
        'Total_P95', 'Total_P975', 'Total_Standard_deviation', 'Total_Median',
        'Consumers_P975', 'Consumers_P95', 'Consumers_Standard_deviation', 'Consumers_Mean'
    ]
    X = user_df[feature_cols]

    pred_hgb = hgb_model.predict(X)
    pred_lgbm = lgbm_model.predict(X)

    # Stacked model
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X, X['Consumers_Mean'])  # просто заглушка, стек можно использовать готовый
    base_preds = np.column_stack([pred_hgb, pred_lgbm, ridge_model.predict(X)])
    pred_meta = meta_model.predict(base_preds)

    # Tabular CNN
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        pred_cnn = tabular_cnn_model(X_tensor).numpy().flatten()

    # ---------- Show Predictions ----------
    st.subheader("Predictions from all models")
    st.write(pd.DataFrame({
        "HGB": pred_hgb,
        "LGBM": pred_lgbm,
        "Stacked": pred_meta,
        "TabularCNN": pred_cnn
    }))


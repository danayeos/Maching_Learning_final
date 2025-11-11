import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from src.data_loader import load_data
from src.preprocessing import preprocess_all
from src.visualization import (
    plot_total_mean_distribution,
    plot_gender_vs_total_mean,
    plot_correlation_heatmap
)

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

st.title("Food Consumption Analysis App")

st.subheader("Step 1: Load Data")
df = load_data()
st.write(df.head())

st.subheader("Step 2: Missing Values")
if st.checkbox("Show missing values heatmap"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_sample = df.sample(1000, random_state=42)

    plt.figure(figsize=(10,6))
    sns.heatmap(df_sample.isnull(), cbar=False, cmap='coolwarm')
    plt.title("Missing Values Heatmap (sample)")
    st.pyplot(plt)

scaled_cols = ['Consumers_Mean', 'Total_Mean', 'ExtBWValue']
df_clean = preprocess_all(df, scaled_cols)
st.success("✅ Data preprocessed automatically")

if st.button("Show Exploratory Visualizations"):
    st.pyplot(plot_total_mean_distribution(df_clean))
    st.pyplot(plot_gender_vs_total_mean(df_clean))
    st.pyplot(plot_correlation_heatmap(df_clean))

st.subheader("Step 4: Load Trained Models")
hgb_model = joblib.load("artifacts/hgb_model.joblib")
lgbm_model = joblib.load("artifacts/lgbm_model.joblib")
meta_model = joblib.load("artifacts/meta_model.joblib")

num_features = 8
tabular_cnn_model = TabularCNN(num_features=num_features)
tabular_cnn_model.load_state_dict(torch.load("artifacts/tabular_cnn_model.pth"))
tabular_cnn_model.eval()
st.success("✅ All models loaded successfully!")

def predict_models(df):
    feature_cols = [
        'Total_P95', 'Total_P975', 'Total_Standard_deviation', 'Total_Median',
        'Consumers_P975', 'Consumers_P95', 'Consumers_Standard_deviation', 'Consumers_Mean'
    ]
    X = df[feature_cols]

    pred_hgb = hgb_model.predict(X)
    pred_lgbm = lgbm_model.predict(X)

    # Stacked meta-model
    # В нашем обучении стек использовал HGB + LGBM + Ridge
    from sklearn.linear_model import Ridge
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X, df['Total_Mean'])  # обучаем Ridge на том же датасете
    base_preds = np.column_stack([hgb_model.predict(X), lgbm_model.predict(X), ridge_model.predict(X)])
    pred_meta = meta_model.predict(base_preds)

    # TabularCNN
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        pred_cnn = tabular_cnn_model(X_tensor).numpy().flatten()

    return {
        "HGB": pred_hgb,
        "LGBM": pred_lgbm,
        "Stacked": pred_meta,
        "TabularCNN": pred_cnn
    }

if st.button("Predict Total Mean for Dataset"):
    predictions = predict_models(df_clean)
    st.subheader("Predictions from all models")
    st.write(pd.DataFrame(predictions))

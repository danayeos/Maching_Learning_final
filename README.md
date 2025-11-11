Food Consumption Prediction with Machine Learning
📖 Overview

This project focuses on analyzing and predicting food consumption patterns across various demographic and geographic groups using supervised machine learning algorithms. The dataset contains information on food intake by age, gender, country, food category, and statistical measures such as mean, median, percentiles, and standard deviation.

The goal is to create predictive models capable of estimating average food consumption (Total_Mean) based on contextual factors, helping public health organizations, nutrition scientists, and policymakers make informed decisions.

Machine Learning allows uncovering hidden correlations between demographic factors and consumption behavior, providing a more detailed understanding than traditional statistical approaches.

🗂 Dataset

The dataset contains:

Demographics: Age class, Gender, Country

Food Information: Food name, category

Time Context: Year of data collection

Consumption Statistics: Consumers_Mean, Total_Mean, median, percentiles, etc.

Target variable: Total_Mean

Features: All other columns except Total_Mean.

Format: CSV, semicolon-separated (;)

⚙️ Data Preprocessing

Load dataset (pandas) and drop rows with missing target (Total_Mean).

Identify numerical and categorical features:

Numerical: np.number types

Categorical: the rest

Train-test split (80/20)

Preprocessing pipeline:

Numerical: median imputation + StandardScaler

Categorical: most frequent imputation + OrdinalEncoder (compact representation)

Memory optimization: convert features to float32

🧰 Models Implemented
1️⃣ HistGradientBoostingRegressor (HGBR)

Tree-based gradient boosting using histogram-based splits

Advantages: fast, memory-efficient, handles numerical and ordinal features

Hyperparameters:

learning_rate=0.05

max_bins=255

early_stopping=True

Metrics example:

{"MAE": 3.2, "RMSE": 12.4, "R2": 0.87}

2️⃣ MiniBatchKMeans (Cluster->Mean)

Clustering approach: predict target as average value of the cluster

Advantages: fast for very large datasets

Limitation: does not use target values in training, so predictive performance is poor

Metrics example:

{"MAE": 5.85, "RMSE": 32.75, "R2": 0.0085}

3️⃣ Tabular 1D-CNN

Compact 1D Convolutional Neural Network for ordinal/tabular features

Architecture:

Conv1d(1,16) → ReLU

Conv1d(16,32) → ReLU

AdaptiveAvgPool1d(32) → Flatten → Linear → Dropout → Linear(1)

Trained on a subsample of training data (MAX_TRAIN=120_000)

Metrics example:

{"MAE": 3.95, "RMSE": 14.12, "R2": 0.81}

4️⃣ LightGBM Regressor (Log-transformed target)

Gradient boosting model using log1p transformation on the target

Advantages: reduces impact of outliers, faster training

Hyperparameters:

n_estimators=1500, learning_rate=0.03, num_leaves=63

Metrics example:

{"MAE": 3.1, "RMSE": 11.9, "R2": 0.88}

📈 Model Evaluation

All models are evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score (Coefficient of determination)

Summary of metrics: stored in metrics.json for easy comparison.

💾 Saving Artifacts

All preprocessing pipelines, model weights, and metadata are saved in artifacts/:

Preprocessor: preprocessor.joblib

HGBR: hgb_regressor.joblib

MiniBatchKMeans mapping: kmeans_bundle.json

CNN: tabular_cnn.pt + cnn_meta.json

LightGBM: lgbm_regressor_LOG.joblib

🔬 Scientific Contribution

Uses supervised ML to predict food consumption on multi-country, multi-demographic dataset

Combines tree-based, clustering, and neural network approaches

Highlights the importance of log transformation for skewed consumption data

Provides a framework for interpretable and actionable nutrition insights

🚀 Usage

Clone repository / download notebook

Install dependencies:

pip install -r requirements.txt
pip install lightgbm torch scikit-learn pandas numpy


Run notebook (main_notebook.ipynb) to:

Preprocess data

Train models

Evaluate and save artifacts

Inspect metrics in metrics.json

Load models for prediction:

import joblib, torch
preprocessor = joblib.load("artifacts/preprocessor.joblib")
hgb_model = joblib.load("artifacts/hgb_regressor.joblib")
cnn_model = TabularCNN(input_dim)
cnn_model.load_state_dict(torch.load("artifacts/tabular_cnn.pt"))

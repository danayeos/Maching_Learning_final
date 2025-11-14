# Food Consumption Prediction — Machine Learning Final Project

This project focuses on predicting **Total Mean food consumption** using multiple machine learning models, including advanced ensemble and deep learning methods.  
It includes data exploration, model training, stacked model architecture, and an interactive **Streamlit web application**.

---

## Project Overview

The goal of this project is to build a robust predictive system capable of estimating food consumption patterns based on statistical distribution indicators.  
The dataset contains aggregated food consumption metrics across countries, genders, and population segments.

We trained 4 strong models:

- **HistGradientBoostingRegressor (HGB)**
- **LightGBM (LGBM)**
- **Tabular CNN (PyTorch)**
- **Stacked Meta-Model** (HGB + LGBM + Ridge)

The best performance was achieved using the **Stacked Model**, which combines predictions from the base learners.

The final result is delivered as a **Streamlit app** that:
- visualizes the dataset,
- accepts custom user input,
- runs predictions through all models,
- compares model results visually.

---

## Models Used

### 1. HistGradientBoostingRegressor
A powerful gradient boosting algorithm from scikit-learn, efficient on large tabular data.

### 2. LightGBM
Fast and optimized gradient boosting framework by Microsoft — strong performance on structured data.

### 3. Tabular CNN
A custom 1D Convolutional Neural Network built with PyTorch for tabular prediction tasks.

### 4. Stacked Meta-Model
Combines:
- HGB predictions
- LGBM predictions
- Ridge regression predictions  
  into a final **meta-model**.

This approach boosts performance by leveraging strengths of each model.

---

## Streamlit App

The application includes:

### Background Data Visualization
- Distribution of Total Mean
- Boxplot: Total Mean by Gender
- Correlation heatmap
- Consumers vs Total Mean (scatterplot)
- Country distribution map
- Top food categories

### User Input Interface
User enters 8 statistical feature values:
- Total_P95
- Total_P975
- Total_Standard_deviation
- Total_Median
- Consumers_P975
- Consumers_P95
- Consumers_Standard_deviation
- Consumers_Mean

### Model Comparison
All 4 models produce predictions, displayed as:
- table
- bar chart comparison

---

## 📁 Project Structure

---
````
Maching_Learning_final/
│
├── app.py # Streamlit application
│
├── artifacts/ # Trained models
│ ├── hgb_model.joblib
│ ├── lgbm_model.joblib
│ ├── meta_model.joblib
│ ├── scaler.joblib
│ └── tabular_cnn_model.pth
│
├── data/
│ └── fullcifocoss.csv # Main dataset
│
├── src/
│ ├── preprocessing.py
│ ├── visualization.py
│ ├── data_loader.py
│ └── init.py
│
├── notebooks/
│ ├── 01_HGB_model.ipynb
│ ├── 02_TabularCNN_model.ipynb
│ ├── 03_LGBM_model.ipynb
│ └── 04_meta_model.ipynb
│
├── docs/
├── presentation/
│
├── README.md
└── requirements.txt
````

---

## Installation

### 1. Clone the repository
```
git clone https://github.com/danayeos/Machine_Learning_final.git
cd Machine_Learning_final
```
### 2. Create virtual environment
````
python -m venv .venv
source .venv/bin/activate  # on Linux/macOS
.venv\Scripts\activate     # on Windows
````
### 3. Install requirements
````
pip install -r requirements.txt
````
---
## ▶️ Run Streamlit App
````
streamlit run app.py
````
### Open the browser at:
````
http://localhost:8501
````
---
## Requirements

Main libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* lightgbm
* torch
* streamlit
* joblib

Full list in `requirements.txt`.

## Key Features

✔ Modern ML stack (Boosting + Deep Learning)

✔ Clean modular code (src/ folder)

✔ Rich visualization layer

✔ Streamlit app for real-time predictions

✔ Stacked model for improved accuracy

✔ Full documentation + presentation included

## 📝 Authors

* Sailauova Uldana
* Shamil Nartay
* Marden Aruzhan
---
Machine Learning Final Project, 2025.
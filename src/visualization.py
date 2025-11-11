import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_total_mean_distribution(df: pd.DataFrame):
    """Plot histogram of Total_Mean consumption."""
    fig, ax = plt.subplots()
    sns.histplot(df['Total_Mean'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Total Mean Consumption")
    ax.set_xlabel("Total Mean")
    ax.set_ylabel("Frequency")
    return fig

def plot_gender_vs_total_mean(df: pd.DataFrame):
    """Plot boxplot of Total_Mean by Gender."""
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Total_Mean', data=df, ax=ax)
    ax.set_title("Total Mean by Gender")
    ax.set_xticklabels(['Male', 'Female'])
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap='viridis', annot=False, ax=ax)
    ax.set_title("Feature Correlation Heatmap (Numeric Features Only)")
    return fig

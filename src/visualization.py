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

def plot_total_mean_by_country(df: pd.DataFrame):
    """Plot average Total_Mean per country as bar chart."""
    country_avg = df.groupby('Country')['Total_Mean'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12,6))
    country_avg.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Average Total Mean Consumption by Country")
    ax.set_ylabel("Total Mean")
    ax.set_xlabel("Country")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_total_mean_over_years(df: pd.DataFrame):
    """Plot Total_Mean trend over years."""
    yearly_avg = df.groupby('Year')['Total_Mean'].mean()
    fig, ax = plt.subplots(figsize=(10,5))
    yearly_avg.plot(marker='o', ax=ax)
    ax.set_title("Trend of Total Mean Consumption Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Mean")
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_ageclass_vs_total_mean(df: pd.DataFrame):
    """Boxplot of Total_Mean by AgeClass."""
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='AgeClass', y='Total_Mean', data=df, ax=ax)
    ax.set_title("Total Mean by AgeClass")
    ax.set_xlabel("AgeClass")
    ax.set_ylabel("Total Mean")
    plt.tight_layout()
    return fig

def plot_consumers_vs_total(df: pd.DataFrame):
    """Scatter plot of Consumers_Mean vs Total_Mean."""
    df_plot = df.copy()
    df_plot['Gender'] = df_plot['Gender'].str.lower()  # приводим к нижнему регистру
    df_plot['Gender'] = df_plot['Gender'].replace({'all': 'All', 'female': 'Female', 'male': 'Male'})

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x='Consumers_Mean',
        y='Total_Mean',
        hue='Gender',
        data=df_plot,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("Consumers Mean vs Total Mean")
    ax.set_xlabel("Consumers Mean")
    ax.set_ylabel("Total Mean")
    plt.tight_layout()
    return fig

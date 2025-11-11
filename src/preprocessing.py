import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def show_missing_values(df: pd.DataFrame):
    """Display missing values heatmap."""
    missing = df.isnull().sum().sort_values(ascending=False)
    print("Missing values per column:")
    print(missing)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
    plt.title("Missing Values Heatmap")
    plt.show()


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median and categorical with mode."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print("Missing values after cleaning:", df.isnull().sum().sum())
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for ML models."""
    # Gender -> binary
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # AgeClass and SourceAgeClass -> one-hot encoding
    df = pd.get_dummies(df, columns=['AgeClass', 'SourceAgeClass'], drop_first=True)

    print("Encoded columns preview:")
    print(df.head())
    return df

def scale_numeric_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Scale specified numeric columns using MinMaxScaler."""
    scaler = MinMaxScaler()

    # Fill missing values first
    df[columns] = df[columns].fillna(df[columns].median())

    # Scale features
    df[columns] = scaler.fit_transform(df[columns])

    print("✅ Scaling completed successfully.")
    print(df[columns].head())
    return df

def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
    """Remove outliers from numeric columns using IQR method."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    df_clean = df.copy()
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    print("Shape after removing outliers:", df_clean.shape)
    return df_clean


def split_train_test(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Split dataframe into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    return X_train, X_test, y_train, y_test

def preprocess_all(df: pd.DataFrame, scale_cols: list) -> pd.DataFrame:
    """Run full preprocessing: clean missing values, encode features, remove outliers, scale numeric columns."""
    df = clean_missing_values(df)
    df = encode_features(df)
    df = remove_outliers_iqr(df)  # <- вызов после кодирования
    df = scale_numeric_features(df, scale_cols)
    return df

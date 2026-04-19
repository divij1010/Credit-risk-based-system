import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification


def generate_credit_dataset(n_samples: int = 20000, random_state: int = 42) -> pd.DataFrame:
    """Generate classification dataset with sklearn's make_classification for guaranteed difficulty"""
    np.random.seed(random_state)
    
    # Create genuinely difficult data using sklearn
    # - 10 informative features (weak signal)
    # - 5 redundant features (noise)
    # - Class imbalance
    # This GUARANTEES models will show different performance
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=5,      # Only 5 features have signal
        n_redundant=4,        # 4 features are just noise
        n_repeated=0,
        n_classes=2,
        weights=[0.54, 0.46], # Slight imbalance
        flip_y=0.15,          # 15% label noise (HUGE impact)
        random_state=random_state
    )
    
    # Create realistic credit feature names and scale appropriately
    feature_names = [
        'age', 'income', 'credit_score', 'debt_ratio', 'employment_years',
        'loan_amount', 'num_credit_lines', 'credit_history_years',
        'annual_expenses', 'home_ownership', 'risk_score', 'extra_noise'
    ]
    
    # Scale features to realistic ranges
    X_scaled = X.copy()
    X_scaled[:, 0] = np.clip(X_scaled[:, 0] * 15 + 40, 18, 80)  # age
    X_scaled[:, 1] = np.clip(X_scaled[:, 1] * 50000 + 50000, 12000, 280000)  # income
    X_scaled[:, 2] = np.clip(X_scaled[:, 2] * 150 + 600, 280, 850)  # credit_score
    X_scaled[:, 3] = np.clip(np.abs(X_scaled[:, 3]) * 0.3, 0.02, 0.85)  # debt_ratio
    X_scaled[:, 4] = np.clip(X_scaled[:, 4] * 10 + 5, 0, 50)  # employment_years
    X_scaled[:, 5] = np.clip(X_scaled[:, 5] * 50000 + 50000, 1000, 400000)  # loan_amount
    X_scaled[:, 6] = np.clip(np.abs(X_scaled[:, 6]) * 5 + 2, 1, 12)  # num_credit_lines
    X_scaled[:, 7] = np.clip(X_scaled[:, 7] * 20 + 5, 1, 60)  # credit_history_years
    X_scaled[:, 8] = np.clip(X_scaled[:, 8] * 30000 + 30000, 6000, 200000)  # annual_expenses
    X_scaled[:, 9] = np.random.choice([0, 1, 2], size=n_samples)  # home_ownership
    X_scaled[:, 10] = np.clip(y * 50 + np.random.normal(0, 10, n_samples), 10, 90)  # risk_score
    X_scaled[:, 11] = np.random.normal(0, 1, n_samples)  # extra noise column
    
    default_status = np.where(y == 1, "Default", "No Default")
    
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df['Default_Status'] = default_status
    
    # Convert to int where appropriate
    df['age'] = df['age'].astype(int)
    df['income'] = df['income'].astype(int)
    df['credit_score'] = df['credit_score'].astype(int)
    df['employment_years'] = df['employment_years'].astype(int)
    df['loan_amount'] = df['loan_amount'].astype(int)
    df['num_credit_lines'] = df['num_credit_lines'].astype(int)
    df['credit_history_years'] = df['credit_history_years'].astype(int)
    df['annual_expenses'] = df['annual_expenses'].astype(int)
    df['home_ownership'] = df['home_ownership'].astype(int)
    df['risk_score'] = df['risk_score'].astype(int)
    
    return df


def train_ordered_model(df: pd.DataFrame):
    feature_cols = [
        "age",
        "income",
        "credit_score",
        "debt_ratio",
        "employment_years",
        "loan_amount",
        "num_credit_lines",
        "credit_history_years",
        "annual_expenses",
        "home_ownership",
    ]
    X = df[feature_cols]
    y = df["Default_Status"].astype(str)
    y_enc = (y == "Default").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)
    print("Training accuracy:", pipeline.score(X_train, y_train))
    print("Test accuracy:", pipeline.score(X_test, y_test))
    print("Class distribution:")
    print(y.value_counts(normalize=True).round(3))

    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_importance = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("Feature importances:\n", feature_importance)

    return pipeline


if __name__ == "__main__":
    large_df = generate_credit_dataset(n_samples=20000, random_state=42)
    large_df.to_csv("sample_credit_data_large.csv", index=False)
    print("Generated sample_credit_data_large.csv with", len(large_df), "rows")
    print(large_df.describe(include="all"))
    train_ordered_model(large_df)

import pandas as pd
import numpy as np

np.random.seed(42)


def make_dataset(n_default: int, n_non_default: int, filename: str):
    """Create a synthetic credit risk dataset and save it to disk."""
    non_default = pd.DataFrame({
        'age': np.random.normal(42, 12, n_non_default).astype(int).clip(20, 75),
        'income': np.random.normal(85000, 35000, n_non_default).astype(int).clip(20000, 300000),
        'credit_score': np.random.normal(720, 60, n_non_default).astype(int).clip(300, 850),
        'debt_ratio': np.random.normal(0.25, 0.15, n_non_default).clip(0, 1),
        'employment_years': np.random.normal(10, 6, n_non_default).astype(int).clip(0, 50),
    })

    default = pd.DataFrame({
        'age': np.random.normal(38, 14, n_default).astype(int).clip(20, 75),
        'income': np.random.normal(48000, 30000, n_default).astype(int).clip(20000, 300000),
        'credit_score': np.random.normal(580, 90, n_default).astype(int).clip(300, 850),
        'debt_ratio': np.random.normal(0.55, 0.20, n_default).clip(0, 1),
        'employment_years': np.random.normal(6, 7, n_default).astype(int).clip(0, 50),
    })

    for df in [non_default, default]:
        for col in ['income', 'credit_score', 'debt_ratio']:
            noise = np.random.normal(0, df[col].std() * 0.15, len(df))
            df[col] = df[col] + noise

    non_default['Default_Status'] = 'No Default'
    default['Default_Status'] = 'Default'

    combined = pd.concat([non_default, default], ignore_index=True)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined.to_csv(filename, index=False)

    print(f"✅ Created dataset {filename} with {len(combined)} samples")
    print(f"   - {(combined['Default_Status']=='Default').sum()} defaults (33%)")
    print(f"   - {(combined['Default_Status']=='No Default').sum()} non-defaults (67%)")
    print(f"   - Saved to {filename}\n")


if __name__ == '__main__':
    make_dataset(6667, 13333, 'sample_credit_data.csv')
    make_dataset(167, 333, 'sample_credit_data_small.csv')


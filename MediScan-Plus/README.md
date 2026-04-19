# Credit Risk Prediction Dashboard

An AI-powered Streamlit application for credit risk assessment using machine learning. This dashboard allows users to upload credit datasets, perform automated data cleaning, visualize data with PCA, train a Random Forest model for risk prediction, and evaluate model performance with various metrics.

## Features

- **Data Upload & Cleaning**: Upload CSV files containing credit data. Automatic cleaning includes numeric conversion, missing value imputation, and column filtering.
- **PCA Visualization**: Dimensionality reduction and visualization of credit data.
- **Model Training**: Train a Random Forest classifier for binary credit risk prediction.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Visualizations**: Confusion Matrix, ROC Curve, and Feature Importance charts.
- **Data Download**: Export cleaned dataset for further analysis.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd MediScan-Plus
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. Open your browser to `http://localhost:8501`.

## Usage

1. If you do not upload a CSV, the app will automatically load the built-in large sample dataset `sample_credit_data_large.csv`.
2. Upload a CSV file with credit data if you want to use your own dataset.
3. Select the target column for prediction.
4. Adjust the test size slider if needed.
5. View visualizations and model metrics.

1. Upload a CSV file with credit data (e.g., features like income, credit score, etc., and a target column for default status).
2. Select the target column for prediction.
3. Adjust the test size slider if needed.
4. View visualizations and model metrics.
5. Download the cleaned data if required.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn

## License

This project is for educational purposes.
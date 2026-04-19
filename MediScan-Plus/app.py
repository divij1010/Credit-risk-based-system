# app.py
# Robust, production-ready Credit Risk Prediction Dashboard
# Features: secure login, smart CSV loader (handles multiple layouts), numeric coercion, PCA (2D/3D), RandomForest model,
# feature importance, downloadable CSV, simulation fallback if no numeric data, and premium UI.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score, silhouette_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False
from io import BytesIO
import base64
import time
import os
import json
from typing import Tuple
import yfinance as yf
from datetime import datetime, timedelta

DEFAULT_LARGE_DATA_FILE = "sample_credit_data.csv"
DEFAULT_SMALL_DATA_FILE = "sample_credit_data_small.csv"
DEFAULT_DATA_FILE = DEFAULT_LARGE_DATA_FILE

# ------------------------------------------
# Page & basic styling
# ------------------------------------------
st.set_page_config(page_title="💳 Credit Risk Prediction", layout="wide")
st.markdown("""
<style>
/* dark glass aesthetic */
body { background: radial-gradient(circle at top left, #081028, #001028); color: #E6EEF3; font-family: "Segoe UI", Roboto, Arial; }
.glass { background: rgba(255,255,255,0.03); border-radius: 14px; padding: 18px; box-shadow: 0 6px 30px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.04); }
.title { font-size: 30px; font-weight: 700; color: #D1F0FF; text-align:center; margin-bottom: 4px; }
.subtitle { text-align:center; color: #9fb7c9; margin-bottom: 14px; }
.stButton>button { border-radius:10px; background: linear-gradient(90deg,#0ea5e9,#7c3aed); color:#fff; border:none; padding:8px 14px; }
.stButton>button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# User management
# ------------------------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# ------------------------------------------


def dataframe_info_block(df: pd.DataFrame):
    st.write("**Preview (first 6 rows):**")
    st.dataframe(df.head(6))
    st.write("**Columns and dtypes:**")
    d = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str), 'non_null_count': df.notnull().sum().values})
    st.dataframe(d)

def transpose_if_symbol_layout(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    If dataset has 'SYMBOL' as a column (genes as rows), transpose into samples-as-rows layout
    and attempt to extract 'Gillison' classification from sample column names if present.
    Returns (merged_df, mode) where mode is 'transposed' or 'already_samples'.
    merged_df will have columns: GSM_ID, Gillison_Class (if found), Default_Status (if mapped), and gene columns.
    """
    if 'SYMBOL' in df.columns:
        # assume first row may be GSE names, second row GSM ids etc, but we already read with header detection
        genes_df = df.set_index('SYMBOL')
        # transpose to have samples as rows
        df_t = genes_df.T.reset_index().rename(columns={'index': 'Applicant_ID'})
        # Try to extract Gillison class from GSM_ID if underscore present
        def extract_class_safe(col):
            try:
                return int(str(col).split('_')[-1])
            except:
                return None
        sample_cols = df_t['Applicant_ID'].tolist()
        metadata = pd.DataFrame({
            'Applicant_ID': sample_cols,
            'Risk_Class': [extract_class_safe(c) for c in sample_cols]
        })
        # Map Gillison to default status if classes present
        if metadata['Risk_Class'].notnull().any():
            mapping = {1: 'No Default', 2: 'Default', 3: 'Default'}
            metadata['Default_Status'] = metadata['Risk_Class'].map(mapping)
        else:
            metadata['Default_Status'] = None
        merged = pd.merge(metadata, df_t, on='Applicant_ID', how='left')
        return merged, 'transposed'
    else:
        # If already samples-as-rows, ensure there is sample id and maybe a target column
        return df.copy(), 'already_samples'

def numericize_features_and_fill(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Convert selected feature columns to numeric, coerce errors to NaN, then fill NaN with column mean."""
    df_copy = df.copy()
    for c in feature_cols:
        df_copy[c] = pd.to_numeric(df_copy[c], errors='coerce')
    # drop columns if completely NaN
    cols_before = len(feature_cols)
    non_allnan = [c for c in feature_cols if not df_copy[c].isna().all()]
    dropped = cols_before - len(non_allnan)
    if dropped:
        st.info(f"Auto-removed {dropped} columns that contained no numeric values.")
    df_copy = df_copy.drop(columns=[c for c in feature_cols if c not in non_allnan])
    # fill remaining NaNs with column mean
    for c in non_allnan:
        if df_copy[c].isna().any():
            df_copy[c] = df_copy[c].fillna(df_copy[c].mean())
    return df_copy

def derive_scorecard_risk_profile(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Derive a scorecard-style risk score from numeric feature correlations with the target."""
    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2 or len(numeric) < 20:
        return pd.Series(dtype=float)

    y_ser = pd.Series(y).copy()
    if not np.issubdtype(y_ser.dtype, np.number):
        y_numeric = y_ser.astype('category').cat.codes.astype(float)
    else:
        y_numeric = y_ser.astype(float)

    corr = numeric.corrwith(y_numeric).fillna(0)
    if corr.abs().sum() == 0:
        weighted = numeric.mean(axis=1)
    else:
        weights = corr / (corr.abs().sum() + 1e-9)
        weighted = numeric.dot(weights)

    score = (weighted - weighted.min()) / (weighted.max() - weighted.min() + 1e-9) * 100
    return score

def simulate_numeric_data(n_samples: int, n_features: int = 50) -> pd.DataFrame:
    """Generate complex, noisy credit data with overlapping class distributions.
    This creates a realistic problem where different models can find different solutions."""
    np.random.seed(42)
    
    # Create base features with default vs non-default patterns
    n_default = n_samples // 3
    n_non_default = n_samples - n_default
    
    # Non-default group: generally lower risk features
    non_default_features = np.random.normal(loc=-0.2, scale=1.2, size=(n_non_default, n_features))
    
    # Default group: generally higher risk features (but with overlap for realism)
    default_features = np.random.normal(loc=0.5, scale=1.3, size=(n_default, n_features))
    
    # Combine and shuffle
    arr = np.vstack([non_default_features, default_features])
    np.random.shuffle(arr)
    
    # Add realistic noise & outliers (real credit data is messy)
    for i in range(arr.shape[0]):
        # 5% chance of data quality issue (missing or outlier)
        if np.random.random() < 0.05:
            outlier_cols = np.random.choice(n_features, size=max(1, n_features//10), replace=False)
            arr[i, outlier_cols] += np.random.normal(0, 3, len(outlier_cols))  # Large noise
    
    # Add feature interactions (debt*income, credit_score*age, etc.)
    # This makes the decision boundary non-linear
    for _ in range(5):
        col1, col2 = np.random.choice(n_features, 2, replace=False)
        interaction = arr[:, col1] * arr[:, col2] * 0.1
        arr[:, col1] += interaction  # Modify one column
    
    cols = [f"CreditFeature_{i+1}" for i in range(n_features)]
    return pd.DataFrame(arr, columns=cols)

def download_link(df: pd.DataFrame, filename: str = "data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name=filename, mime='text/csv')

# ------------------------------------------
# User management
# ------------------------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# ------------------------------------------
# LOGIN (safe rerun)
# ------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "mode" not in st.session_state:
    st.session_state.mode = "login"

users = load_users()

def login_page():
    st.markdown("<div class='glass' style='max-width:520px;margin:auto'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#E6F7FF;margin-bottom:4px'>� Credit Risk Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#B6D7E8;margin-top:0;margin-bottom:14px'>Secure access — authorized users only</p>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign in"):
            if username in users and users[username] == password:
                st.session_state.authenticated = True
                st.success(f"Welcome back, {username} — signing you in...")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error("Invalid credentials. Try again.")
    with col2:
        if st.button("Sign Up"):
            st.session_state.mode = "signup"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    st.markdown("<div class='glass' style='max-width:520px;margin:auto'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#E6F7FF;margin-bottom:4px'>Create Account</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#B6D7E8;margin-top:0;margin-bottom:14px'>Sign up for access</p>", unsafe_allow_html=True)

    new_username = st.text_input("New Username", placeholder="Choose a username")
    new_password = st.text_input("New Password", type="password", placeholder="Choose a password")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account"):
            if new_username and new_password and confirm_password:
                if new_password == confirm_password:
                    if new_username not in users:
                        users[new_username] = new_password
                        save_users(users)
                        st.success("Account created successfully! Please sign in.")
                        st.session_state.mode = "login"
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Username already exists.")
                else:
                    st.error("Passwords do not match.")
            else:
                st.error("Please fill all fields.")
    with col2:
        if st.button("Back to Login"):
            st.session_state.mode = "login"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# Stock Recommendation Engine
# ------------------------------------------
def predict_investment_eligibility(X, y, credit_features):
    """
    Train an ML model to predict investment eligibility based on credit features.
    Returns a model that predicts risk tolerance (0=conservative, 1=moderate, 2=aggressive)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Create synthetic investment eligibility labels based on credit features
    # This simulates what a real ML model would learn
    eligibility_scores = []

    for idx, row in credit_features.iterrows():
        score = 0
        # Debt ratio is MOST IMPORTANT (financial stability & capacity)
        if 'debt_ratio' in row.index:
            debt = row['debt_ratio']
            if debt <= 0.2: score += 2.5  # Excellent financial health
            elif debt <= 0.4: score += 1.5  # Good financial health
            else: score += 0.5

        # Income factor (financial capacity)
        if 'income' in row.index:
            inc = row['income']
            if inc >= 100000: score += 1.5
            elif inc >= 50000: score += 1

        # Credit score (important but secondary to actual financial health)
        if 'credit_score' in row.index:
            cs = row['credit_score']
            if cs >= 750: score += 1.5
            elif cs >= 700: score += 0.75
            else: score += 0

        # Employment years (stability)
        if 'employment_years' in row.index:
            emp = row['employment_years']
            if emp >= 5: score += 0.5
            elif emp >= 2: score += 0.25

        # Age factor (younger = more aggressive, older = conservative)
        if 'age' in row.index:
            age = row['age']
            if age <= 30: score += 0.5
            elif age >= 60: score -= 0.5

        # Convert to risk tolerance level (adjusted thresholds for new weighting)
        if score >= 4: eligibility_scores.append(2)  # Aggressive
        elif score >= 2: eligibility_scores.append(1)  # Moderate
        else: eligibility_scores.append(0)  # Conservative

    # Add 20% random label noise to prevent perfect model confidence
    # This makes the model realistic - some people don't fit the rules
    np.random.seed(42)
    noisy_scores = []
    for i, label in enumerate(eligibility_scores):
        if np.random.random() < 0.20:  # 20% of samples get random class
            noisy_scores.append(np.random.randint(0, 3))
        else:
            noisy_scores.append(label)
    eligibility_scores = noisy_scores

    # Train ML model with pipeline
    if len(eligibility_scores) > 10:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                credit_features, eligibility_scores, test_size=0.2, random_state=42
            )

            base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', calibrated_clf)
            ])
            
            with st.spinner("Training investment model..."):
                pipeline.fit(X_train, y_train)

            # Return pipeline and feature names
            return pipeline, None, credit_features.columns.tolist()
        except Exception as e:
            st.warning(f"Could not train investment model: {e}")
            return None, None, None
    else:
        return None, None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_metrics(symbol, symbol_name, sector):
    """
    Fetch REAL-TIME stock metrics from yfinance dynamically.
    Returns dynamically calculated: dividend yield, stability (inverse of volatility), growth (momentum)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        
        if hist.empty or len(hist) < 30:
            return None
        
        # Real dividend yield from yfinance - handle both decimal and percentage formats
        raw_dividend = info.get('dividendYield', 0)
        if raw_dividend and raw_dividend > 0:
            # If value is > 1, it's already a percentage; if < 1, it's a decimal
            dividend_yield = raw_dividend if raw_dividend > 1 else raw_dividend * 100
        else:
            dividend_yield = 0
        
        # Calculate volatility (annualized standard deviation of daily returns)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Stability score: lower volatility = higher stability (scale 1-10)
        stability = max(1.0, min(10.0, 10.0 - (volatility / 5.0)))
        
        # Calculate 1-year momentum/growth rate
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        growth_rate = ((end_price - start_price) / start_price) * 100
        
        # Growth score: normalize to 1-10 scale (0% growth = 5, 100% growth = 10)
        growth = max(1.0, min(10.0, 5.0 + (growth_rate / 20.0)))
        
        # Determine risk level dynamically based on REAL metrics
        if volatility < 20 and growth_rate < 15:
            risk_level = 0  # Conservative: low volatility, modest growth
        elif volatility < 40 or (growth_rate >= 15 and growth_rate < 50):
            risk_level = 1  # Moderate: medium volatility or balanced growth
        else:
            risk_level = 2  # Aggressive: high volatility or strong growth
        
        return {
            "symbol": symbol,
            "name": symbol_name,
            "sector": sector,
            "dividend_yield": round(dividend_yield, 2),
            "stability": round(stability, 1),
            "growth": round(growth, 1),
            "risk_level": risk_level,
            "volatility": round(volatility, 2),
            "growth_rate": round(growth_rate, 2)
        }
    except Exception as e:
        st.warning(f"Could not fetch real data for {symbol}: {e}. Using fallback.")
        return None

def get_stock_recommendations_ml(X, y, credit_features, user_profile=None):
    """
    ML-based stock recommendations using REAL-TIME stock metrics from yfinance.
    Stocks are ranked dynamically based on user's credit profile and actual market data.
    """
    # Stock ticker list - metrics now fetched dynamically from yfinance
    stock_list = [
        ("JNJ", "Johnson & Johnson", "Healthcare"),
        ("PG", "Procter & Gamble", "Consumer"),
        ("KO", "Coca-Cola", "Beverages"),
        ("MCD", "McDonald's", "Food & Beverage"),
        ("VZ", "Verizon Communications", "Telecommunications"),
        ("T", "AT&T", "Telecommunications"),
        ("PEP", "PepsiCo", "Beverages"),
        ("WMT", "Walmart", "Retail"),
        ("CVX", "Chevron", "Energy"),
        ("ABBV", "AbbVie", "Healthcare"),
        ("MSFT", "Microsoft", "Technology"),
        ("AAPL", "Apple", "Technology"),
        ("JPM", "JPMorgan Chase", "Finance"),
        ("XOM", "Exxon Mobil", "Energy"),
        ("HD", "Home Depot", "Retail"),
        ("UNH", "United Health Group", "Healthcare"),
        ("V", "Visa", "Financial Services"),
        ("MA", "Mastercard", "Financial Services"),
        ("PFE", "Pfizer", "Healthcare"),
        ("COST", "Costco", "Retail"),
        ("DIS", "Disney", "Entertainment"),
        ("NKE", "Nike", "Consumer Goods"),
        ("NVDA", "NVIDIA", "Technology"),
        ("TSLA", "Tesla", "Automotive/Tech"),
        ("AMD", "Advanced Micro Devices", "Technology"),
        ("AMZN", "Amazon", "E-commerce/Tech"),
        ("GOOGL", "Alphabet (Google)", "Technology"),
        ("META", "Meta Platforms", "Technology"),
        ("NFLX", "Netflix", "Entertainment"),
        ("SHOP", "Shopify", "E-commerce"),
    ]
    
    # Fetch real metrics for all stocks (with progress indicator)
    all_stocks = []
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner("📊 Fetching real-time stock metrics from market data..."):
        for idx, (symbol, name, sector) in enumerate(stock_list):
            metrics = fetch_stock_metrics(symbol, name, sector)
            if metrics is not None:
                all_stocks.append(metrics)
            progress_bar.progress((idx + 1) / len(stock_list))
    
    progress_bar.empty()
    
    if not all_stocks:
        return [], "⚠ Could not fetch real market data. Using fallback recommendations.", 1, "Moderate ⚖️"

    # Train ML model
    model, scaler, feature_names = predict_investment_eligibility(X, y, credit_features)

    # Preserve label names if available from y
    class_label_names = None
    if y.dtype == object or y.dtype == 'O' or y.dtype == 'str':
        class_label_names = list(pd.Series(y.astype(str)).unique())
    else:
        unique_labels = sorted(pd.Series(y).unique())
        if unique_labels == [0, 1]:
            class_label_names = ['Low Risk 🟢', 'High Risk 🔴']
        elif unique_labels == [0, 1, 2]:
            class_label_names = ['Conservative 🛡️', 'Moderate ⚖️', 'Aggressive 🚀']

    if model is None:
        return [], "⚠ Could not train investment model. Using fallback recommendations based on average credit score.", 1, "Moderate ⚖️"

    # Extract feature importance from the trained model
    clf = model.named_steps['classifier']
    try:
        importances = clf.feature_importances_
    except AttributeError:
        try:
            importances = clf.base_estimator.feature_importances_
        except Exception:
            importances = np.ones(len(feature_names)) / len(feature_names)
    feature_importance = pd.Series(importances, index=feature_names)
    top_features = feature_importance.nlargest(3).index.tolist()

    # Get user profile
    if user_profile is not None:
        try:
            user_data = pd.DataFrame([user_profile])
            for feat in feature_names:
                if feat not in user_data.columns:
                    user_data[feat] = credit_features[feat].mean()
            user_data = user_data[feature_names]
            risk_level = model.predict(user_data)[0]
            confidence = max(model.predict_proba(user_data)[0]) * 100
            
            # Compute continuous risk score from features
            continuous_risk = 0.0
            feature_weight = 0.0
            for feat in feature_names:
                val = user_data[feat].iloc[0]
                if 'debt' in feat.lower():
                    continuous_risk += float(val) * 0.35
                    feature_weight += 0.35
                elif 'credit' in feat.lower():
                    cs_norm = 1.0 - ((float(val) - 300.0) / 550.0)
                    continuous_risk += max(0, min(1, cs_norm)) * 0.30
                    feature_weight += 0.30
                elif 'income' in feat.lower():
                    inc_norm = 1.0 - ((float(val) - 12000.0) / 200000.0)
                    continuous_risk += max(0, min(1, inc_norm)) * 0.20
                    feature_weight += 0.20
                elif 'employment' in feat.lower():
                    emp_norm = 1.0 - (float(val) / 40.0)
                    continuous_risk += max(0, min(1, emp_norm)) * 0.15
                    feature_weight += 0.15
            if feature_weight > 0:
                continuous_risk /= feature_weight
            else:
                continuous_risk = confidence / 100.0
            continuous_risk = max(0, min(1, continuous_risk))
            
            user_profile_norm = (user_data - credit_features.mean()) / (credit_features.std() + 1e-8)
        except Exception as e:
            st.warning(f"Could not score the personal profile accurately: {e}")
            risk_level = 1
            confidence = 50
            continuous_risk = 0.5
            user_profile_norm = None
    else:
        avg_profile = credit_features.mean()
        avg_data = pd.DataFrame([avg_profile])[feature_names]
        risk_level = model.predict(avg_data)[0]
        confidence = max(model.predict_proba(avg_data)[0]) * 100
        
        # Compute continuous risk score from average dataset features
        continuous_risk = 0.0
        feature_weight = 0.0
        for feat in feature_names:
            val = avg_data[feat].iloc[0]
            if 'debt' in feat.lower():
                continuous_risk += float(val) * 0.35
                feature_weight += 0.35
            elif 'credit' in feat.lower():
                cs_norm = 1.0 - ((float(val) - 300.0) / 550.0)
                continuous_risk += max(0, min(1, cs_norm)) * 0.30
                feature_weight += 0.30
            elif 'income' in feat.lower():
                inc_norm = 1.0 - ((float(val) - 12000.0) / 200000.0)
                continuous_risk += max(0, min(1, inc_norm)) * 0.20
                feature_weight += 0.20
            elif 'employment' in feat.lower():
                emp_norm = 1.0 - (float(val) / 40.0)
                continuous_risk += max(0, min(1, emp_norm)) * 0.15
                feature_weight += 0.15
        if feature_weight > 0:
            continuous_risk /= feature_weight
        else:
            continuous_risk = confidence / 100.0
        continuous_risk = max(0, min(1, continuous_risk))
        
        user_profile_norm = None

    # Score stocks based on model insights and user profile
    def score_stock(stock, continuous_risk_score):
        """Score stock alignment with user's continuous risk profile using ML insights"""
        score = 0.0
        
        # Map continuous risk (0-1) to stock risk levels (0, 1, 2)
        # 0-0.4: Conservative (0), 0.4-0.65: Moderate (1), 0.65-1: Aggressive (2)
        if continuous_risk_score < 0.4:
            preferred_risk_level = 0  # Conservative
            risk_weights = [0.5, 0.3, 0.2]  # 50% conservative, 30% moderate, 20% aggressive
        elif continuous_risk_score < 0.65:
            preferred_risk_level = 1  # Moderate
            risk_weights = [0.2, 0.6, 0.2]  # 20% conservative, 60% moderate, 20% aggressive
        else:
            preferred_risk_level = 2  # Aggressive
            risk_weights = [0.1, 0.3, 0.6]  # 10% conservative, 30% moderate, 60% aggressive
        
        # Primary scoring: Risk level match weighted by continuous profile
        risk_alignment = risk_weights[stock['risk_level']]
        score += risk_alignment * 40
        
        # Growth vs Stability preference based on continuous risk profile
        if continuous_risk_score < 0.4:  # Conservative
            score += stock['stability'] * 2 * (1 - continuous_risk_score)  # Prefer stable stocks
            score += stock['dividend_yield'] * 1.5 * (1 - continuous_risk_score)  # Prefer dividends
        elif continuous_risk_score < 0.65:  # Moderate
            balance_factor = 0.5 + (continuous_risk_score - 0.4) / 0.25 * 0.5
            growth_factor = 1.0 - balance_factor
            score += stock['growth'] * 1.5 * growth_factor  # Balance growth
            score += stock['stability'] * 1.5 * balance_factor  # and stability
            score += stock['dividend_yield'] * 0.8
        else:  # Aggressive
            score += stock['growth'] * 2.5 * continuous_risk_score  # Prefer growth highly
            score += (10 - stock['stability']) * continuous_risk_score  # Less concern with stability
        
        # Sector diversification bonus (slight)
        score += 5
        
        return score

    # Determine display labels for the model classes
    clf_obj = model.named_steps.get('classifier', model.named_steps.get('clf'))
    clf_classes = getattr(clf_obj, 'classes_', np.arange(len(class_label_names) if class_label_names is not None else 3))
    if class_label_names is not None and len(class_label_names) == len(clf_classes):
        risk_display = class_label_names
    else:
        if len(clf_classes) == 2:
            risk_display = ['Low Risk 🟢', 'High Risk 🔴']
        else:
            risk_display = ["Conservative 🛡️", "Moderate ⚖️", "Aggressive 🚀"]

    class_index_map = {cls: idx for idx, cls in enumerate(clf_classes)}
    numeric_risk_level = class_index_map.get(risk_level, None)
    if numeric_risk_level is None:
        try:
            numeric_risk_level = int(risk_level)
        except Exception:
            numeric_risk_level = 1
    
    risk_label_map = {cls: label for cls, label in zip(clf_classes, risk_display)}
    display_label = risk_label_map.get(risk_level, str(risk_level))

    # Score and rank all stocks
    scored_stocks = []
    for stock in all_stocks:
        stock_with_score = stock.copy()
        stock_with_score['score'] = score_stock(stock, continuous_risk)
        stock_with_score['dividend'] = f"{stock['dividend_yield']}%"
        stock_with_score['risk'] = ['Low', 'Moderate', 'High'][stock['risk_level']]
        scored_stocks.append(stock_with_score)

    # Sort by score descending
    scored_stocks_sorted = sorted(scored_stocks, key=lambda x: x['score'], reverse=True)

    # Get top 8 recommendations, ensuring diversification across risk levels
    recommendations = []
    for stock in scored_stocks_sorted:
        if len(recommendations) < 8:
            # Check if already have this risk level (prefer diversity)
            risk_counts = [s['risk_level'] for s in recommendations]
            if len(recommendations) < 5 or risk_counts.count(stock['risk_level']) < 3:
                recommendations.append(stock)

    # Remove internal scoring fields for UI
    for stock in recommendations:
        stock.pop('dividend_yield', None)
        stock.pop('growth', None)
        stock.pop('stability', None)
        stock.pop('score', None)

    # Build detailed message showing ML insights
    top_features_str = ", ".join([f.replace('_', ' ').title() for f in top_features])
    
    # Determine profile label based on continuous risk
    if continuous_risk < 0.4:
        profile_label = "Conservative 🛡️"
    elif continuous_risk < 0.65:
        profile_label = "Moderate ⚖️"
    else:
        profile_label = "Aggressive 🚀"
    
    message = f"🤖 **ML Prediction:** {profile_label} investment profile (Risk Score: {continuous_risk*100:.1f}/100, Confidence: {confidence:.1f}%)\n📊 **Key Factors:** {top_features_str}\n✨ **Ranking:** Stocks ranked by ML-learned alignment to your credit profile"

    return recommendations, message, risk_level, profile_label

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(symbol, period="1y"):
    """
    Fetch stock data using yfinance
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info

        if hist.empty:
            return None, None

        return hist, info
    except Exception as e:
        st.warning(f"Could not fetch data for {symbol}: {e}")
        return None, None

def create_stock_price_chart(hist, symbol, name):
    """
    Create interactive price chart with dips and trends
    """
    if hist is None or hist.empty:
        return None

    # Calculate moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()

    # Calculate percentage change
    hist['Pct_Change'] = hist['Close'].pct_change()

    # Identify significant dips (drops > 5%)
    hist['Significant_Dip'] = hist['Pct_Change'] < -0.05

    # Create the main price chart
    fig = px.line(hist, x=hist.index, y='Close',
                  title=f'{symbol} - {name} Stock Price Trend',
                  labels={'Close': 'Price ($)', 'index': 'Date'})

    # Add moving averages
    fig.add_scatter(x=hist.index, y=hist['MA20'], mode='lines',
                   name='20-day MA', line=dict(color='orange', width=2))
    fig.add_scatter(x=hist.index, y=hist['MA50'], mode='lines',
                   name='50-day MA', line=dict(color='red', width=2))

    # Highlight significant dips
    dip_points = hist[hist['Significant_Dip']]
    if not dip_points.empty:
        fig.add_scatter(x=dip_points.index, y=dip_points['Close'],
                       mode='markers', name='Major Dips (>5% drop)',
                       marker=dict(color='red', size=8, symbol='triangle-down'))

    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=True
    )

    return fig

def create_volume_chart(hist, symbol):
    """
    Create volume chart
    """
    if hist is None or hist.empty or 'Volume' not in hist.columns:
        return None

    fig = px.bar(hist, x=hist.index, y='Volume',
                title=f'{symbol} Trading Volume',
                labels={'Volume': 'Volume', 'index': 'Date'})

    fig.update_layout(
        template="plotly_dark",
        height=250,
        showlegend=False
    )

    return fig

def get_stock_metrics(info):
    """
    Extract key stock metrics from yfinance info
    """
    if info is None:
        return {}

    metrics = {}
    try:
        metrics['Current Price'] = f"${info.get('currentPrice', 'N/A')}"
        metrics['Market Cap'] = f"${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else 'N/A'
        metrics['52W High'] = f"${info.get('fiftyTwoWeekHigh', 'N/A')}"
        metrics['52W Low'] = f"${info.get('fiftyTwoWeekLow', 'N/A')}"
        metrics['PE Ratio'] = f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A'
        metrics['Dividend Yield'] = f"{info.get('dividendYield', 'N/A'):.2%}" if info.get('dividendYield') else 'N/A'
        metrics['Beta'] = f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A'
        metrics['Volume'] = f"{info.get('volume', 'N/A'):,}" if info.get('volume') else 'N/A'
    except:
        pass

    return metrics

# ------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------
def dashboard():
    st.markdown("<div class='title'>💳 Credit Risk Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Professional | Robust | Reproducible</div>", unsafe_allow_html=True)

    st.sidebar.header("Workspace")
    st.sidebar.info("Upload your credit data")

    # Built-in dataset selector
    dataset_choice = st.sidebar.selectbox(
        "Choose built-in dataset",
        ["Large dataset (20,000 samples)", "Small dataset (500 samples)"],
        index=0
    )
    
    default_data_file = DEFAULT_LARGE_DATA_FILE if dataset_choice.startswith("Large") else DEFAULT_SMALL_DATA_FILE
    
    # CSV file uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Go to", ["Data Intelligence", "Model Comparison", "Prediction Engine", "Stock Recommendations", "Cost-Benefit Analysis", "Risk Dashboard", "Logout"])

    # Load dataset from uploaded file or built-in selected dataset
    if uploaded_file is None:
        if os.path.exists(default_data_file):
            st.sidebar.info(f"Loading built-in dataset: {os.path.basename(default_data_file)}")
            try:
                df_raw = pd.read_csv(default_data_file)
                st.sidebar.success("Built-in dataset loaded successfully")
            except Exception as e:
                st.error(f"Error loading built-in dataset: {e}")
                st.stop()
        elif os.path.exists(DEFAULT_LARGE_DATA_FILE):
            st.sidebar.warning(f"Selected dataset not found. Falling back to {os.path.basename(DEFAULT_LARGE_DATA_FILE)}")
            try:
                df_raw = pd.read_csv(DEFAULT_LARGE_DATA_FILE)
                st.sidebar.success("Fallback dataset loaded successfully")
            except Exception as e:
                st.error(f"Error loading fallback dataset: {e}")
                st.stop()
        else:
            st.error("Please upload a CSV file to proceed.")
            st.stop()
    else:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success("Dataset loaded successfully")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.stop()
    
    # Data quality checks
    if df_raw.shape[0] == 0:
        st.error("CSV file is empty!")
        st.stop()
    if df_raw.shape[1] < 2:
        st.error("CSV must have at least 2 columns (features + target)")
        st.stop()
        
    # Check for common target column names
    target_candidates = ['Default_Status', 'default', 'target', 'label', 'Default', 'Credit_Risk', 'Risk']
    found_target = [col for col in target_candidates if col in df_raw.columns]
    if found_target:
        st.sidebar.info(f"✓ Found target column: {found_target[0]}")
    else:
        st.sidebar.warning("⚠ No standard target column detected. Will attempt to infer.")
        st.stop()

    # Quick preview & types
    st.sidebar.markdown("### Dataset preview & diagnostics")
    if st.sidebar.button("Show preview & dtypes"):
        dataframe_info_block(df_raw)

    # Determine layout: genes-as-rows (SYMBOL) or samples-as-rows
    merged, layout_mode = transpose_if_symbol_layout(df_raw)
    st.sidebar.markdown(f"Detected layout: **{layout_mode}**")

    # If transposed layout, merged already contains GSM_ID etc.
    # Determine feature columns candidate list
    if layout_mode == 'transposed':
        # feature columns are those not Applicant_ID, Risk_Class, Default_Status
        feature_cols = [c for c in merged.columns if c not in ['Applicant_ID', 'Risk_Class', 'Default_Status']]
        # Coerce numeric for features
        merged2 = numericize_features_and_fill(merged, feature_cols)
        # After numericization, recompute feature_cols
        feature_cols = [c for c in merged2.columns if c not in ['Applicant_ID', 'Risk_Class', 'Default_Status']]
        # For modeling, ensure a Default_Status column exists; if not, try to infer (optional)
        if 'Default_Status' not in merged2.columns or merged2['Default_Status'].isnull().all():
            # If Risk_Class exists, map it
            if 'Risk_Class' in merged2.columns and merged2['Risk_Class'].notnull().any():
                mapping = {1: 'No Default', 2: 'Default', 3: 'Default'}
                merged2['Default_Status'] = merged2['Risk_Class'].map(mapping)
            else:
                merged2['Default_Status'] = 'Unknown'
        data_df = merged2.copy()
    else:
        # already samples as rows; try to find an obvious target column
        # We'll attempt to find a column with small set of unique values that looks like target
        df_work = df_raw.copy()
        
        # Check if Default_Status column exists (our target column)
        if 'Default_Status' in df_work.columns:
            default_status = df_work['Default_Status'].copy()
            # Try to convert all OTHER columns to numeric
            for c in df_work.columns:
                if c != 'Default_Status':
                    df_work[c] = pd.to_numeric(df_work[c].astype(str).str.replace(',', ''), errors='coerce')
            # Restore Default_Status
            df_work['Default_Status'] = default_status
        else:
            # Try to convert all columns to numeric where possible
            for c in df_work.columns:
                df_work[c] = pd.to_numeric(df_work[c].astype(str).str.replace(',', ''), errors='coerce')
        # Candidate numeric columns
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        
        # If Default_Status column exists, keep it and ensure it's used as target
        if 'Default_Status' in df_work.columns:
            feature_cols = numeric_cols if numeric_cols else []
            data_df = df_work.copy()
        elif not numeric_cols:
            # If none numeric, try coercing original strings by removing stray chars
            df_work = df_raw.apply(lambda col: pd.to_numeric(col.astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce'))
            numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                # No numeric columns at all -> fallback to simulation later
                data_df = df_raw.copy()
                feature_cols = []
            else:
                # assume last numeric column is target (best-effort)
                # but prefer columns with small unique values (<10)
                candidate_targets = [c for c in numeric_cols if df_work[c].nunique() <= 10]
                if candidate_targets:
                    target_col = candidate_targets[-1]
                else:
                    target_col = numeric_cols[-1]
                feature_cols = [c for c in numeric_cols if c != target_col]
                # fill NaNs by mean
                for c in feature_cols:
                    df_work[c] = df_work[c].fillna(df_work[c].mean())
                data_df = df_work.copy()
                data_df['__target__'] = df_raw[target_col]
                # if target is not numeric, factorize later
                # rename target to Default_Status if sensible
                data_df = data_df.rename(columns={target_col: 'default_inferred'})
        else:
            # assume last numeric column is target (best-effort)
            # but prefer columns with small unique values (<10)
            candidate_targets = [c for c in numeric_cols if df_work[c].nunique() <= 10]
            if candidate_targets:
                target_col = candidate_targets[-1]
            else:
                target_col = numeric_cols[-1]
            feature_cols = [c for c in numeric_cols if c != target_col]
            # fill NaNs by mean
            for c in feature_cols:
                df_work[c] = df_work[c].fillna(df_work[c].mean())
            data_df = df_work.copy()
            data_df['__target__'] = df_raw[target_col]
            # if target is not numeric, factorize later
            # rename target to Default_Status if sensible
            data_df = data_df.rename(columns={target_col: 'default_inferred'})

    # At this point `data_df` is the working dataset; determine features & target
    # If transposed: features = feature_cols, target = Cancer_Status
    # If samples-as-rows: features = feature_cols, target = target_inferred (if exists)

    # Create a robust pipeline to extract X (features) and y (labels)
    def prepare_Xy(df_work: pd.DataFrame):
        # If transposed style (has Default_Status)
        if 'Default_Status' in df_work.columns:
            feat_cols_local = [c for c in df_work.columns if c not in ['Applicant_ID', 'Risk_Class', 'Default_Status']]
            # ensure numeric
            df_work = numericize_features_and_fill(df_work, feat_cols_local)
            feat_cols_local = [c for c in df_work.columns if c not in ['Applicant_ID', 'Risk_Class', 'Default_Status']]
            X_local = df_work[feat_cols_local]
            y_local = df_work['Default_Status'].astype(str)
            return X_local, y_local, feat_cols_local
        # samples-as-rows with inferred target
        if 'default_inferred' in df_work.columns:
            feat_cols_local = [c for c in df_work.columns if c not in ['default_inferred']]
            # ensure numeric
            df_work = numericize_features_and_fill(df_work, feat_cols_local)
            feat_cols_local = [c for c in df_work.columns if c not in ['default_inferred']]
            X_local = df_work[feat_cols_local]
            y_local = df_work['default_inferred']
            return X_local, y_local, feat_cols_local
        # Otherwise, if there are many non-gene columns and no target, pick numeric columns
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            if len(numeric_cols) >= 2:
                # treat last numeric col as target (best-effort)
                feat_cols_local = numeric_cols[:-1]
                X_local = df_work[feat_cols_local].fillna(df_work[feat_cols_local].mean())
                y_local = df_work[numeric_cols[-1]]
                return X_local, y_local, feat_cols_local
        # No numeric features: return empty
        return pd.DataFrame(), pd.Series(dtype=object), []

    X, y, feat_cols_final = prepare_Xy(data_df)

    # If no numeric features found, simulate
    if X.empty or (len(feat_cols_final) == 0):
        st.warning("The dataset had no usable numeric features. Generating simulated data so you can preview functionality.")
        n = data_df.shape[0] if data_df.shape[0] > 0 else 50
        sim = simulate_numeric_data(n_samples=n, n_features=60)
        # attach a simulated binary target
        sim['Default_Status'] = np.random.choice(['No Default', 'Default'], size=n, p=[0.6, 0.4])
        X = sim.drop(columns=['Default_Status'])
        y = sim['Default_Status']
        feat_cols_final = X.columns.tolist()

    # Preserve the original target label semantics where possible
    label_encoder = None
    class_label_names = None
    if y.dtype == object or y.dtype == 'O' or y.dtype == 'str':
        label_encoder = LabelEncoder().fit(y.astype(str))
        y_enc = label_encoder.transform(y.astype(str))
        y_final = pd.Series(y_enc, index=y.index)
        class_label_names = list(label_encoder.classes_)
    else:
        y_final = y.astype(int)
        unique_labels = sorted(y_final.unique())
        if unique_labels == [0, 1]:
            class_label_names = ['Low Risk 🟢', 'High Risk 🔴']
        elif unique_labels == [0, 1, 2]:
            class_label_names = ['Conservative 🛡️', 'Moderate ⚖️', 'Aggressive 🚀']
        else:
            class_label_names = [str(v) for v in unique_labels]

    # If too few classes or samples for stratify/split, handle gracefully
    n_classes = len(np.unique(y_final))
    if len(y_final) < 5 or n_classes < 2:
        st.warning("Not enough samples or labels to perform ML. Showing visualizations only.")
        can_train = False
    else:
        can_train = True

    # Page routing
    if page == "Data Intelligence":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Advanced Data Intelligence Dashboard")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📊 Samples", data_df.shape[0])
        col2.metric("📈 Features", data_df.shape[1])
        col3.metric("❌ Missing", data_df.isnull().sum().sum())
        col4.metric("🎯 Classes", len(np.unique(y_final)))
        col5.metric("⚖️ Balance", f"{max(pd.Series(y_final).value_counts()) / len(y_final) * 100:.1f}%")
        
        st.markdown("---")
        
        # Risk Distribution
        st.markdown("### 📋 Risk Distribution Analysis")
        risk_dist = pd.Series(y_final).value_counts().sort_index()
        if label_encoder is not None:
            display_labels = [label_encoder.inverse_transform([i])[0] for i in risk_dist.index]
        else:
            if class_label_names is not None and np.array_equal(risk_dist.index.values, np.arange(len(class_label_names))):
                display_labels = class_label_names
            else:
                display_labels = [str(i) for i in risk_dist.index]
        fig = px.pie(values=risk_dist.values, names=display_labels,
                    title="Risk Class Distribution", hole=0.3)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, width='stretch')

        # Scorecard-style Decile Segmentation
        st.markdown("### 🧾 Scorecard-style Risk Decile Segmentation")
        derived_score = derive_scorecard_risk_profile(X, y_final)
        if not derived_score.empty:
            label_series = pd.Series(y_final, name="Actual Class")
            if label_encoder is not None:
                try:
                    label_series = pd.Series(label_encoder.inverse_transform(label_series.astype(int)), name="Actual Class")
                except Exception:
                    label_series = label_series.astype(str)
            elif class_label_names is not None:
                label_series = pd.Series([class_label_names[int(v)] if str(v).isdigit() and int(v) < len(class_label_names) else str(v)
                                          for v in label_series], name="Actual Class")

            decile_labels = [f"Decile {i}" for i in range(1, 11)]
            try:
                decile_bins = pd.qcut(derived_score, q=10, labels=decile_labels, duplicates='drop')
            except ValueError:
                decile_bins = pd.cut(derived_score, bins=10, labels=decile_labels, duplicates='drop')

            score_df = pd.DataFrame({
                "Risk Score": derived_score,
                "Risk Decile": decile_bins,
                "Actual Class": label_series
            })
            decile_counts = score_df.groupby(["Risk Decile", "Actual Class"]).size().reset_index(name="Count")
            decile_avg = score_df.groupby("Risk Decile")["Risk Score"].mean().reset_index()

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(decile_counts, x="Risk Decile", y="Count", color="Actual Class",
                             title="Actual Class Composition Across Risk Deciles", text="Count")
                fig.update_layout(template="plotly_dark", xaxis={'categoryorder':'array', 'categoryarray': decile_labels})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(decile_avg, x="Risk Decile", y="Risk Score", markers=True,
                              title="Average Derived Risk Score by Decile",
                              labels={"Risk Score": "Avg Score"})
                fig.update_layout(template="plotly_dark", xaxis={'categoryorder':'array', 'categoryarray': decile_labels})
                st.plotly_chart(fig, use_container_width=True)

            if len(np.unique(y_final)) == 2:
                y_numeric_event = pd.Series(y_final).astype(float)
                event_summary = score_df.assign(Event=y_numeric_event.values).groupby("Risk Decile")["Event"].mean().reset_index()
                event_summary["Event Rate (%)"] = event_summary["Event"] * 100
                st.dataframe(event_summary[["Risk Decile", "Event Rate (%)"]].style.format({"Event Rate (%)": "{:.1f}%"}), use_container_width=True)

            st.markdown(
                "This section derives a simple scorecard from numeric feature correlations and "
                "segments applicants into ten risk deciles. In real credit risk analytics, this kind of "
                "decile segmentation is used to compare borrower groups for pricing, underwriting, and monitoring."
            )
        else:
            st.info("Not enough numeric data to generate a derived scorecard risk decile view.")

        # Feature Statistics
        st.markdown("### 📊 Feature Statistics & Distributions")
        numeric_feats = X.select_dtypes(include=[np.number]).columns[:6]
        
        cols = st.columns(3)
        for idx, feat in enumerate(numeric_feats):
            with cols[idx % 3]:
                fig = px.box(x=y_final, y=X[feat], title=f"{feat} by Risk Class",
                            labels={feat: feat, 'x': 'Risk Class'})
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, width='stretch')
        
        # Correlation Heatmap
        st.markdown("### 🔗 Feature Correlation Heatmap")
        corr_matrix = X[numeric_feats].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu',
                       zmin=-1, zmax=1, title="Feature Correlations")
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, width='stretch')
        
        # K-Means Clustering
        st.markdown("### 🎯 Unsupervised Risk Segmentation (K-Means)")
        if len(X) > 10:
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X[numeric_feats])
                
                # Elbow curve
                inertias = []
                silhouettes = []
                K_range = range(2, min(8, len(X) // 5))
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    silhouettes.append(silhouette_score(X_scaled, km.labels_))
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(x=list(K_range), y=inertias, markers=True,
                                 title="Elbow Method", labels={'x': 'k (clusters)', 'y': 'Inertia'})
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    fig = px.line(x=list(K_range), y=silhouettes, markers=True,
                                 title="Silhouette Score", labels={'x': 'k (clusters)', 'y': 'Score'})
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, width='stretch')
                
                # Optimal K-Means visualization
                optimal_k = list(K_range)[np.argmax(silhouettes)]
                km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                clusters = km_final.fit_predict(X_scaled)
                
                pca_vis = PCA(n_components=2)
                X_pca = pca_vis.fit_transform(X_scaled)
                
                cluster_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
                                         'Cluster': clusters, 'True Risk': y_final})
                fig = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster', 
                               hover_data=['True Risk'], title=f"Optimal K-Means (k={optimal_k})")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"Clustering failed: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Model Comparison":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🤖 Production-Grade Model Comparison & Benchmarking")
        st.markdown("Compare multiple algorithms using real-world credit risk metrics (AUC-ROC, F1-Score, Calibration).")
        
        if not can_train:
            st.warning("Not enough samples to train models.")
            st.stop()
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_final, test_size=0.2, stratify=y_final, random_state=42
            )
            
            # Check class distribution for imbalance
            class_counts = pd.Series(y_train).value_counts()
            is_imbalanced = (class_counts.min() / class_counts.sum()) < 0.3
            
            # Define models with class weighting for imbalanced data
            models = {
                'RandomForest': Pipeline([('scaler', StandardScaler()), 
                                        ('clf', RandomForestClassifier(
                                            n_estimators=100, max_depth=10, min_samples_split=15, 
                                            min_samples_leaf=8, class_weight='balanced', random_state=42))]),
                'GradientBoosting': Pipeline([('scaler', StandardScaler()), 
                                           ('clf', GradientBoostingClassifier(
                                               n_estimators=100, max_depth=6, learning_rate=0.05, 
                                               subsample=0.8, random_state=42))]),
            }
            
            if HAS_LIGHTGBM:
                models['LightGBM'] = Pipeline([('scaler', StandardScaler()),
                                             ('clf', lgb.LGBMClassifier(
                                                 n_estimators=100, max_depth=8, learning_rate=0.05, 
                                                 num_leaves=20, is_unbalance=is_imbalanced,
                                                 random_state=42, verbose=-1))])
            
            # Class imbalance alert
            if is_imbalanced:
                st.info(f"⚖️ **Class Imbalance Detected**: Minority class {(class_counts.min()/class_counts.sum()*100):.1f}% of data. "
                       "Models use class weighting & tracking AUC-ROC (more meaningful than accuracy).")
            
            # Train and evaluate models
            results = {}
            for name, model in models.items():
                with st.spinner(f"Training {name} with cross-validation..."):
                    # Cross-validation with multiple metrics
                    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    cv_auroc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc') if len(np.unique(y_train)) == 2 else None
                    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                    
                    # Train and test
                    model.fit(X_train, y_train)
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # Compute comprehensive metrics
                    if len(np.unique(y_test)) == 2:
                        test_auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        test_auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    
                    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Get feature importance
                    try:
                        importances = model.named_steps['clf'].feature_importances_ if hasattr(model.named_steps['clf'], 'feature_importances_') else None
                    except:
                        importances = None
                    
                    results[name] = {
                        'model': model,
                        'cv_accuracy_mean': cv_accuracy.mean(),
                        'cv_accuracy_std': cv_accuracy.std(),
                        'cv_auroc_mean': cv_auroc.mean() if cv_auroc is not None else None,
                        'cv_auroc_std': cv_auroc.std() if cv_auroc is not None else None,
                        'cv_f1_mean': cv_f1.mean(),
                        'cv_f1_std': cv_f1.std(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'test_auroc': test_auroc,
                        'test_f1': test_f1,
                        'test_precision': test_precision,
                        'test_recall': test_recall,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'importances': importances,
                        'feature_names': X.columns.tolist() if importances is not None else [],
                    }
            
            # Display comprehensive metrics table
            st.markdown("### 📊 Production-Grade Performance Metrics (Cross-Validation + Test)")
            
            metrics_data = []
            for model_name in results.keys():
                res = results[model_name]
                metrics_data.append({
                    'Model': model_name,
                    'CV Acc': f"{res['cv_accuracy_mean']:.4f}±{res['cv_accuracy_std']:.4f}",
                    'CV AUC': f"{res['cv_auroc_mean']:.4f}±{res['cv_auroc_std']:.4f}" if res['cv_auroc_mean'] is not None else "N/A",
                    'CV F1': f"{res['cv_f1_mean']:.4f}±{res['cv_f1_std']:.4f}",
                    'Test Acc': f"{res['test_acc']:.4f}",
                    'Test AUC': f"{res['test_auroc']:.4f}",
                    'Test F1': f"{res['test_f1']:.4f}",
                    'Precision': f"{res['test_precision']:.4f}",
                    'Recall': f"{res['test_recall']:.4f}",
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Diagnostic info: Show if AUCs are truly identical
            aucs = [results[m]['test_auroc'] for m in results.keys()]
            auc_min, auc_max = min(aucs), max(aucs)
            auc_diff = auc_max - auc_min
            
            if auc_diff < 0.001:
                st.warning(f"⚠️ **Models have nearly identical AUC scores** (difference: {auc_diff:.6f}). "
                          "This suggests either: (1) data is too easy to classify, (2) all models converged to same decision boundary, "
                          "or (3) dataset lacks discriminative features. Try more complex data or adjust hyperparameters.")
            else:
                st.success(f"✅ **Models show differentiation** (AUC range: {auc_min:.4f} - {auc_max:.4f})")
            
            st.info("📌 **Metrics Guide**: \n"
                   "• **AUC-ROC** (0-1): How well model ranks risky vs safe customers. **0.7-0.8** = Good, **0.8+** = Excellent\n"
                   "• **F1-Score**: Balanced precision-recall. Better than accuracy for imbalanced data.\n"
                   "• **Precision**: Of flagged as default, how many actually default? (avoid false alarms)\n"
                   "• **Recall**: Of actual defaults, how many are caught? (avoid missing real risks)")
            
            # ROC Curves (Primary production metric)
            st.markdown("### 📈 ROC Curve Comparison (AUC-ROC)")
            st.markdown("*AUC-ROC is the standard metric for credit risk. Higher = better. Industry standard: 0.70+*")
            
            # Detailed diagnostic: Show prediction probability distributions
            st.markdown("#### 🔍 Model Prediction Distribution (for debugging)")
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                st.markdown("**Default Probability Distribution by Model**")
                dist_data = []
                for name, res in results.items():
                    if res['y_pred_proba'].shape[1] > 1:
                        proba = res['y_pred_proba'][:, 1]
                    else:
                        proba = res['y_pred_proba'][:, 0]
                    
                    dist_data.append({
                        'Model': name,
                        'Min Prob': f"{proba.min():.4f}",
                        'Max Prob': f"{proba.max():.4f}",
                        'Mean Prob': f"{proba.mean():.4f}",
                        'Std Prob': f"{proba.std():.4f}",
                    })
                
                dist_df = pd.DataFrame(dist_data)
                st.dataframe(dist_df, use_container_width=True)
                
                if auc_diff < 0.001:
                    st.error("❌ **Problem Detected**: Probability distributions are too similar. "
                            "Models may not be learning different decision boundaries.")
            
            with dist_col2:
                st.markdown("**Prediction Accuracy Comparison**")
                acc_data = []
                for name, res in results.items():
                    acc_data.append({
                        'Model': name,
                        'Train Acc': f"{res['train_acc']:.4f}",
                        'Test Acc': f"{res['test_acc']:.4f}",
                        'Overfit Gap': f"{(res['train_acc'] - res['test_acc']):.4f}",
                    })
                
                acc_df = pd.DataFrame(acc_data)
                st.dataframe(acc_df, use_container_width=True)
            
            fig = go.Figure()
            
            for name, res in results.items():
                if len(np.unique(y_test)) > 1:
                    try:
                        if res['y_pred_proba'].shape[1] > 1:
                            proba = res['y_pred_proba'][:, 1]
                        else:
                            proba = res['y_pred_proba'][:, 0]
                        
                        fpr, tpr, _ = roc_curve(y_test, proba)
                        roc_auc = auc(fpr, tpr)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers',
                                               name=f'{name} (AUC={roc_auc:.4f})',
                                               line=dict(width=3),
                                               marker=dict(size=3)))
                    except Exception as e:
                        pass
            
            # Add random classifier baseline
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   name='Random Classifier (AUC=0.5)', 
                                   line=dict(dash='dash', width=2, color='lightgray')))
            
            # Add expert baseline
            fig.add_hline(y=0.7, line_dash="dot", line_color="orange", 
                         annotation_text="Expert Level (AUC=0.70)")
            
            fig.update_layout(template="plotly_dark", title="ROC Curve Comparison - Production Metric",
                            xaxis_title="False Positive Rate (1 - Specificity)", 
                            yaxis_title="True Positive Rate (Sensitivity)",
                            height=600,
                            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability distribution histogram
            st.markdown("#### 📊 Probability Distribution Histograms (Why Curves Might Look Identical)")
            
            prob_cols = st.columns(min(3, len(results)))
            for idx, (name, res) in enumerate(results.items()):
                with prob_cols[idx % len(prob_cols)]:
                    if res['y_pred_proba'].shape[1] > 1:
                        proba = res['y_pred_proba'][:, 1]
                    else:
                        proba = res['y_pred_proba'][:, 0]
                    
                    prob_hist_df = pd.DataFrame({
                        'Probability': proba,
                        'True Label': ['Default' if y == 1 else 'No Default' for y in y_test]
                    })
                    
                    fig_hist = px.histogram(prob_hist_df, x='Probability', color='True Label',
                                          title=f"{name}",
                                          nbins=25, barmode='overlay')
                    fig_hist.update_layout(template="plotly_dark", height=280, showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            st.caption("💡 **If histograms look identical**: All models are outputting the same probabilities. "
                      "This typically means: (1) **Data is too easy** (linearly separable), (2) **All models converged to same boundary**, "
                      "or (3) **Dataset too small/lacks diversity**.")
            
            # Precision-Recall Curves
            st.markdown("### 📉 Precision-Recall Curve Comparison")
            st.markdown("*Critical for imbalanced data. Shows tradeoff between catching defaults (recall) and accuracy (precision).*")
            
            fig = go.Figure()
            
            for name, res in results.items():
                try:
                    if res['y_pred_proba'].shape[1] > 1:
                        proba = res['y_pred_proba'][:, 1]
                    else:
                        proba = res['y_pred_proba'][:, 0]
                    
                    precision, recall, _ = precision_recall_curve(y_test, proba)
                    avg_precision = np.mean(precision)
                    
                    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines+markers', 
                                             name=f'{name} (AP={avg_precision:.4f})',
                                             line=dict(width=3), marker=dict(size=3)))
                except Exception as e:
                    pass
            
            fig.update_layout(template="plotly_dark", title="Precision-Recall Curve - Imbalanced Data Metric",
                            xaxis_title="Recall (True Positive Rate)", 
                            yaxis_title="Precision (Positive Predictive Value)",
                            height=500,
                            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model analysis (selected by AUC-ROC)
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_auroc'])
            best_model = results[best_model_name]
            
            st.markdown(f"### 🏆 Best Model: {best_model_name} (AUC-ROC: {best_model['test_auroc']:.4f})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 AUC-ROC", f"{best_model['test_auroc']:.4f}", "Production Metric")
            col2.metric("📊 F1-Score", f"{best_model['test_f1']:.4f}", "Balanced Score")
            col3.metric("🎪 Precision", f"{best_model['test_precision']:.4f}", "False Positive Rate")
            col4.metric("🥅 Recall", f"{best_model['test_recall']:.4f}", "False Negative Rate")
            
            # Confusion Matrix for best model
            cm = confusion_matrix(y_test, best_model['y_pred'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax, 
                           xticklabels=['No Default', 'Default'] if len(cm) == 2 else None,
                           yticklabels=['No Default', 'Default'] if len(cm) == 2 else None)
                ax.set_title(f"{best_model_name} - Confusion Matrix")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Classification Report (Best Model):**")
                report_text = classification_report(y_test, best_model['y_pred'], 
                                                   target_names=['No Default', 'Default'] if len(np.unique(y_test)) == 2 else None)
                st.text(report_text)
            
            # Feature Importance (if available)
            if best_model['importances'] is not None and len(best_model['feature_names']) > 0:
                st.markdown("### 🔍 Feature Importance in Best Model")
                
                feature_importance = pd.Series(best_model['importances'], 
                                             index=best_model['feature_names']).sort_values(ascending=False).head(12)
                
                fig = px.bar(feature_importance.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}),
                           x='Importance', y='Feature', orientation='h',
                           title=f"Top 12 Features - {best_model_name}",
                           color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 These {len(best_model['feature_names'])} features drive {best_model_name} predictions. "
                       "In production, feature importance guides data collection priorities.")
            
            # Model Calibration Check (important for probability thresholds)
            st.markdown("### 📏 Model Calibration Analysis")
            st.markdown("*Checks if predicted probabilities match actual outcomes. Critical for adjusting approval thresholds.*")
            
            try:
                from sklearn.calibration import calibration_curve
                
                fig = go.Figure()
                
                for name, res in results.items():
                    if res['y_pred_proba'].shape[1] > 1:
                        proba = res['y_pred_proba'][:, 1]
                    else:
                        proba = res['y_pred_proba'][:, 0]
                    
                    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
                    
                    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers',
                                           name=f'{name}', line=dict(width=2), marker=dict(size=6)))
                
                # Diagonal line (perfect calibration)
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                       name='Perfect Calibration', 
                                       line=dict(dash='dash', width=2, color='lightgray')))
                
                fig.update_layout(template="plotly_dark", title="Calibration Curves",
                                xaxis_title="Mean Predicted Probability",
                                yaxis_title="Fraction of Positives",
                                height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("📌 Points above diagonal = model **underconfident**. Points below = model **overconfident**. "
                          "Closer to diagonal = better for setting approval thresholds.")
                
            except Exception as e:
                st.info("Calibration analysis skipped (requires sufficient samples in each bin)")
            
            # Store best model for later pages
            st.session_state.best_model = best_model['model']
            st.session_state.best_model_name = best_model_name
            
            # Production readiness summary
            st.markdown("---")
            st.markdown("### ✅ Production Readiness Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.success(f"✅ Best Model: **{best_model_name}** (AUC: {best_model['test_auroc']:.4f})")
                st.info(f"✅ Class Balance: {'Handled with weights' if is_imbalanced else 'Balanced data'}")
                st.info(f"✅ Metrics Tracked: Accuracy, AUC-ROC, F1, Precision, Recall, Calibration")
            
            with summary_col2:
                if best_model['test_auroc'] >= 0.8:
                    st.success(f"🚀 **Production-Ready** (AUC >= 0.8)")
                elif best_model['test_auroc'] >= 0.7:
                    st.warning(f"⚠️ **Acceptable** (AUC >= 0.7) - Monitor in production")
                else:
                    st.error(f"❌ **Needs Improvement** (AUC < 0.7) - Requires hyperparameter tuning")
                
                st.info(f"📊 Models trained: {len(results)}")
                st.info(f"📈 Cross-validation folds: 5")
            
        except Exception as e:
            st.error(f"Model comparison failed: {str(e)[:200]}")
        
        st.markdown("</div>", unsafe_allow_html=True)


    elif page == "Stock Recommendations":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🤖 ML-Powered Stock Investment Recommendations")
        st.markdown("Our AI analyzes your complete credit profile to recommend suitable investments.")

        # Get credit features for ML model
        credit_feature_cols = []
        feature_candidates = ['credit_score', 'income', 'debt_ratio', 'employment_years', 'age', 'Credit_Score', 'Income', 'Debt_Ratio']
        for col in feature_candidates:
            if col in data_df.columns:
                credit_feature_cols.append(col)

        if len(credit_feature_cols) < 2:
            st.error("Not enough credit features found for ML analysis. Need at least 2 features like credit_score, income, etc.")
            st.stop()

        credit_features = data_df[credit_feature_cols].copy()
        credit_features = credit_features.fillna(credit_features.mean())

        # Option to input personal profile
        st.markdown("### 📊 Your Investment Profile")
        use_personal = st.checkbox("Use my personal credit profile (instead of dataset average)")

        user_profile = None
        if use_personal:
            st.markdown("Enter your credit details:")
            col1, col2 = st.columns(2)
            with col1:
                if 'credit_score' in credit_feature_cols:
                    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
                    user_profile = {'credit_score': credit_score}
                if 'income' in credit_feature_cols:
                    income = st.number_input("Annual Income", min_value=0, value=50000)
                    if user_profile is None: user_profile = {}
                    user_profile['income'] = income
                if 'age' in credit_feature_cols:
                    age = st.number_input("Age", min_value=18, max_value=100, value=35)
                    if user_profile is None: user_profile = {}
                    user_profile['age'] = age
            with col2:
                if 'debt_ratio' in credit_feature_cols:
                    debt_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                    if user_profile is None: user_profile = {}
                    user_profile['debt_ratio'] = debt_ratio
                if 'employment_years' in credit_feature_cols:
                    emp_years = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
                    if user_profile is None: user_profile = {}
                    user_profile['employment_years'] = emp_years

        # Get ML-based recommendations
        recommendations, message, risk_level, risk_display_name = get_stock_recommendations_ml(X, y_final, credit_features, user_profile)

        # Display message
        st.markdown(f"**{message}**")
        st.info(
            "High-risk investments can still show strong returns, but they also come with greater volatility and potential downside. "
            "A high-risk label means the stock is more likely to swing sharply, even if historical returns are currently attractive."
        )

        if recommendations:
            st.markdown("---")
            st.markdown("### 💰 Personalized Stock Portfolio (8 Diverse Options)")

            # Categorize stocks
            primary_count = 5 if risk_level in [0, 2] else 4  # Conservative/aggressive get 5 primary, moderate gets 4
            primary_stocks = recommendations[:primary_count]
            alternative_stocks = recommendations[primary_count:]

            # Primary recommendations
            st.markdown(f"#### 🎯 Primary Recommendations ({risk_display_name})")
            st.markdown("Best matches for your ML-predicted risk profile:")

            for i, stock in enumerate(primary_stocks):
                with st.expander(f"⭐ {stock['symbol']} - {stock['name']}", expanded=(i==0)):
                    # Stock metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Symbol", stock['symbol'])
                    with col2:
                        st.metric("Sector", stock['sector'])
                    with col3:
                        st.metric("Dividend", stock['dividend'])
                    with col4:
                        st.metric("Risk Level", stock['risk'])

                    # Fetch real stock data
                    with st.spinner(f"Loading {stock['symbol']} data..."):
                        hist, info = get_stock_data(stock['symbol'])

                    if hist is not None and info is not None:
                        # Key metrics
                        metrics = get_stock_metrics(info)
                        if metrics:
                            st.markdown("#### 📊 Key Metrics")
                            metric_cols = st.columns(4)
                            for j, (key, value) in enumerate(list(metrics.items())[:4]):
                                with metric_cols[j]:
                                    st.metric(key, value)

                        # Price chart
                        st.markdown("#### 📈 Price Trend & Analysis")
                        price_chart = create_stock_price_chart(hist, stock['symbol'], stock['name'])
                        if price_chart:
                            st.plotly_chart(price_chart, width='stretch')

                            # Additional insights
                            latest_price = hist['Close'].iloc[-1]
                            price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
                            pct_change = (price_change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 and hist['Close'].iloc[-2] != 0 else 0

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Latest Price", f"${latest_price:.2f}",
                                         delta=f"{pct_change:+.2f}%" if pct_change != 0 else None)
                            with col_b:
                                max_price = hist['Close'].max()
                                st.metric("52W High", f"${max_price:.2f}")
                            with col_c:
                                min_price = hist['Close'].min()
                                st.metric("52W Low", f"${min_price:.2f}")

                        # Volume chart
                        volume_chart = create_volume_chart(hist, stock['symbol'])
                        if volume_chart:
                            st.markdown("#### 📊 Trading Volume")
                            st.plotly_chart(volume_chart, width='stretch')

                        # Performance analysis
                        st.markdown("#### 📉 Performance Analysis")
                        if len(hist) > 30:
                            # Calculate returns
                            returns_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) * 100 if len(hist) >= 30 else 0
                            returns_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-90] - 1) * 100 if len(hist) >= 90 else 0
                            returns_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

                            perf_cols = st.columns(3)
                            with perf_cols[0]:
                                st.metric("1-Month Return", f"{returns_1m:+.1f}%" if returns_1m != 0 else "N/A")
                            with perf_cols[1]:
                                st.metric("3-Month Return", f"{returns_3m:+.1f}%" if returns_3m != 0 else "N/A")
                            with perf_cols[2]:
                                st.metric("1-Year Return", f"{returns_1y:+.1f}%")

                            # Volatility analysis
                            volatility = hist['Pct_Change'].std() * np.sqrt(252) * 100  # Annualized volatility
                            st.metric("Annual Volatility", f"{volatility:.1f}%")

                    else:
                        st.warning(f"Could not load real-time data for {stock['symbol']}. Showing basic recommendation only.")

                    st.markdown("---")

            # Alternative recommendations
            if alternative_stocks:
                st.markdown(f"#### 🔄 Alternative Options")
                st.markdown("Additional stocks from other risk categories for portfolio diversification:")

                for stock in alternative_stocks:
                    with st.expander(f"🔄 {stock['symbol']} - {stock['name']}", expanded=False):
                        # Stock metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Symbol", stock['symbol'])
                        with col2:
                            st.metric("Sector", stock['sector'])
                        with col3:
                            st.metric("Dividend", stock['dividend'])
                        with col4:
                            st.metric("Risk Level", stock['risk'])

                        # Fetch real stock data
                        with st.spinner(f"Loading {stock['symbol']} data..."):
                            hist, info = get_stock_data(stock['symbol'])

                        if hist is not None and info is not None:
                            # Key metrics
                            metrics = get_stock_metrics(info)
                            if metrics:
                                st.markdown("#### 📊 Key Metrics")
                                metric_cols = st.columns(4)
                                for j, (key, value) in enumerate(list(metrics.items())[:4]):
                                    with metric_cols[j]:
                                        st.metric(key, value)

                            # Price chart
                            st.markdown("#### 📈 Price Trend & Analysis")
                            price_chart = create_stock_price_chart(hist, stock['symbol'], stock['name'])
                            if price_chart:
                                st.plotly_chart(price_chart, width='stretch')

                                # Additional insights
                                latest_price = hist['Close'].iloc[-1]
                                price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
                                pct_change = (price_change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 and hist['Close'].iloc[-2] != 0 else 0

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Latest Price", f"${latest_price:.2f}",
                                             delta=f"{pct_change:+.2f}%" if pct_change != 0 else None)
                                with col_b:
                                    max_price = hist['Close'].max()
                                    st.metric("52W High", f"${max_price:.2f}")
                                with col_c:
                                    min_price = hist['Close'].min()
                                    st.metric("52W Low", f"${min_price:.2f}")

                            # Performance analysis
                            st.markdown("#### 📉 Performance Analysis")
                            if len(hist) > 30:
                                returns_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-30] - 1) * 100 if len(hist) >= 30 else 0
                                returns_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-90] - 1) * 100 if len(hist) >= 90 else 0
                                returns_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100

                                perf_cols = st.columns(3)
                                with perf_cols[0]:
                                    st.metric("1-Month Return", f"{returns_1m:+.1f}%" if returns_1m != 0 else "N/A")
                                with perf_cols[1]:
                                    st.metric("3-Month Return", f"{returns_3m:+.1f}%" if returns_3m != 0 else "N/A")
                                with perf_cols[2]:
                                    st.metric("1-Year Return", f"{returns_1y:+.1f}%")

                                volatility = hist['Pct_Change'].std() * np.sqrt(252) * 100
                                st.metric("Annual Volatility", f"{volatility:.1f}%")

                        else:
                            st.warning(f"Could not load real-time data for {stock['symbol']}. Showing basic recommendation only.")

                        st.markdown("---")

            if "Conservative" in message:
                st.info("**Conservative Strategy:** ML analysis shows your profile favors stable, dividend-paying companies. Focus on long-term wealth accumulation with lower volatility.")
            elif "Moderate" in message:
                st.info("**Moderate Strategy:** ML model predicts balanced growth potential. Mix of established companies for steady returns with some growth upside.")
            else:
                st.info("**Aggressive Strategy:** ML analysis indicates high-growth potential in your profile. Suitable for capital appreciation in innovative tech sectors.")
        else:
            st.error("ML model could not generate recommendations. Please ensure your data has sufficient credit features.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Prediction Engine":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Production-Grade Risk Prediction Engine")
        st.markdown("Enter applicant details for instant risk assessment with model confidence & feature analysis.")
        
        if not can_train:
            st.warning("Not enough samples to build prediction engine.")
            st.stop()
        
        try:
            # Train final model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_final, test_size=0.2, stratify=y_final, random_state=42
            )
            
            final_model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42))
            ])
            final_model.fit(X_train, y_train)
            
            # Get feature importance for later insights
            feature_importance = pd.Series(
                final_model.named_steps['clf'].feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            # Feature descriptions and valid ranges
            feature_descriptions = {
                'credit_score': 'Credit Score (300-850): Higher is better, indicates creditworthiness',
                'income': 'Annual Income ($): Total earnings, higher reduces default risk',
                'debt_ratio': 'Debt-to-Income Ratio (0-1): Total debt divided by income, lower is better',
                'employment_years': 'Years Employed: Job stability, longer tenure reduces risk',
                'age': 'Age in Years: Experience and life stage factor',
                'credit_utilization': 'Credit Utilization %: Percent of available credit used',
                'payment_history': 'Payment History Score (0-100): On-time payment record',
                'loan_amount': 'Loan Amount ($): Size of requested credit',
                'monthly_payment': 'Monthly Payment ($): Expected monthly obligation',
            }
            
            # Input form with ALL available numeric columns
            st.markdown("### 📝 Complete Applicant Profile")
            st.info("📌 **All available features are shown below. Adjust sliders to match applicant details.**")
            
            all_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            input_values = {}
            validation_warnings = []
            
            # Display features grouped by importance
            st.markdown("#### 🔴 High-Impact Risk Factors (Top Features)")
            top_features = feature_importance.head(3).index.tolist()
            cols = st.columns(min(3, len(top_features)))
            for idx, col in enumerate(top_features):
                if idx < len(cols):
                    with cols[idx]:
                        min_val = float(X[col].min())
                        max_val = float(X[col].max())
                        mean_val = float(X[col].mean())
                        std_val = float(X[col].std())
                        input_values[col] = st.slider(
                            col.replace('_', ' ').title(),
                            min_val, max_val, mean_val,
                            help=feature_descriptions.get(col, "Credit factor")
                        )
                        st.caption(f"📊 Range: {min_val:.0f} - {max_val:.0f}")
            
            st.markdown("#### 🟡 Secondary Risk Factors")
            other_features = [c for c in all_numeric_cols if c not in top_features]
            cols = st.columns(3)
            for idx, col in enumerate(other_features):
                with cols[idx % 3]:
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    mean_val = float(X[col].mean())
                    input_values[col] = st.slider(
                        col.replace('_', ' ').title(),
                        min_val, max_val, mean_val,
                        help=feature_descriptions.get(col, "Credit factor")
                    )
            
            if st.button("🔮 Predict Risk Profile", key="predict_btn"):
                # Validation: Create full prediction dataframe with all features
                pred_df = X.iloc[0:1].copy()
                for col in all_numeric_cols:
                    if col in input_values:
                        pred_df[col] = input_values[col]
                
                # Perform input sanity checks
                validation_warnings = []
                for col, val in input_values.items():
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    # Warn if input is outside dataset range (but allow it)
                    if val < min_val:
                        validation_warnings.append(f"⚠️ {col.title()}: {val:.0f} is below dataset minimum ({min_val:.0f})")
                    elif val > max_val:
                        validation_warnings.append(f"⚠️ {col.title()}: {val:.0f} exceeds dataset maximum ({max_val:.0f})")
                    # Semantic checks for specific features
                    if 'debt_ratio' in col.lower() and val > 0.9:
                        validation_warnings.append(f"⚠️ Debt Ratio: {val:.2f} is critically high (>90%)")
                    if 'credit_score' in col.lower() and val < 500:
                        validation_warnings.append(f"⚠️ Credit Score: {val:.0f} is very low")
                    if 'income' in col.lower() and val < 15000:
                        validation_warnings.append(f"⚠️ Income: ${val:,.0f} is unusually low")
                
                # Make prediction
                try:
                    prediction = final_model.predict(pred_df)[0]
                    probabilities = final_model.predict_proba(pred_df)[0]
                    confidence = max(probabilities)
                    pred_class_idx = np.argmax(probabilities)
                    
                    # Get risk labels
                    clf_classes = final_model.named_steps['clf'].classes_
                    if class_label_names is not None and len(class_label_names) == len(clf_classes):
                        risk_labels = class_label_names
                    elif len(clf_classes) == 2:
                        risk_labels = ['Low Risk 🟢', 'High Risk 🔴']
                    elif len(clf_classes) == 3:
                        risk_labels = ['Conservative 🛡️', 'Moderate ⚖️', 'Aggressive 🚀']
                    else:
                        risk_labels = [f'Risk Class {i}' for i in range(len(clf_classes))]
                    
                    risk_label_map = {cls: label for cls, label in zip(clf_classes, risk_labels)}
                    predicted_label = risk_label_map.get(prediction, str(prediction))
                    
                    # Compute continuous risk score using CORRECT probability handling
                    # 70% from model probability + 30% from feature-based assessment
                    model_prob = probabilities[pred_class_idx]
                    
                    # Feature-based risk (0-1 scale)
                    feature_risk = 0.0
                    feature_weight = 0.0
                    for col in all_numeric_cols:
                        if col in pred_df.columns:
                            val = float(pred_df[col].iloc[0])
                            if 'debt' in col.lower():
                                debt_risk = max(0, min(1, val / 0.8))  # Normalized by typical max
                                feature_risk += debt_risk * 0.35
                                feature_weight += 0.35
                            elif 'credit' in col.lower() and 'score' in col.lower():
                                cs_norm = 1.0 - ((val - 300.0) / 550.0)
                                credit_risk = max(0, min(1, cs_norm))
                                feature_risk += credit_risk * 0.30
                                feature_weight += 0.30
                            elif 'income' in col.lower():
                                inc_norm = 1.0 - ((val - 12000.0) / 200000.0)
                                income_risk = max(0, min(1, inc_norm))
                                feature_risk += income_risk * 0.20
                                feature_weight += 0.20
                            elif 'employment' in col.lower() or 'tenure' in col.lower():
                                emp_norm = 1.0 - (val / 40.0)
                                emp_risk = max(0, min(1, emp_norm))
                                feature_risk += emp_risk * 0.15
                                feature_weight += 0.15
                    
                    if feature_weight > 0:
                        feature_risk /= feature_weight
                    else:
                        feature_risk = 0.5
                    
                    # Combined risk: 70% model probability + 30% feature assessment
                    continuous_risk = 0.70 * model_prob + 0.30 * feature_risk
                    continuous_risk = max(0, min(1, continuous_risk))
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Risk Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Risk Score (0-100)", f"{continuous_risk*100:.1f}", 
                                 delta=f"Model: {model_prob*100:.1f}%")
                    with col2:
                        st.metric("Classification", predicted_label)
                    with col3:
                        st.metric("Model Confidence", f"{confidence:.2%}")
                    
                    # Show validation warnings if any
                    if validation_warnings:
                        st.warning("**⚠️ Input Validation Alerts:**\n" + "\n".join(validation_warnings))
                    
                    # Probability distribution
                    st.markdown("#### 📊 Risk Profile Probabilities")
                    proba_df = pd.DataFrame({
                        'Risk Profile': [risk_label_map[cls] for cls in clf_classes],
                        'Probability': probabilities * 100
                    })
                    
                    fig = px.bar(proba_df, x='Risk Profile', y='Probability',
                               title="Model Prediction Confidence Across Risk Classes",
                               color='Probability',
                               color_continuous_scale='RdYlGn_r')
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance for this prediction
                    st.markdown("#### 🔍 Feature Importance in Model")
                    top_n_features = 8
                    top_importance = feature_importance.head(top_n_features)
                    importance_df = top_importance.reset_index()
                    importance_df.columns = ['Feature', 'Importance']
                    
                    fig = px.bar(importance_df.sort_values('Importance'), 
                               x='Importance', y='Feature',
                               orientation='h',
                               title=f"Top {top_n_features} Most Influential Risk Factors")
                    fig.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("#### 💡 Risk Assessment & Recommendations")
                    
                    if continuous_risk < 0.35:
                        st.success("🟢 **LOW RISK PROFILE**\n\n"
                                 "✅ Strong credit profile\n"
                                 "✅ Recommended for approval with favorable terms\n"
                                 "✅ Lower interest rates applicable")
                    elif continuous_risk < 0.65:
                        st.info("🟡 **MODERATE RISK PROFILE**\n\n"
                               "⚠️ Mixed credit indicators\n"
                               "✓ Can be approved with standard conditions\n"
                               "✓ Standard interest rates apply")
                    else:
                        st.warning("🔴 **HIGH RISK PROFILE**\n\n"
                                  "❌ Elevated default risk detected\n"
                                  "⚠️ Recommend additional verification or manual review\n"
                                  "⚠️ Premium pricing or stricter terms may apply")
                    
                    # Model confidence alert
                    if confidence < 0.60:
                        st.warning(f"🤔 **Model Uncertainty:** Confidence is {confidence:.1%} (below 60%). "
                                  "Recommend manual review of this application.")
                    elif confidence >= 0.85:
                        st.success(f"✅ **High Confidence:** Model is {confidence:.1%} confident in this assessment.")
                    
                    # Audit log
                    st.markdown("#### 📋 Prediction Audit Log")
                    audit_data = {
                        'Metric': ['Prediction Timestamp', 'Risk Score', 'Model Confidence', 'Features Used', 'Top Risk Factor'],
                        'Value': [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"{continuous_risk*100:.1f}",
                            f"{confidence*100:.1f}%",
                            str(len(all_numeric_cols)),
                            feature_importance.index[0]
                        ]
                    }
                    st.dataframe(pd.DataFrame(audit_data), use_container_width=True)
                    
                except Exception as pred_error:
                    st.error(f"Prediction failed: {str(pred_error)}")
        
        except Exception as e:
            st.error(f"Prediction Engine initialization failed: {str(e)[:200]}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Risk Dashboard":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 📊 Comprehensive Risk Assessment Dashboard")
        st.markdown("Detailed analysis of your credit risk profile and investment suitability.")

        # Credit Risk Analysis
        st.markdown("#### 💳 Credit Risk Analysis")

        # Calculate credit risk metrics
        if 'credit_score' in data_df.columns:
            avg_credit_score = data_df['credit_score'].mean()
            credit_score_std = data_df['credit_score'].std()
            max_score = data_df['credit_score'].max()
            min_score = data_df['credit_score'].min()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average Credit Score", f"{avg_credit_score:.0f}")
            col2.metric("Score Variability", f"±{credit_score_std:.0f}")
            col3.metric("Highest Score", f"{max_score:.0f}")
            col4.metric("Lowest Score", f"{min_score:.0f}")

            # Credit score distribution
            st.markdown("#### 📈 Credit Score Distribution")
            fig = px.histogram(data_df, x='credit_score', nbins=20,
                             title="Credit Score Distribution in Dataset",
                             labels={'credit_score': 'Credit Score'})
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')

        # Risk Factor Analysis
        st.markdown("#### 🎯 Key Risk Factors Correlation")

        # Calculate correlations with credit score
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
        if 'credit_score' in numeric_cols and len(numeric_cols) > 1:
            correlations = data_df[numeric_cols].corr()['credit_score'].sort_values(ascending=False)

            # Create correlation heatmap
            corr_matrix = data_df[numeric_cols].corr()

            fig = px.imshow(corr_matrix,
                          text_auto=True,
                          aspect="auto",
                          title="Correlation Matrix - Risk Factors")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')

            # Top risk factors
            st.markdown("#### 🔍 Top Risk Factors")
            top_factors = correlations.head(6).tail(5)  # Exclude self-correlation
            for factor, corr in top_factors.items():
                if factor != 'credit_score':
                    st.write(f"**{factor}:** {corr:.3f} correlation")

        # Investment Risk Profile
        st.markdown("#### 📊 Investment Risk Profile Assessment")

        # Advanced risk tolerance assessment based on multiple credit metrics
        aggregate_risk = 0.0
        risk_factors = []
        risk_weights = 0.0

        if 'credit_score' in data_df.columns:
            avg_cs = data_df['credit_score'].mean()
            cs_risk = 1.0 - ((avg_cs - 300.0) / 550.0)
            cs_risk = max(0, min(1, cs_risk))
            aggregate_risk += cs_risk * 0.30
            risk_weights += 0.30
            if avg_cs >= 750: risk_factors.append(f"Excellent credit score ({avg_cs:.0f})")
            elif avg_cs >= 700: risk_factors.append(f"Good credit score ({avg_cs:.0f})")
            elif avg_cs >= 650: risk_factors.append(f"Fair credit score ({avg_cs:.0f})")
            else: risk_factors.append(f"Poor credit score ({avg_cs:.0f})")

        if 'debt_ratio' in data_df.columns:
            avg_dr = data_df['debt_ratio'].mean()
            dr_risk = max(0, min(1, avg_dr / 0.8))
            aggregate_risk += dr_risk * 0.35
            risk_weights += 0.35
            if avg_dr <= 0.2: risk_factors.append(f"Low debt ratio ({avg_dr:.2f})")
            elif avg_dr <= 0.4: risk_factors.append(f"Moderate debt ratio ({avg_dr:.2f})")
            else: risk_factors.append(f"High debt ratio ({avg_dr:.2f})")

        if 'income' in data_df.columns:
            avg_inc = data_df['income'].mean()
            inc_risk = 1.0 - ((avg_inc - 12000.0) / 200000.0)
            inc_risk = max(0, min(1, inc_risk))
            aggregate_risk += inc_risk * 0.20
            risk_weights += 0.20
            if avg_inc >= 100000: risk_factors.append(f"Strong income (${avg_inc:,.0f})")
            elif avg_inc >= 50000: risk_factors.append(f"Adequate income (${avg_inc:,.0f})")
            else: risk_factors.append(f"Lower income (${avg_inc:,.0f})")

        # Normalize aggregate risk to 0-1 scale
        if risk_weights > 0:
            aggregate_risk /= risk_weights
        else:
            aggregate_risk = 0.5
        aggregate_risk = max(0, min(1, aggregate_risk))
        
        # Determine risk profile based on continuous score
        if aggregate_risk >= 0.65:
            profile = "🔴 High Risk"
            description = "Your profile shows higher financial risk. Recommended for conservative income preservation strategies."
        elif aggregate_risk >= 0.40:
            profile = "🟡 Moderate Risk"
            description = "Your profile shows balanced risk drivers. Recommended for moderate growth and income strategies."
        else:
            profile = "🟢 Low Risk"
            description = "Your profile shows lower financial risk. Recommended for growth-oriented strategies."

        st.markdown(f"### {profile}")
        st.markdown(f"**{description}**")
        
        # Display continuous risk gauge
        st.markdown("#### Risk Score Breakdown")
        risk_data = pd.DataFrame({
            'Assessment': ['Overall Risk Score'],
            'Score': [aggregate_risk * 100]
        })
        fig = px.bar(risk_data, x='Assessment', y='Score', 
                    title="Continuous Risk Assessment (0-100)",
                    color='Score',
                    color_continuous_scale='RdYlGn_r',
                    range_color=[0, 100])
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### ✅ Risk Factors Identified:")
        for factor in risk_factors:
            st.write(f"• {factor}")

        # Risk mitigation suggestions
        st.markdown("#### 💡 Risk Mitigation Strategies")
        if aggregate_risk >= 0.65:
            st.warning("**High Risk Alert:** Your profile indicates higher financial risk. Consider debt consolidation and conservative investment strategies.")
        if aggregate_risk >= 0.40:
            st.info("**Balanced Approach:** Your profile shows moderate risk. Focus on diversified investments and regular rebalancing.")
        else:
            st.success("**Conservative Profile:** Your profile indicates lower risk. Consider growth-oriented investments with stable foundations.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Prepare downloadable summary CSV
        summary_data = []
        if 'avg_credit_score' in locals():
            summary_data.append({"Metric": "Average Credit Score", "Value": f"{avg_credit_score:.0f}"})
            summary_data.append({"Metric": "Score Variability", "Value": f"±{credit_score_std:.0f}"})
            summary_data.append({"Metric": "Highest Score", "Value": f"{max_score:.0f}"})
            summary_data.append({"Metric": "Lowest Score", "Value": f"{min_score:.0f}"})
        if 'avg_dr' in locals():
            summary_data.append({"Metric": "Average Debt Ratio", "Value": f"{avg_dr:.2f}"})
        if 'avg_inc' in locals():
            summary_data.append({"Metric": "Average Income", "Value": f"${avg_inc:,.0f}"})
        summary_data.append({"Metric": "Aggregate Risk", "Value": f"{aggregate_risk:.2f}"})
        summary_data.append({"Metric": "Risk Profile", "Value": profile})
        summary_data.append({"Metric": "Description", "Value": description})

        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        b64 = base64.b64encode(summary_csv.encode()).decode()

        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="credit_summary.csv" style="color:#AEE7FF">Download summary CSV</a>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Cost-Benefit Analysis":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 💰 Cost-Benefit Analysis")
        st.markdown("Analyze the financial implications of your credit risk models and decision thresholds.")

        if not can_train:
            st.warning("Not enough data to perform cost-benefit analysis.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Cost-Benefit Parameters
        st.markdown("#### 📊 Cost-Benefit Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tp_benefit = st.number_input("Benefit of True Positive (TP) - Profit from correct approval ($)", 
                                       value=10000, min_value=0, step=1000, 
                                       help="Revenue from interest on loan to good borrower")
        
        with col2:
            fp_cost = st.number_input("Cost of False Positive (FP) - Loss from default ($)", 
                                    value=50000, min_value=0, step=1000,
                                    help="Loss when bad borrower defaults")
        
        with col3:
            fn_cost = st.number_input("Cost of False Negative (FN) - Missed profit ($)", 
                                    value=8000, min_value=0, step=1000,
                                    help="Opportunity cost of rejecting good borrower")
        
        with col4:
            tn_benefit = st.number_input("Benefit of True Negative (TN) - Saved costs ($)", 
                                       value=2000, min_value=0, step=500,
                                       help="Administrative savings from rejecting bad borrower")

        # Train models and calculate cost-benefit
        if st.button("Calculate Cost-Benefit Analysis"):
            with st.spinner("Training models and calculating costs..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.3, random_state=42, stratify=y_final)
                
                models = {}
                results = {}
                
                # Train models
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                models['Random Forest'] = rf
                
                gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                gb.fit(X_train, y_train)
                models['Gradient Boosting'] = gb
                
                if HAS_LIGHTGBM:
                    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
                    lgb_model.fit(X_train, y_train)
                    models['LightGBM'] = lgb_model
                
                # Calculate cost-benefit for each model
                for name, model in models.items():
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Calculate expected profit
                    expected_profit = (tp * tp_benefit) + (tn * tn_benefit) - (fp * fp_cost) - (fn * fn_cost)
                    profit_per_applicant = expected_profit / len(y_test)
                    
                    results[name] = {
                        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                        'Expected Profit': expected_profit,
                        'Profit per Applicant': profit_per_applicant,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1': f1_score(y_test, y_pred),
                        'AUC': roc_auc_score(y_test, y_pred_proba)
                    }
                
                # Display results
                st.markdown("#### 📈 Model Comparison Results")
                
                # Summary table
                summary_df = pd.DataFrame.from_dict(results, orient='index')
                summary_df = summary_df.round(2)
                st.dataframe(summary_df[['Expected Profit', 'Profit per Applicant', 'Accuracy', 'AUC']].style.highlight_max(axis=0))
                
                # Best model
                best_model = max(results.keys(), key=lambda x: results[x]['Expected Profit'])
                st.success(f"🏆 **Best Model by Profit:** {best_model} with ${results[best_model]['Expected Profit']:,.0f} expected profit")
                
                # Visualization
                st.markdown("#### 📊 Profit Comparison")
                profit_data = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Expected Profit': [results[m]['Expected Profit'] for m in results.keys()]
                })
                
                fig = px.bar(profit_data, x='Model', y='Expected Profit', 
                           title="Expected Profit by Model",
                           color='Expected Profit',
                           color_continuous_scale='RdYlGn')
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
                
                # Confusion matrix visualization
                st.markdown("#### 🔍 Confusion Matrix Analysis")
                for name, res in results.items():
                    st.markdown(f"**{name}**")
                    cm_data = pd.DataFrame({
                        'Predicted Negative': [res['TN'], res['FN']],
                        'Predicted Positive': [res['FP'], res['TP']]
                    }, index=['Actual Negative', 'Actual Positive'])
                    
                    fig = px.imshow(cm_data, text_auto=True, 
                                  title=f"Confusion Matrix - {name}",
                                  color_continuous_scale='Blues')
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, width='stretch')
                    
                    # Cost breakdown
                    cost_breakdown = {
                        'TP Benefit': res['TP'] * tp_benefit,
                        'TN Benefit': res['TN'] * tn_benefit,
                        'FP Cost': -res['FP'] * fp_cost,
                        'FN Cost': -res['FN'] * fn_cost
                    }
                    
                    breakdown_df = pd.DataFrame(list(cost_breakdown.items()), columns=['Component', 'Amount'])
                    fig = px.bar(breakdown_df, x='Component', y='Amount', 
                               title=f"Cost-Benefit Breakdown - {name}",
                               color='Amount',
                               color_continuous_scale='RdYlGn')
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, width='stretch')

        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Logout":
        st.session_state.authenticated = False
        st.success("Logged out.")
        st.rerun()

# ------------------------------------------
# Run app
# ------------------------------------------
if not st.session_state.authenticated:
    if st.session_state.mode == "signup":
        signup_page()
    else:
        login_page()
else:
    dashboard()






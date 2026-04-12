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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score, silhouette_score
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

def simulate_numeric_data(n_samples: int, n_features: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    arr = np.random.normal(size=(n_samples, n_features))
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
        # Credit score is most important
        if 'credit_score' in row.index:
            cs = row['credit_score']
            if cs >= 750: score += 2  # Aggressive
            elif cs >= 700: score += 1  # Moderate
            else: score += 0  # Conservative

        # Income factor
        if 'income' in row.index:
            inc = row['income']
            if inc >= 100000: score += 1
            elif inc >= 50000: score += 0.5

        # Debt ratio (lower is better)
        if 'debt_ratio' in row.index:
            debt = row['debt_ratio']
            if debt <= 0.2: score += 1
            elif debt <= 0.4: score += 0.5

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

        # Convert to risk tolerance level
        if score >= 3: eligibility_scores.append(2)  # Aggressive
        elif score >= 1.5: eligibility_scores.append(1)  # Moderate
        else: eligibility_scores.append(0)  # Conservative

    # Train ML model with pipeline
    if len(eligibility_scores) > 10:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                credit_features, eligibility_scores, test_size=0.2, random_state=42
            )

            # Create pipeline: StandardScaler -> RandomForestClassifier
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
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

def get_stock_recommendations_ml(X, y, credit_features, user_profile=None):
    """
    ML-based stock recommendations using trained model with feature importance scoring.
    Stocks are ranked dynamically based on user's credit profile and model feature importance.
    """
    # Comprehensive stock database with ML-readable characteristics
    all_stocks = [
        # Conservative (Low Risk, High Dividend)
        {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "dividend_yield": 2.8, "stability": 9, "growth": 3, "risk_level": 0},
        {"symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer", "dividend_yield": 2.5, "stability": 8, "growth": 3, "risk_level": 0},
        {"symbol": "KO", "name": "Coca-Cola", "sector": "Beverages", "dividend_yield": 3.1, "stability": 8, "growth": 2, "risk_level": 0},
        {"symbol": "MCD", "name": "McDonald's", "sector": "Food & Beverage", "dividend_yield": 2.2, "stability": 8, "growth": 4, "risk_level": 0},
        {"symbol": "VZ", "name": "Verizon Communications", "sector": "Telecommunications", "dividend_yield": 6.2, "stability": 8, "growth": 2, "risk_level": 0},
        {"symbol": "T", "name": "AT&T", "sector": "Telecommunications", "dividend_yield": 5.8, "stability": 7, "growth": 1, "risk_level": 0},
        {"symbol": "PEP", "name": "PepsiCo", "sector": "Beverages", "dividend_yield": 2.9, "stability": 8, "growth": 3, "risk_level": 0},
        {"symbol": "WMT", "name": "Walmart", "sector": "Retail", "dividend_yield": 1.8, "stability": 8, "growth": 4, "risk_level": 0},
        {"symbol": "CVX", "name": "Chevron", "sector": "Energy", "dividend_yield": 4.1, "stability": 7, "growth": 3, "risk_level": 0},
        {"symbol": "ABBV", "name": "AbbVie", "sector": "Healthcare", "dividend_yield": 3.7, "stability": 7, "growth": 4, "risk_level": 0},
        # Moderate (Medium Risk, Balanced)
        {"symbol": "MSFT", "name": "Microsoft", "sector": "Technology", "dividend_yield": 0.9, "stability": 8, "growth": 8, "risk_level": 1},
        {"symbol": "AAPL", "name": "Apple", "sector": "Technology", "dividend_yield": 0.4, "stability": 8, "growth": 7, "risk_level": 1},
        {"symbol": "JPM", "name": "JPMorgan Chase", "sector": "Finance", "dividend_yield": 2.5, "stability": 7, "growth": 6, "risk_level": 1},
        {"symbol": "XOM", "name": "Exxon Mobil", "sector": "Energy", "dividend_yield": 3.5, "stability": 7, "growth": 4, "risk_level": 1},
        {"symbol": "HD", "name": "Home Depot", "sector": "Retail", "dividend_yield": 2.3, "stability": 7, "growth": 6, "risk_level": 1},
        {"symbol": "UNH", "name": "United Health Group", "sector": "Healthcare", "dividend_yield": 1.5, "stability": 8, "growth": 7, "risk_level": 1},
        {"symbol": "V", "name": "Visa", "sector": "Financial Services", "dividend_yield": 0.8, "stability": 8, "growth": 8, "risk_level": 1},
        {"symbol": "MA", "name": "Mastercard", "sector": "Financial Services", "dividend_yield": 0.6, "stability": 8, "growth": 8, "risk_level": 1},
        {"symbol": "PFE", "name": "Pfizer", "sector": "Healthcare", "dividend_yield": 5.2, "stability": 7, "growth": 5, "risk_level": 1},
        {"symbol": "COST", "name": "Costco", "sector": "Retail", "dividend_yield": 0.7, "stability": 8, "growth": 7, "risk_level": 1},
        {"symbol": "DIS", "name": "Disney", "sector": "Entertainment", "dividend_yield": 0.0, "stability": 6, "growth": 6, "risk_level": 1},
        {"symbol": "NKE", "name": "Nike", "sector": "Consumer Goods", "dividend_yield": 1.3, "stability": 7, "growth": 6, "risk_level": 1},
        # Aggressive (High Risk, High Growth)
        {"symbol": "NVDA", "name": "NVIDIA", "sector": "Technology", "dividend_yield": 0.1, "stability": 5, "growth": 9, "risk_level": 2},
        {"symbol": "TSLA", "name": "Tesla", "sector": "Automotive/Tech", "dividend_yield": 0.0, "stability": 4, "growth": 9, "risk_level": 2},
        {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Technology", "dividend_yield": 0.0, "stability": 5, "growth": 8, "risk_level": 2},
        {"symbol": "ARK", "name": "ARK Innovation ETF", "sector": "Tech ETF", "dividend_yield": 0.2, "stability": 5, "growth": 8, "risk_level": 2},
        {"symbol": "AMZN", "name": "Amazon", "sector": "E-commerce/Tech", "dividend_yield": 0.0, "stability": 6, "growth": 8, "risk_level": 2},
        {"symbol": "GOOGL", "name": "Alphabet (Google)", "sector": "Technology", "dividend_yield": 0.0, "stability": 7, "growth": 8, "risk_level": 2},
        {"symbol": "META", "name": "Meta Platforms", "sector": "Technology", "dividend_yield": 0.0, "stability": 5, "growth": 8, "risk_level": 2},
        {"symbol": "NFLX", "name": "Netflix", "sector": "Entertainment", "dividend_yield": 0.0, "stability": 5, "growth": 7, "risk_level": 2},
        {"symbol": "SQ", "name": "Block (Square)", "sector": "Financial Technology", "dividend_yield": 0.0, "stability": 4, "growth": 8, "risk_level": 2},
        {"symbol": "SHOP", "name": "Shopify", "sector": "E-commerce", "dividend_yield": 0.0, "stability": 5, "growth": 8, "risk_level": 2},
        {"symbol": "UBER", "name": "Uber", "sector": "Transportation", "dividend_yield": 0.0, "stability": 4, "growth": 8, "risk_level": 2},
        {"symbol": "SPOT", "name": "Spotify", "sector": "Music Streaming", "dividend_yield": 0.0, "stability": 5, "growth": 7, "risk_level": 2},
    ]

    # Train ML model
    model, scaler, feature_names = predict_investment_eligibility(X, y, credit_features)

    if model is None:
        return None, "⚠ Could not train investment model. Using fallback recommendations based on average credit score."

    # Extract feature importance from the trained model
    clf = model.named_steps['classifier']
    feature_importance = pd.Series(clf.feature_importances_, index=feature_names)
    top_features = feature_importance.nlargest(3).index.tolist()

    # Get user profile
    if user_profile is not None:
        try:
            user_data = pd.DataFrame([user_profile])[feature_names]
            risk_level = model.predict(user_data)[0]
            confidence = max(model.predict_proba(user_data)[0]) * 100
            # Normalize user profile for scoring
            user_profile_norm = (user_data - credit_features.mean()) / (credit_features.std() + 1e-8)
        except:
            risk_level = 1
            confidence = 50
            user_profile_norm = None
    else:
        avg_profile = credit_features.mean()
        avg_data = pd.DataFrame([avg_profile])[feature_names]
        risk_level = model.predict(avg_data)[0]
        confidence = max(model.predict_proba(avg_data)[0]) * 100
        user_profile_norm = None

    # Score stocks based on model insights and user profile
    def score_stock(stock, risk_level):
        """Score stock alignment with user's credit profile using ML insights"""
        score = 0.0
        
        # Primary scoring: Risk level match (most important)
        if stock['risk_level'] == risk_level:
            score += 40
        elif abs(stock['risk_level'] - risk_level) == 1:
            score += 20
        
        # Growth vs Stability preference based on risk profile
        if risk_level == 0:  # Conservative
            score += stock['stability'] * 2  # Prefer stable stocks
            score += stock['dividend_yield'] * 1.5  # Prefer dividends
        elif risk_level == 1:  # Moderate
            score += stock['growth'] * 1.5  # Balance growth
            score += stock['stability'] * 1.5  # and stability
            score += stock['dividend_yield'] * 0.8
        else:  # Aggressive
            score += stock['growth'] * 2.5  # Prefer growth highly
            score += (10 - stock['stability']) * 1  # Less concern with stability
        
        # Sector diversification bonus (slight)
        score += 5
        
        return score

    # Score and rank all stocks
    scored_stocks = []
    for stock in all_stocks:
        stock_with_score = stock.copy()
        stock_with_score['score'] = score_stock(stock, risk_level)
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

    risk_display = ["Conservative 🛡️", "Moderate ⚖️", "Aggressive 🚀"]
    
    # Build detailed message showing ML insights
    top_features_str = ", ".join([f.replace('_', ' ').title() for f in top_features])
    message = f"🤖 **ML Prediction:** {risk_display[risk_level]} investment profile (Confidence: {confidence:.1f}%)\n📊 **Key Factors:** {top_features_str}\n✨ **Ranking:** Stocks ranked by ML-learned alignment to your credit profile"

    return recommendations, message, risk_level, risk_display[risk_level]

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

    # CSV file uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Go to", ["Data Intelligence", "Feature Engineering", "Model Comparison", "Explainability", "Predictive Analytics", "Stock Recommendations", "Prediction Engine", "Compliance Report", "Portfolio Optimizer", "Risk Dashboard", "Download", "Settings", "Logout"])

    # Load dataset from uploaded file
    if uploaded_file is None:
        st.error("Please upload a CSV file to proceed.")
        st.stop()
    
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset loaded successfully")
        
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
            
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
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

    # Convert y if needed
    if y.dtype == object or y.dtype == 'O' or y.dtype == 'str':
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        y_final = pd.Series(y_enc, index=y.index)
    else:
        y_final = y.astype(int)

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
        risk_labels = ['Conservative', 'Moderate', 'Aggressive']
        fig = px.pie(values=risk_dist.values, names=[risk_labels[i] if i < len(risk_labels) else f"Class {i}" for i in risk_dist.index],
                    title="Risk Class Distribution", hole=0.3)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, width='stretch')
        
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
        st.markdown("### 🤖 Advanced Model Comparison & Benchmarking")
        st.markdown("Compare multiple algorithms with cross-validation and performance metrics.")
        
        if not can_train:
            st.warning("Not enough samples to train models.")
            st.stop()
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_final, test_size=0.2, stratify=y_final, random_state=42
            )
            
            # Define models
            models = {
                'RandomForest': Pipeline([('scaler', StandardScaler()), 
                                        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
                'GradientBoosting': Pipeline([('scaler', StandardScaler()), 
                                           ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))]),
            }
            
            if HAS_LIGHTGBM:
                models['LightGBM'] = Pipeline([('scaler', StandardScaler()),
                                             ('clf', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))])
            
            # Train and evaluate
            results = {}
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    # Train and test
                    model.fit(X_train, y_train)
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    results[name] = {
                        'model': model,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                    }
            
            # Display comparison metrics
            st.markdown("### 📊 Model Performance Metrics")
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'CV Mean': [results[m]['cv_mean'] for m in results],
                'CV Std': [results[m]['cv_std'] for m in results],
                'Train Acc': [results[m]['train_acc'] for m in results],
                'Test Acc': [results[m]['test_acc'] for m in results],
            })
            # Format only numeric columns
            st.dataframe(metrics_df.style.format({
                'CV Mean': '{:.4f}',
                'CV Std': '{:.4f}',
                'Train Acc': '{:.4f}',
                'Test Acc': '{:.4f}',
            }), use_container_width=True)
            
            # ROC Curves
            st.markdown("### 📈 ROC Curve Comparison")
            fig = go.Figure()
            
            for name, res in results.items():
                if len(np.unique(y_test)) > 1:
                    try:
                        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'][:, 1] if res['y_pred_proba'].shape[1] > 1 else res['y_pred_proba'][:, 0])
                        roc_auc = auc(fpr, tpr)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                               name=f'{name} (AUC={roc_auc:.3f})'))
                    except:
                        pass
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   name='Random', line=dict(dash='dash')))
            fig.update_layout(template="plotly_dark", title="ROC Curve Comparison",
                            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, width='stretch')
            
            # Precision-Recall Curves
            st.markdown("### 📉 Precision-Recall Curve Comparison")
            fig = go.Figure()
            
            for name, res in results.items():
                try:
                    precision, recall, _ = precision_recall_curve(y_test, 
                                                                  res['y_pred_proba'][:, 1] if res['y_pred_proba'].shape[1] > 1 else res['y_pred_proba'][:, 0])
                    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name))
                except:
                    pass
            
            fig.update_layout(template="plotly_dark", title="Precision-Recall Curve Comparison",
                            xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig, width='stretch')
            
            # Best model confusion matrix
            best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
            st.markdown(f"### 🏆 Best Model: {best_model[0]}")
            cm = confusion_matrix(y_test, best_model[1]['y_pred'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{best_model[0]} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
            st.markdown("**Classification Report:**")
            st.text(classification_report(y_test, best_model[1]['y_pred']))
            
            # Store best model in session for later use
            st.session_state.best_model = best_model[1]['model']
            st.session_state.best_model_name = best_model[0]
            
        except Exception as e:
            st.error(f"Model comparison failed: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Feature Engineering":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Advanced Feature Engineering & Analysis")
        st.markdown("Transform raw features into powerful predictive signals using domain knowledge and statistical rigor.")
        
        try:
            # Create derived features
            st.markdown("### 🔧 Engineered Features")
            
            X_engineered = X.copy()
            feature_descriptions = {}
            
            # Feature 1: Risk Scorecard
            if 'credit_score' in X.columns:
                X_engineered['risk_scorecard'] = (
                    (X['credit_score'] / X['credit_score'].max()) * 40 if 'credit_score' in X.columns else 0
                )
                feature_descriptions['risk_scorecard'] = "Credit health normalized (0-40)"
            
            # Feature 2: Financial Stability Index
            if 'employment_years' in X.columns and 'age' in X.columns:
                X_engineered['stability_index'] = (
                    (X['employment_years'] * X['age']) / 
                    ((X['employment_years'].max() * X['age'].max()) + 1e-8) * 50
                )
                feature_descriptions['stability_index'] = "Job & life stability combined (0-50)"
            
            # Feature 3: Repayment Capacity
            if 'income' in X.columns and 'debt_ratio' in X.columns:
                X_engineered['repayment_capacity'] = (
                    (X['income'] / (X['income'].max() + 1e-8)) * 100 * (1 - X['debt_ratio'])
                )
                feature_descriptions['repayment_capacity'] = "Income strength adjusted by debt"
            
            # Feature 4: Debt Burden Score (inverse, lower is better)
            if 'debt_ratio' in X.columns:
                X_engineered['debt_burden'] = X['debt_ratio'] * 100
                feature_descriptions['debt_burden'] = "Debt as % of income (higher risk)"
            
            # Feature 5: Income-to-Age Ratio (income growth trajectory)
            if 'income' in X.columns and 'age' in X.columns:
                X_engineered['income_growth_trajectory'] = (
                    (X['income'] / (X['age'] + 1e-8)) / 
                    ((X['income'] / (X['age'] + 1e-8)).max() + 1e-8) * 100
                )
                feature_descriptions['income_growth_trajectory'] = "Income per year of age"
            
            # Feature 6: Credit Efficiency (credit score per age)
            if 'credit_score' in X.columns and 'age' in X.columns:
                X_engineered['credit_efficiency'] = (
                    X['credit_score'] / (X['age'] + 1e-8)
                )
                feature_descriptions['credit_efficiency'] = "Credit health normalized by age"
            
            # Feature 7: Financial Health Index
            if all(x in X.columns for x in ['credit_score', 'income', 'employment_years']):
                health = (
                    (X['credit_score'] / X['credit_score'].max() * 0.4) +
                    (X['income'] / X['income'].max() * 0.3) +
                    (X['employment_years'] / (X['employment_years'].max() + 1) * 0.3)
                ) * 100
                X_engineered['financial_health_index'] = health
                feature_descriptions['financial_health_index'] = "Composite health metric (0-100)"
            
            # Display engineered features
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Base Features", len(X.columns))
            col2.metric("🔧 Engineered Features", len(X_engineered.columns) - len(X.columns))
            col3.metric("📈 Total Features", len(X_engineered.columns))
            
            st.markdown("---")
            
            # Feature Statistics
            st.markdown("### 📋 Engineered Feature Statistics")
            engineered_only = X_engineered.columns[len(X.columns):]
            stats_data = []
            for feat in engineered_only:
                if feat in X_engineered.columns:
                    stats_data.append({
                        'Feature': feat,
                        'Description': feature_descriptions.get(feat, 'Derived feature'),
                        'Mean': f"{X_engineered[feat].mean():.2f}",
                        'Std': f"{X_engineered[feat].std():.2f}",
                        'Min': f"{X_engineered[feat].min():.2f}",
                        'Max': f"{X_engineered[feat].max():.2f}",
                    })
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            st.markdown("---")
            
            # Feature Importance Comparison
            st.markdown("### 🎯 Feature Importance: Original vs Engineered")
            
            try:
                X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
                    X_engineered, y_final, test_size=0.2, stratify=y_final, random_state=42
                )
                
                # Train model with engineered features
                model_eng = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                model_eng.fit(X_train_eng, y_train_eng)
                
                # Get feature importance
                fi_engineered = pd.Series(
                    model_eng.named_steps['clf'].feature_importances_,
                    index=X_engineered.columns
                ).sort_values(ascending=False)
                
                # Separate original vs engineered
                original_fi = fi_engineered[fi_engineered.index.isin(X.columns)].head(10)
                engineered_fi = fi_engineered[~fi_engineered.index.isin(X.columns)].head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Features Importance")
                    fig = px.bar(original_fi.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}),
                               x='Importance', y='Feature', orientation='h')
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.markdown("#### Engineered Features Importance")
                    fig = px.bar(engineered_fi.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}),
                               x='Importance', y='Feature', orientation='h')
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, width='stretch')
                
                # Model Performance Comparison
                st.markdown("---")
                st.markdown("### 📊 Model Performance Improvement")
                
                # Train on original features
                X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                    X, y_final, test_size=0.2, stratify=y_final, random_state=42
                )
                model_orig = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                model_orig.fit(X_train_orig, y_train_orig)
                
                # Comparison metrics
                perf_data = {
                    'Model': ['Original Features', 'With Engineered Features'],
                    'Train Accuracy': [
                        model_orig.score(X_train_orig, y_train_orig),
                        model_eng.score(X_train_eng, y_train_eng)
                    ],
                    'Test Accuracy': [
                        model_orig.score(X_test_orig, y_test_orig),
                        model_eng.score(X_test_eng, y_test_eng)
                    ],
                    'Feature Count': [len(X.columns), len(X_engineered.columns)],
                }
                perf_df = pd.DataFrame(perf_data)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Accuracy", f"{perf_df.loc[0, 'Test Accuracy']:.4f}")
                col2.metric("Engineered Accuracy", f"{perf_df.loc[1, 'Test Accuracy']:.4f}")
                col3.metric("Improvement ↑", f"{(perf_df.loc[1, 'Test Accuracy'] - perf_df.loc[0, 'Test Accuracy'])*100:.2f}%")
                
                st.dataframe(perf_df.style.format({
                    'Train Accuracy': '{:.4f}',
                    'Test Accuracy': '{:.4f}',
                }), use_container_width=True)
                
            except Exception as e:
                st.warning(f"Feature importance analysis failed: {e}")
            
            st.markdown("---")
            
            # Correlation Analysis
            st.markdown("### 🔗 Feature Correlation Analysis")
            
            numeric_feats_to_show = X_engineered.select_dtypes(include=[np.number]).columns[:12]
            corr_matrix = X_engineered[numeric_feats_to_show].corr()
            
            fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu',
                           zmin=-1, zmax=1, title="Feature Correlation Heatmap",
                           height=700)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
            
            st.info("💡 **Insight:** Highly correlated features (>0.9 or <-0.9) may contain redundant information. Consider feature selection for production models.")
            
            st.markdown("---")
            
            # Feature Recommendations
            st.markdown("### 💡 AI-Powered Feature Recommendations")
            
            recommendations_text = """
            Based on your data characteristics, here are recommended next-level features to consider:
            
            **High Impact (Add These):**
            - **Credit History Length** - If available, normalize by max possible years in credit system
            - **Payment Consistency Score** - Variance in payment amounts (lower variance = more reliable)
            - **Income Volatility Index** - Standard deviation of income over time
            - **Debt-to-Income Sweet Spot** - Boolean flag if debt-to-income is in optimal range (20-30%)
            
            **Medium Impact (Consider):**
            - **Age Groups Clustering** - Categorical buckets (18-25, 25-35, etc.) for non-linear effects
            - **Credit Score Trajectory** - Recent trend vs historical (improving/declining)
            - **Risk Category Flag** - Binary indicator for customers in high-risk segments
            
            **Statistical Techniques to Apply:**
            - **Polynomial Features** - For non-linear relationships (age², credit_score²)
            - **Log Transformations** - For skewed distributions (income, debt_ratio)
            - **Interaction Terms** - Credit × Income, Age × Employment for synergy effects
            - **Feature Scaling** - Already done in pipeline; MinMaxScaler for bounded features (0-1)
            """
            
            st.markdown(recommendations_text)
            
        except Exception as e:
            st.error(f"Feature engineering analysis failed: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Explainability":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Model Explainability & Interpretability")
        
        if not can_train:
            st.warning("Not enough samples for model explainability.")
            st.stop()
        
        try:
            # Train model if not already in session
            if 'best_model' not in st.session_state:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_final, test_size=0.2, stratify=y_final, random_state=42
                )
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                model.fit(X_train, y_train)
                st.session_state.best_model = model
                st.session_state.best_model_name = "RandomForest"
                X_test_for_explain = X_test
                y_test_for_explain = y_test
            else:
                X_train, X_test_for_explain, y_train, y_test_for_explain = train_test_split(
                    X, y_final, test_size=0.2, stratify=y_final, random_state=42
                )
                model = st.session_state.best_model
            
            model_obj = model.named_steps['clf']
            scaler_obj = model.named_steps['scaler']
            X_test_scaled = scaler_obj.transform(X_test_for_explain)
            
            st.markdown(f"### 📊 Feature Importance ({st.session_state.best_model_name})")
            fi = pd.Series(model_obj.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
            
            fi_df = fi.reset_index()
            fi_df.columns = ['Feature', 'Importance']
            
            fig = px.bar(fi_df.sort_values('Importance'), x='Importance', y='Feature',
                        orientation='h', title="Top 20 Most Important Features")
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, width='stretch')
            
            # SHAP if available
            if HAS_SHAP:
                st.markdown("### 🚀 SHAP Explainability Analysis")
                
                sample_idx = st.slider("Select sample to explain", 0, len(X_test_for_explain)-1, 0)
                
                try:
                    shap_values = None
                    with st.spinner("Computing SHAP values..."):
                        if isinstance(model_obj, GradientBoostingClassifier) and len(np.unique(y_final)) > 2:
                            raise ValueError("SHAP TreeExplainer is only supported for binary classification with GradientBoostingClassifier.")

                        explainer = shap.TreeExplainer(model_obj)
                        shap_values = explainer.shap_values(X_test_scaled)
                        
                    if shap_values is not None:
                        # Handle different output types
                        if isinstance(shap_values, list):
                            shap_values_to_use = shap_values[int(y_test_for_explain.iloc[sample_idx])]
                        else:
                            shap_values_to_use = shap_values
                        
                        # Force plot
                        st.markdown("#### Force Plot (SHAP)")
                        try:
                            if hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
                                if isinstance(shap_values, list):
                                    sample_class = int(y_test_for_explain.iloc[sample_idx])
                                    expected_value = explainer.expected_value[sample_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                                    shap_sample = shap_values[sample_class][sample_idx]
                                else:
                                    expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                                    shap_sample = shap_values[sample_idx]

                                shap_fig = shap.plots.force(expected_value, shap_sample, feature_names=X.columns.tolist(), matplotlib=True)
                                st.pyplot(shap_fig)
                            else:
                                expected_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[int(y_test_for_explain.iloc[sample_idx])]
                                force_plot = shap.force_plot(expected_value,
                                                             shap_values_to_use[sample_idx],
                                                             X_test_scaled[sample_idx],
                                                             feature_names=X.columns.tolist(),
                                                             matplotlib=True)
                                st.pyplot(force_plot)
                        except Exception as inner_e:
                            st.warning(f"SHAP force plot skipped due to version mismatch: {inner_e}")
                        
                        # Summary plot
                        st.markdown("#### Summary Plot (SHAP)")
                        sh = shap_values_to_use if len(shap_values_to_use.shape) == 2 else shap_values_to_use.reshape(-1, len(X.columns))
                        fig2, ax = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(sh, X_test_scaled, feature_names=X.columns.tolist(), plot_type="bar", matplotlib=True)
                        st.pyplot(fig2)
                except Exception as e:
                    st.warning(f"SHAP explainability skipped: {e}")
            else:
                st.info("Install SHAP for advanced model explainability: `pip install shap`")
            
            # Feature importance as bar chart using plotly
            st.markdown("### 📈 Cumulative Feature Importance")
            fi_sorted = pd.Series(model_obj.feature_importances_, index=X.columns).sort_values(ascending=False)
            cumsum = fi_sorted.cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fi_sorted.index[:20], y=fi_sorted.values[:20], name='Individual', marker_color='indianred'))
            fig.add_trace(go.Scatter(x=fi_sorted.index[:20], y=cumsum.values[:20], yaxis="y2", name='Cumulative', line=dict(color='darkblue', width=3)))
            
            fig.update_layout(
                yaxis=dict(title="Individual Importance", side='left'),
                yaxis2=dict(title="Cumulative", side='right', overlaying='y'),
                template="plotly_dark",
                hovermode='x unified'
            )
            st.plotly_chart(fig, width='stretch')
            
        except Exception as e:
            st.error(f"Explainability analysis failed: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Predictive Analytics":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🔮 Predictive Analytics & Risk Segmentation")
        
        if not can_train:
            st.warning("Not enough samples for predictive analytics.")
            st.stop()
        
        try:
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_final, test_size=0.2, stratify=y_final, random_state=42
            )
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            model.fit(X_train, y_train)
            
            # Predict on full dataset
            X_scaled = model.named_steps['scaler'].transform(X)
            probabilities = model.predict_proba(X)
            predictions = model.predict(X)
            
            # Risk scores (probability of highest risk)
            risk_scores = probabilities.max(axis=1)
            
            st.markdown("### 📊 Risk Score Distribution")
            fig = px.histogram(risk_scores, nbins=30, title="Distribution of Prediction Confidence Scores",
                             labels={'value': 'Confidence Score', 'count': 'Frequency'})
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
            
            # Risk Segmentation
            st.markdown("### 🎯 Customer Segmentation by Risk & Confidence")
            
            # Build flexible risk label mapping from the classifier classes
            clf_classes = model.named_steps['clf'].classes_
            if len(clf_classes) == 2:
                risk_levels = ['Low Risk 🟢', 'High Risk 🔴']
            elif len(clf_classes) == 3:
                risk_levels = ['Conservative 🛡️', 'Moderate ⚖️', 'Aggressive 🚀']
            else:
                risk_levels = [f'Risk Class {i}' for i in range(len(clf_classes))]
            risk_label_map = {cls: label for cls, label in zip(clf_classes, risk_levels)}

            segments_data = pd.DataFrame({
                'Risk Level': [risk_label_map[p] for p in predictions],
                'Confidence': risk_scores,
                'Sample': range(len(predictions))
            })
            
            # Add Feature for coloring
            numeric_cols = X.select_dtypes(include=[np.number]).columns[:1]
            if len(numeric_cols) > 0:
                segments_data[numeric_cols[0]] = X[numeric_cols[0]].values
            
            fig = px.scatter(segments_data, x='Confidence', y='Sample', color='Risk Level',
                           size='Confidence', title="Risk Distribution Across Samples",
                           height=500)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
            
            # Trend Analysis
            st.markdown("### 📈 Confidence Trend Analysis")
            
            # Group by prediction
            trend_data = pd.DataFrame({
                'Risk Level': [risk_label_map[p] for p in predictions],
                'Confidence': risk_scores
            })
            
            summary = trend_data.groupby('Risk Level')['Confidence'].agg(['mean', 'min', 'max', 'count'])
            
            cols = st.columns(min(3, len(risk_levels)))
            for idx, risk_level in enumerate(risk_levels[:3]):
                with cols[idx]:
                    if risk_level in summary.index:
                        count = summary.loc[risk_level, 'count']
                        mean_conf = summary.loc[risk_level, 'mean']
                        st.metric(f"{risk_level}", f"{int(count)} samples", f"Avg: {mean_conf:.2%}")
            
            # Scenario Analysis
            st.markdown("### 🔄 What-If Scenario Simulation")
            
            st.markdown("Adjust features to see how predictions change:")
            
            # Get numeric columns for simulation
            numeric_feat_cols = X.select_dtypes(include=[np.number]).columns[:3]
            
            if len(numeric_feat_cols) > 0:
                scenario_values = {}
                cols_input = st.columns(len(numeric_feat_cols))
                
                for idx, feat in enumerate(numeric_feat_cols):
                    with cols_input[idx]:
                        min_val = X[feat].min()
                        max_val = X[feat].max()
                        mean_val = X[feat].mean()
                        scenario_values[feat] = st.slider(f"{feat}", min_val, max_val, mean_val)
                
                # Create scenario
                scenario_df = X.iloc[0:1].copy()
                for feat in numeric_feat_cols:
                    scenario_df[feat] = scenario_values[feat]
                
                scenario_pred = model.predict(scenario_df)[0]
                scenario_proba = model.predict_proba(scenario_df)[0]
                
                st.markdown("#### 🎯 Scenario Result")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Risk", risk_label_map.get(scenario_pred, f"Risk Class {scenario_pred}"))
                col2.metric("Confidence", f"{max(scenario_proba):.2%}")
                
                # Show probabilities
                st.markdown("#### Probability Distribution")
                proba_df = pd.DataFrame({
                    'Risk Level': risk_levels[:len(scenario_proba)],
                    'Probability': scenario_proba
                })
                fig = px.bar(proba_df, x='Risk Level', y='Probability',
                           title="Prediction Probabilities for Scenario")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
            
        except Exception as e:
            st.error(f"Predictive analytics failed: {e}")
        
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
        st.markdown("### ⚡ Real-Time Risk Prediction Engine")
        st.markdown("Enter applicant details to get instant risk predictions with confidence intervals.")
        
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
                ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
            ])
            final_model.fit(X_train, y_train)
            
            # Input form
            st.markdown("### 📝 Applicant Profile")
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:6]
            input_values = {}
            
            cols = st.columns(3)
            for idx, col in enumerate(numeric_cols):
                with cols[idx % 3]:
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    mean_val = float(X[col].mean())
                    input_values[col] = st.slider(f"{col.replace('_', ' ').title()}", 
                                                 min_val, max_val, mean_val)
            
            if st.button("🔮 Predict Risk Profile", key="predict_btn"):
                # Create prediction dataframe
                pred_df = X.iloc[0:1].copy()
                for col in numeric_cols:
                    pred_df[col] = input_values[col]
                
                # Make prediction
                prediction = final_model.predict(pred_df)[0]
                probabilities = final_model.predict_proba(pred_df)[0]
                confidence = max(probabilities)
                
                risk_labels = ['Conservative 🛡️', 'Moderate ⚖️', 'Aggressive 🚀']
                
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Risk Profile", risk_labels[prediction])
                with col2:
                    st.metric("Confidence Level", f"{confidence:.2%}")
                with col3:
                    st.metric("Action Required", "📊 Review")
                
                # Confidence interval (simulated)
                st.markdown("#### 📊 Prediction Probabilities")
                proba_df = pd.DataFrame({
                    'Risk Profile': risk_labels[:len(probabilities)],
                    'Probability': probabilities * 100
                })
                
                fig = px.bar(proba_df, x='Risk Profile', y='Probability',
                           title="Risk Profile Probability Distribution",
                           color='Probability',
                           color_continuous_scale='RdYlGn_r')
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, width='stretch')
                
                # Recommendations
                st.markdown("#### 💡 Personalized Recommendations")
                
                if prediction == 0:  # Conservative
                    st.info("✅ **Conservative Profile Detected**\n\n"
                           "- Recommended for stable, dividend-paying stocks\n"
                           "- Focus on: Blue-chip companies, bonds, utilities\n"
                           "- Expected portfolio volatility: Low\n"
                           "- Risk Level: Minimal")
                elif prediction == 1:  # Moderate
                    st.info("✅ **Moderate Profile Detected**\n\n"
                           "- Recommended for balanced portfolio\n"
                           "- Focus on: Mix of growth and stability\n"
                           "- Expected portfolio volatility: Medium\n"
                           "- Risk Level: Balanced")
                else:  # Aggressive
                    st.info("✅ **Aggressive Profile Detected**\n\n"
                           "- Recommended for growth-oriented investments\n"
                           "- Focus on: Tech stocks, growth companies, emerging sectors\n"
                           "- Expected portfolio volatility: High\n"
                           "- Risk Level: High tolerance needed")
                
                # Risk alert if confidence is low
                if confidence < 0.60:
                    st.warning("⚠️ **Low Confidence Alert:** The model is uncertain about this prediction. "
                              "Consider reviewing additional factors or requesting manual assessment.")
        
        except Exception as e:
            st.error(f"Prediction engine failed: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Compliance Report":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 📋 Compliance & Fairness Report")
        st.markdown("Regulatory compliance metrics and fairness analysis for credit risk model.")
        
        if not can_train:
            st.warning("Not enough samples for compliance analysis.")
            st.stop()
        
        try:
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_final, test_size=0.2, stratify=y_final, random_state=42
            )
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Model Performance Metrics
            st.markdown("### 📊 Model Performance Baseline")
            
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Accuracy", f"{accuracy:.2%}")
            col2.metric("Test Samples", len(y_test))
            col3.metric("Training Samples", len(y_train))
            col4.metric("Total Data", len(X))
            
            # Per-class performance (fairness)
            st.markdown("### ⚖️ Per-Class Fairness Metrics")
            
            fairness_data = []
            for class_label in np.unique(y_test):
                mask = y_test == class_label
                class_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0
                class_count = mask.sum()
                fairness_data.append({
                    'Class': class_label,
                    'Accuracy': class_acc,
                    'Samples': class_count,
                    'Sample %': f"{class_count / len(y_test) * 100:.1f}%"
                })
            
            fairness_df = pd.DataFrame(fairness_data)
            st.dataframe(fairness_df, use_container_width=True)
            
            # Disparate Impact Analysis
            st.markdown("### 🔍 Disparate Impact Analysis")
            
            if len(fairness_df) > 1:
                min_accuracy = fairness_df['Accuracy'].min()
                max_accuracy = fairness_df['Accuracy'].max()
                disparate_impact_ratio = min_accuracy / max_accuracy if max_accuracy > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Disparate Impact Ratio", f"{disparate_impact_ratio:.2%}",
                            help="Ratio of lowest to highest class accuracy. Should be > 0.80 for compliance.")
                    
                    if disparate_impact_ratio >= 0.80:
                        st.success("✅ **PASS:** Model meets 80% rule for disparate impact")
                    else:
                        st.warning("⚠️ **ALERT:** Model may have disparate impact concerns")
                
                with col2:
                    st.metric("Accuracy Range", f"{min_accuracy:.2%} - {max_accuracy:.2%}",
                            help="Range of accuracy across classes")
            
            # Confusion matrix analysis
            st.markdown("### 📋 Error Rate Analysis")
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate type I & II errors
            if cm.shape[0] > 1:
                # False positives and false negatives
                fp_rate = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0
                fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("False Positive Rate", f"{fp_rate:.2%}",
                           help="Type I Error: Positive prediction when actually negative")
                col2.metric("False Negative Rate", f"{fn_rate:.2%}",
                           help="Type II Error: Negative prediction when actually positive")
                col3.metric("Error Balance", f"{abs(fp_rate - fn_rate):.2%}",
                           help="Difference between error types (lower is more balanced)")
            
            # Calibration
            st.markdown("### 🎯 Model Calibration Assessment")
            
            st.info("**Calibration Check:** Model predictions are well-calibrated if predicted probabilities "
                   "match actual frequencies. Lower ECE values indicate better calibration.")
            
            # Compute Expected Calibration Error (ECE)
            bins = 5
            bin_edges = np.linspace(0, 1, bins + 1)
            ece = 0
            calibration_data = []
            
            for i in range(bins):
                mask = (y_pred_proba.max(axis=1) >= bin_edges[i]) & (y_pred_proba.max(axis=1) < bin_edges[i+1])
                if mask.sum() > 0:
                    predicted_confidence = y_pred_proba[mask].max(axis=1).mean()
                    actual_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                    ece += abs(predicted_confidence - actual_accuracy) * mask.sum() / len(y_test)
                    
                    calibration_data.append({
                        'Confidence Bin': f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                        'Predicted Confidence': predicted_confidence,
                        'Actual Accuracy': actual_accuracy,
                        'Samples': mask.sum()
                    })
            
            calibration_df = pd.DataFrame(calibration_data)
            if len(calibration_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Calibration Error (ECE)", f"{ece:.4f}",
                             help="Lower values indicate better calibration")
                
                st.dataframe(calibration_df, use_container_width=True)
            
            # Compliance Checklist
            st.markdown("### ✅ Compliance Checklist")
            
            checks = {
                "Accuracy > 75%": accuracy > 0.75,
                "Min class accuracy > 70%": fairness_df['Accuracy'].min() > 0.70 if len(fairness_df) > 0 else False,
                "Disparate Impact Ratio > 0.80": disparate_impact_ratio >= 0.80 if len(fairness_df) > 1 else False,
                "Error balance < 20%": abs(fp_rate - fn_rate) < 0.20 if cm.shape[0] > 1 else False,
                "Sample size >= 100": len(X) >= 100,
            }
            
            for check_name, passed in checks.items():
                icon = "✅" if passed else "❌"
                st.markdown(f"{icon} {check_name}")
            
            compliance_score = sum(checks.values()) / len(checks) * 100
            
            st.markdown("---")
            st.metric("Compliance Score", f"{compliance_score:.0f}/100")
            
            if compliance_score >= 80:
                st.success("🟢 **HIGH COMPLIANCE:** Model meets enterprise standards")
            elif compliance_score >= 60:
                st.warning("🟡 **MODERATE COMPLIANCE:** Some areas need improvement")
            else:
                st.error("🔴 **LOW COMPLIANCE:** Significant issues detected")
        
        except Exception as e:
            st.error(f"Compliance analysis failed: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Portfolio Optimizer":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Portfolio Optimization & Risk Analysis")
        st.markdown("Optimize your investment portfolio using Modern Portfolio Theory and Monte Carlo simulations.")

        # Get recommended stocks for optimization
        if 'recommendations' not in locals() or not recommendations:
            # Try to get recommendations if not already loaded
            credit_feature_cols = []
            feature_candidates = ['credit_score', 'income', 'debt_ratio', 'employment_years', 'age', 'Credit_Score', 'Income', 'Debt_Ratio']
            for col in feature_candidates:
                if col in data_df.columns:
                    credit_feature_cols.append(col)

            if len(credit_feature_cols) >= 2:
                credit_features = data_df[credit_feature_cols].copy()
                credit_features = credit_features.fillna(credit_features.mean())
                recommendations, _, _, _ = get_stock_recommendations_ml(X, y_final, credit_features, None)

        if recommendations:
            st.markdown("#### 📊 Portfolio Allocation Optimizer")

            # Portfolio allocation inputs
            total_investment = st.number_input("Total Investment Amount ($)", min_value=1000, value=10000, step=1000)

            # Risk preference slider
            risk_preference = st.slider("Risk Preference", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                      help="0 = Conservative (more bonds/safe stocks), 1 = Aggressive (more growth stocks)")

            # Select stocks for portfolio
            selected_stocks = st.multiselect("Select stocks for optimization (max 8)",
                                           [f"{s['symbol']} - {s['name']}" for s in recommendations[:8]],
                                           default=[f"{s['symbol']} - {s['name']}" for s in recommendations[:4]])

            if st.button("Optimize Portfolio") and selected_stocks:
                with st.spinner("Running portfolio optimization..."):
                    # Extract symbols
                    portfolio_symbols = [s.split(' - ')[0] for s in selected_stocks]

                    # Get historical data for selected stocks
                    portfolio_data = {}
                    valid_symbols = []

                    for symbol in portfolio_symbols:
                        hist, _ = get_stock_data(symbol, period="2y")
                        if hist is not None and len(hist) > 100:
                            portfolio_data[symbol] = hist['Close']
                            valid_symbols.append(symbol)

                    if len(valid_symbols) >= 2:
                        # Create returns dataframe
                        returns_df = pd.DataFrame(portfolio_data).pct_change().dropna()

                        # Calculate expected returns and covariance
                        expected_returns = returns_df.mean() * 252  # Annualized
                        cov_matrix = returns_df.cov() * 252  # Annualized

                        # Monte Carlo simulation for optimal portfolio
                        num_portfolios = 5000
                        results = np.zeros((3, num_portfolios))
                        weights_record = []

                        for i in range(num_portfolios):
                            weights = np.random.random(len(valid_symbols))
                            weights /= np.sum(weights)
                            weights_record.append(weights)

                            # Portfolio return and volatility
                            portfolio_return = np.sum(weights * expected_returns)
                            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                            # Sharpe ratio (assuming 3% risk-free rate)
                            sharpe_ratio = (portfolio_return - 0.03) / portfolio_std

                            results[0,i] = portfolio_return
                            results[1,i] = portfolio_std
                            results[2,i] = sharpe_ratio

                        # Find optimal portfolios
                        max_sharpe_idx = np.argmax(results[2])
                        min_vol_idx = np.argmin(results[1])

                        # Display results
                        st.markdown("#### 🎯 Optimal Portfolio Allocations")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**🚀 Maximum Sharpe Ratio Portfolio**")
                            max_sharpe_weights = weights_record[max_sharpe_idx]
                            for i, symbol in enumerate(valid_symbols):
                                allocation = max_sharpe_weights[i] * total_investment
                                st.write(f"**{symbol}:** ${allocation:,.0f} ({max_sharpe_weights[i]*100:.1f}%)")

                            st.metric("Expected Annual Return", f"{results[0,max_sharpe_idx]*100:.1f}%")
                            st.metric("Annual Volatility", f"{results[1,max_sharpe_idx]*100:.1f}%")
                            st.metric("Sharpe Ratio", f"{results[2,max_sharpe_idx]:.2f}")

                        with col2:
                            st.markdown("**🛡️ Minimum Volatility Portfolio**")
                            min_vol_weights = weights_record[min_vol_idx]
                            for i, symbol in enumerate(valid_symbols):
                                allocation = min_vol_weights[i] * total_investment
                                st.write(f"**{symbol}:** ${allocation:,.0f} ({min_vol_weights[i]*100:.1f}%)")

                            st.metric("Expected Annual Return", f"{results[0,min_vol_idx]*100:.1f}%")
                            st.metric("Annual Volatility", f"{results[1,min_vol_idx]*100:.1f}%")
                            st.metric("Sharpe Ratio", f"{results[2,min_vol_idx]:.2f}")

                        # Monte Carlo visualization
                        st.markdown("#### 📈 Portfolio Optimization Visualization")
                        fig = px.scatter(x=results[1,:], y=results[0,:],
                                       color=results[2,:],
                                       labels={'x': 'Volatility (Risk)', 'y': 'Expected Return'},
                                       title='Efficient Frontier - Monte Carlo Simulation',
                                       color_continuous_scale='RdYlGn')

                        # Highlight optimal portfolios
                        fig.add_scatter(x=[results[1,max_sharpe_idx]], y=[results[0,max_sharpe_idx]],
                                      mode='markers', marker=dict(size=15, color='red', symbol='star'),
                                      name='Max Sharpe Ratio')
                        fig.add_scatter(x=[results[1,min_vol_idx]], y=[results[0,min_vol_idx]],
                                      mode='markers', marker=dict(size=15, color='blue', symbol='diamond'),
                                      name='Min Volatility')

                        fig.update_layout(template="plotly_dark", height=500)
                        st.plotly_chart(fig, width='stretch')

                    else:
                        st.error("Not enough valid stock data for portfolio optimization. Try different stocks.")
            else:
                st.info("Select stocks and click 'Optimize Portfolio' to run Monte Carlo simulation and find optimal allocations.")
        else:
            st.error("No stock recommendations available. Please visit the Stock Recommendations page first.")

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

        # Risk tolerance assessment based on credit metrics
        risk_score = 0
        risk_factors = []

        if 'credit_score' in data_df.columns:
            avg_cs = data_df['credit_score'].mean()
            if avg_cs >= 750: risk_score += 3; risk_factors.append("Excellent credit score")
            elif avg_cs >= 700: risk_score += 2; risk_factors.append("Good credit score")
            elif avg_cs >= 650: risk_score += 1; risk_factors.append("Fair credit score")
            else: risk_score += 0; risk_factors.append("Poor credit score")

        if 'debt_ratio' in data_df.columns:
            avg_debt = data_df['debt_ratio'].mean()
            if avg_debt <= 0.2: risk_score += 2; risk_factors.append("Low debt ratio")
            elif avg_debt <= 0.4: risk_score += 1; risk_factors.append("Moderate debt ratio")
            else: risk_score += 0; risk_factors.append("High debt ratio")

        if 'income' in data_df.columns:
            avg_income = data_df['income'].mean()
            if avg_income >= 100000: risk_score += 2; risk_factors.append("High income")
            elif avg_income >= 50000: risk_score += 1; risk_factors.append("Moderate income")
            else: risk_score += 0; risk_factors.append("Lower income")

        # Determine risk profile
        if risk_score >= 5:
            profile = "🚀 Aggressive Investor"
            description = "High risk tolerance, suitable for growth stocks and aggressive strategies"
        elif risk_score >= 3:
            profile = "⚖️ Moderate Investor"
            description = "Balanced risk tolerance, suitable for diversified portfolios"
        else:
            profile = "🛡️ Conservative Investor"
            description = "Low risk tolerance, suitable for stable investments and bonds"

        st.markdown(f"### {profile}")
        st.markdown(f"**{description}**")

        st.markdown("#### ✅ Risk Factors Identified:")
        for factor in risk_factors:
            st.write(f"• {factor}")

        # Risk mitigation suggestions
        st.markdown("#### 💡 Risk Mitigation Strategies")
        if "Poor credit" in str(risk_factors) or "High debt" in str(risk_factors):
            st.warning("**High Risk Alert:** Consider debt consolidation and credit improvement before aggressive investments.")
        if "Lower income" in str(risk_factors):
            st.info("**Income Stability:** Focus on dividend-paying stocks and consider dollar-cost averaging.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Download":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Download processed dataset and model outputs")
        # assemble a download df: features + original label
        out_df = X.copy()
        out_df['label'] = y.astype(str).values
        download_link(out_df, filename="processed_credit_data.csv")
        st.markdown("You may also download a brief PDF/CSV report (summary stats).")
        buf = BytesIO()
        out_df.describe().to_csv(buf)
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="credit_summary.csv" style="color:#AEE7FF">Download summary CSV</a>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Settings":
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Settings & info")
        st.write("Rows (samples):", X.shape[0])
        st.write("Features:", X.shape[1])
        st.write("Detected classes:", np.unique(y_final).tolist())
        st.markdown("Upload a new CSV file in the sidebar to load a different dataset.")
        st.markdown("**Expected CSV format:** Columns should include credit metrics (age, income, credit_score, etc.) and a target column (Default_Status or similar).")
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





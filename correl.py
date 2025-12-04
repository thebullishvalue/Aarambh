import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Dependencies Check ---
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.regression.quantile_regression import QuantReg
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.decomposition import PCA
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Page Config ---
st.set_page_config(
    page_title="Regression Lab Pro | Quant Correlate",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium CSS ---
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary-color: #FFC300;
            --primary-rgb: 255, 195, 0;
            --background-color: #0F0F0F;
            --secondary-background-color: #1A1A1A;
            --bg-card: #1A1A1A;
            --bg-elevated: #2A2A2A;
            --text-primary: #EAEAEA;
            --text-secondary: #EAEAEA;
            --text-muted: #888888;
            --border-color: #2A2A2A;
            --border-light: #3A3A3A;
            
            --success-green: #10b981;
            --danger-red: #ef4444;
            --warning-amber: #f59e0b;
            --info-cyan: #06b6d4;
            --purple: #8b5cf6;
            --pink: #ec4899;
            
            --neutral: #888888;
        }
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main, [data-testid="stSidebar"] {
            background-color: var(--background-color);
            color: var(--text-primary);
        }
        
        .stApp > header {
            background-color: transparent;
        }
        
        .block-container {
            padding-top: 1rem;
            max-width: 1400px;
        }
        
        /* Premium Header */
        .premium-header {
            background: var(--secondary-background-color);
            padding: 1.25rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
            margin-top: 2.5rem;
        }
        
        .premium-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .premium-header h1 {
            margin: 0;
            font-size: 2.50rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.50px;
            position: relative;
        }
        
        .premium-header .tagline {
            color: var(--text-muted);
            font-size: 1rem;
            margin-top: 0.25rem;
            font-weight: 400;
            position: relative;
        }
        
        /* Metric Card */
        .metric-card {
            background-color: var(--bg-card);
            padding: 1.25rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
            margin-bottom: 0.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            border-color: var(--border-light);
        }
        
        .metric-card h4 {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-card h2 {
            color: var(--text-primary);
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            line-height: 1;
        }
        
        .metric-card .sub-metric {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-weight: 500;
        }

        .metric-card.primary h2 { color: var(--primary-color); }
        .metric-card.success h2 { color: var(--success-green); }
        .metric-card.danger h2 { color: var(--danger-red); }
        .metric-card.info h2 { color: var(--info-cyan); }
        .metric-card.gold h2 { color: var(--primary-color); }
        .metric-card.purple h2 { color: var(--purple); }
        .metric-card.pink h2 { color: var(--pink); }

        /* Guide Box */
        .guide-box {
            background: rgba(var(--primary-rgb), 0.05);
            border-left: 3px solid var(--primary-color);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .interpretation-item {
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        .interp-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }
        
        .feature-tag {
            background: rgba(255, 195, 0, 0.1);
            color: var(--primary-color);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .equation-box {
            font-family: 'Courier New', monospace;
            background: #000000;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            color: var(--success-green);
            overflow-x: auto;
            white-space: nowrap;
            margin: 1rem 0;
        }
        
        /* Model Comparison Cards */
        .model-compare-card {
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .model-compare-card.winner {
            border-color: var(--success-green);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
        }
        
        .model-compare-card h3 {
            color: var(--primary-color);
            margin: 0 0 1rem 0;
            font-size: 1.1rem;
        }
        
        /* Table Styling */
        .stMarkdown table {
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        .stMarkdown table th {
            background-color: var(--bg-elevated);
            color: var(--text-primary);
            font-weight: 600;
            text-align: left;
            padding: 12px 10px;
        }
        
        .stMarkdown table td {
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 12px 10px;
        }

        /* Buttons */
        .stButton>button {
            border: 2px solid var(--primary-color);
            background: transparent;
            color: var(--primary-color);
            font-weight: 700;
            border-radius: 12px;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background: var(--primary-color);
            color: #1A1A1A;
        }
        
        /* Footer */
        .app-footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
        }
        
        /* New Feature Badges */
        .new-badge {
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-left: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- Helper Functions ---

def clean_data(df, target, features, date_col=None):
    """Cleans dataframe for specific columns and optional date."""
    cols = [target] + features
    if date_col and date_col != "None" and date_col in df.columns:
        cols.append(date_col)
    
    data = df[cols].copy()
    
    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    numeric_subset = data[[target] + features]
    is_finite = np.isfinite(numeric_subset).all(axis=1)
    data = data[is_finite]
    
    if date_col and date_col != "None" and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col])
        except:
            pass
            
    return data

def run_regression(data, target, features):
    """Runs OLS regression on multiple features."""
    if not STATSMODELS_AVAILABLE:
        return None, "Statsmodels library missing."
    try:
        y = data[target]
        X = data[features]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model, None
    except Exception as e:
        return None, str(e)

def render_metric(label, value, sub=None, style="gold", tooltip=""):
    """Renders a metric card."""
    tooltip_attr = f'title="{tooltip}"' if tooltip else ''
    st.markdown(f"""
    <div class="metric-card {style}" {tooltip_attr}>
        <h4>{label}</h4>
        <h2>{value}</h2>
        {f'<div class="sub-metric">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)

def update_chart_theme(fig):
    """Applies the dark theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font_color="#EAEAEA",
        xaxis=dict(gridcolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A"),
        margin=dict(t=40, l=20, r=20, b=20)
    )
    return fig

# ============================================================================
# NEW FEATURE 1: ADVANCED REGRESSION TYPES
# ============================================================================

def run_ridge_regression(X, y, alpha=1.0):
    """Ridge Regression (L2 regularization)"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    return model, scaler

def run_lasso_regression(X, y, alpha=1.0):
    """Lasso Regression (L1 regularization)"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_scaled, y)
    return model, scaler

def run_elastic_net(X, y, alpha=1.0, l1_ratio=0.5):
    """Elastic Net (L1 + L2)"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_scaled, y)
    return model, scaler

def run_huber_regression(X, y, epsilon=1.35):
    """Huber Robust Regression"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = HuberRegressor(epsilon=epsilon, max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler

def run_ransac_regression(X, y):
    """RANSAC Robust Regression"""
    model = RANSACRegressor(random_state=42)
    model.fit(X, y)
    return model, None

def run_quantile_regression(data, target, features, quantile=0.5):
    """Quantile Regression"""
    y = data[target]
    X = sm.add_constant(data[features])
    model = QuantReg(y, X).fit(q=quantile)
    return model

# ============================================================================
# NEW FEATURE 2: ROLLING WINDOW ANALYSIS
# ============================================================================

def rolling_regression(data, target, features, window_size, date_col=None):
    """Performs rolling OLS regression and returns coefficient time series."""
    if date_col and date_col != "None" and date_col in data.columns:
        data_sorted = data.sort_values(by=date_col).reset_index(drop=True)
    else:
        data_sorted = data.sort_index().reset_index(drop=True)
    
    results = []
    
    for i in range(window_size, len(data_sorted) + 1):
        window_data = data_sorted.iloc[i-window_size:i]
        y = window_data[target]
        X = sm.add_constant(window_data[features])
        
        try:
            model = sm.OLS(y, X).fit()
            row = {
                'window_end': i - 1,
                'r_squared': model.rsquared,
                'const': model.params['const']
            }
            for feat in features:
                row[f'coef_{feat}'] = model.params[feat]
                row[f'pval_{feat}'] = model.pvalues[feat]
            
            if date_col and date_col != "None" and date_col in data_sorted.columns:
                row['date'] = data_sorted.iloc[i-1][date_col]
            
            results.append(row)
        except:
            continue
    
    return pd.DataFrame(results)

def detect_structural_breaks(rolling_df, feature_cols, threshold=2.0):
    """Detects structural breaks using coefficient volatility."""
    breaks = []
    
    for feat in feature_cols:
        coef_col = f'coef_{feat}'
        if coef_col in rolling_df.columns:
            coef_series = rolling_df[coef_col]
            rolling_std = coef_series.rolling(10).std()
            mean_std = rolling_std.mean()
            
            # Find periods where volatility exceeds threshold
            high_vol_periods = rolling_df[rolling_std > mean_std * threshold]
            if len(high_vol_periods) > 0:
                breaks.append({
                    'feature': feat,
                    'break_periods': high_vol_periods.index.tolist(),
                    'volatility_ratio': (rolling_std / mean_std).max()
                })
    
    return breaks

# ============================================================================
# NEW FEATURE 3: FEATURE ENGINEERING MODULE
# ============================================================================

def generate_lags(data, columns, max_lag=3):
    """Generates lagged features."""
    result = data.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            result[f'{col}_lag{lag}'] = result[col].shift(lag)
    return result

def generate_differences(data, columns, periods=1):
    """Generates differenced features."""
    result = data.copy()
    for col in columns:
        result[f'{col}_diff{periods}'] = result[col].diff(periods)
    return result

def generate_rolling_stats(data, columns, window=5):
    """Generates rolling statistics."""
    result = data.copy()
    for col in columns:
        result[f'{col}_roll_mean{window}'] = result[col].rolling(window).mean()
        result[f'{col}_roll_std{window}'] = result[col].rolling(window).std()
    return result

def generate_interactions(data, columns):
    """Generates interaction terms."""
    result = data.copy()
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            result[f'{col1}_x_{col2}'] = result[col1] * result[col2]
    return result

def generate_polynomial_features(data, columns, degree=2):
    """Generates polynomial features."""
    result = data.copy()
    for col in columns:
        for d in range(2, degree + 1):
            result[f'{col}_pow{d}'] = result[col] ** d
    return result

def apply_log_transform(data, columns):
    """Applies log transformation (handles negatives with sign preservation)."""
    result = data.copy()
    for col in columns:
        # Log transform with sign preservation for negative values
        result[f'{col}_log'] = np.sign(result[col]) * np.log1p(np.abs(result[col]))
    return result

def winsorize_data(data, columns, limits=(0.05, 0.05)):
    """Winsorizes data to handle outliers."""
    result = data.copy()
    for col in columns:
        lower = result[col].quantile(limits[0])
        upper = result[col].quantile(1 - limits[1])
        result[f'{col}_winsor'] = result[col].clip(lower=lower, upper=upper)
    return result

def run_pca_preview(data, columns, n_components=None):
    """Runs PCA and returns explained variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[columns].dropna())
    
    if n_components is None:
        n_components = min(len(columns), len(X_scaled))
    
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'feature_names': columns
    }

# ============================================================================
# NEW FEATURE 4: MODEL COMPARISON FRAMEWORK
# ============================================================================

def compare_models(data, target, feature_sets, date_col=None):
    """Compares multiple models with different feature sets."""
    results = []
    
    for name, features in feature_sets.items():
        try:
            clean = clean_data(data, target, features, date_col)
            if len(clean) < 10:
                continue
                
            model, err = run_regression(clean, target, features)
            if err:
                continue
            
            y = clean[target]
            X = sm.add_constant(clean[features])
            preds = model.predict(X)
            
            results.append({
                'Model': name,
                'Features': len(features),
                'R²': model.rsquared,
                'Adj R²': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic,
                'RMSE': np.sqrt(mean_squared_error(y, preds)),
                'MAE': mean_absolute_error(y, preds),
                'F-stat': model.fvalue,
                'F p-val': model.f_pvalue,
                'N': len(clean)
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

def nested_model_f_test(model_restricted, model_full, n_obs):
    """F-test for nested models."""
    ssr_r = model_restricted.ssr
    ssr_f = model_full.ssr
    df_r = model_restricted.df_resid
    df_f = model_full.df_resid
    
    if df_r <= df_f or ssr_f >= ssr_r:
        return None, None
    
    f_stat = ((ssr_r - ssr_f) / (df_r - df_f)) / (ssr_f / df_f)
    p_value = 1 - stats.f.cdf(f_stat, df_r - df_f, df_f)
    
    return f_stat, p_value

# ============================================================================
# NEW FEATURE 5: ENHANCED BACKTESTING
# ============================================================================

def walk_forward_backtest(data, target, features, n_splits=5, train_ratio=0.8, date_col=None):
    """Walk-forward optimization with rolling retrain."""
    if date_col and date_col != "None" and date_col in data.columns:
        data_sorted = data.sort_values(by=date_col).reset_index(drop=True)
    else:
        data_sorted = data.sort_index().reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(data_sorted)):
        train_data = data_sorted.iloc[train_idx]
        test_data = data_sorted.iloc[test_idx]
        
        try:
            y_train = train_data[target]
            X_train = sm.add_constant(train_data[features])
            
            model = sm.OLS(y_train, X_train).fit()
            
            X_test = sm.add_constant(test_data[features])
            preds = model.predict(X_test)
            
            y_test = test_data[target]
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            
            # Directional accuracy
            if len(y_test) > 1:
                actual_dir = np.sign(y_test.diff().dropna())
                pred_dir = np.sign(pd.Series(preds).diff().dropna())
                min_len = min(len(actual_dir), len(pred_dir))
                dir_acc = (actual_dir.values[:min_len] == pred_dir.values[:min_len]).mean()
            else:
                dir_acc = np.nan
            
            results.append({
                'Fold': fold + 1,
                'Train Size': len(train_data),
                'Test Size': len(test_data),
                'R²': model.rsquared,
                'RMSE': rmse,
                'MAE': mae,
                'Dir Accuracy': dir_acc,
                'Test Start': test_idx[0],
                'Test End': test_idx[-1]
            })
        except:
            continue
    
    return pd.DataFrame(results)

def monte_carlo_simulation(model, data, features, n_simulations=1000):
    """Bootstrap confidence intervals on predictions."""
    n_obs = len(data)
    predictions = []
    
    for _ in range(n_simulations):
        # Bootstrap sample
        boot_idx = np.random.choice(n_obs, size=n_obs, replace=True)
        boot_data = data.iloc[boot_idx]
        
        try:
            y_boot = boot_data.iloc[:, 0]  # Target is first column
            X_boot = sm.add_constant(boot_data[features])
            
            boot_model = sm.OLS(y_boot, X_boot).fit()
            
            # Predict on original data
            X_orig = sm.add_constant(data[features])
            preds = boot_model.predict(X_orig)
            predictions.append(preds.values)
        except:
            continue
    
    if len(predictions) == 0:
        return None
    
    predictions = np.array(predictions)
    
    return {
        'mean': np.mean(predictions, axis=0),
        'std': np.std(predictions, axis=0),
        'ci_lower': np.percentile(predictions, 2.5, axis=0),
        'ci_upper': np.percentile(predictions, 97.5, axis=0),
        'ci_5': np.percentile(predictions, 5, axis=0),
        'ci_95': np.percentile(predictions, 95, axis=0)
    }

def calculate_trading_metrics(actual, predicted):
    """Calculates trading-specific metrics."""
    # Direction accuracy
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    hit_rate = (actual_dir == pred_dir).mean()
    
    # Simulated PnL (assuming we trade based on predicted direction)
    returns = np.diff(actual)
    position = pred_dir
    pnl = returns * position[:-1] if len(position) > len(returns) else returns * position
    
    # Profit factor
    gross_profit = pnl[pnl > 0].sum() if len(pnl[pnl > 0]) > 0 else 0
    gross_loss = abs(pnl[pnl < 0].sum()) if len(pnl[pnl < 0]) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss
    
    # Sharpe-like ratio (simplified)
    if pnl.std() > 0:
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
    
    return {
        'hit_rate': hit_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'total_pnl': pnl.sum(),
        'avg_pnl': pnl.mean(),
        'max_drawdown': (np.cumsum(pnl) - np.maximum.accumulate(np.cumsum(pnl))).min()
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Premium Header
    st.markdown("""
    <div class="premium-header">
        <h1>Regression Lab <span style="color:#FFC300;">Pro</span></h1>
        <div class="tagline">Advanced Multi-Variable Modeling & Predictive Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.markdown("### 📁 Data Setup")
    uploaded_file = st.sidebar.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Data must have at least 2 numeric columns.")
                return

            st.sidebar.markdown("---")
            target_col = st.sidebar.selectbox("🎯 Target (Y)", numeric_cols)
            available_features = [c for c in numeric_cols if c != target_col]
            feature_cols = st.sidebar.multiselect("📊 Predictors (X)", available_features, default=available_features[:1])
            
            st.sidebar.markdown("---")
            date_candidates = [c for c in all_cols if 'date' in c.lower() or 'time' in c.lower()]
            default_date_idx = all_cols.index(date_candidates[0]) + 1 if date_candidates else 0
            
            date_col_option = st.sidebar.selectbox(
                "📅 Date Reference (Optional)", 
                ["None"] + all_cols,
                index=default_date_idx if default_date_idx < len(all_cols) + 1 else 0
            )
            
            if not feature_cols:
                st.info("👈 Select at least one predictor variable to begin.")
                return

            data = clean_data(df, target_col, feature_cols, date_col_option)
            if len(data) < 5:
                st.error("Not enough clean data points for regression (>5 needed).")
                return

            model, err = run_regression(data, target_col, feature_cols)
            
            if err:
                st.error(f"Model Error: {err}")
                return

            # --- TABS ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
                "📊 Model Performance", 
                "📐 Coefficients", 
                "🛠 Diagnostics",
                "🔍 Prediction Analysis",
                "🌊 Delta Analysis",
                "🔮 Simulation",
                "⚙️ Advanced Regression",
                "📈 Rolling Analysis",
                "🧪 Feature Engineering",
                "🏆 Model Comparison"
            ])

            # ================================================================
            # TAB 1: MODEL PERFORMANCE (Original)
            # ================================================================
            with tab1:
                c1, c2, c3, c4 = st.columns(4)
                
                r2 = model.rsquared
                adj_r2 = model.rsquared_adj
                rmse = np.sqrt(model.mse_resid)
                f_pval = model.f_pvalue
                
                with c1: render_metric("R-Squared", f"{r2:.4f}", "Fit Quality", "gold")
                with c2: render_metric("Adj. R-Squared", f"{adj_r2:.4f}", "Robust Fit", "gold")
                with c3: render_metric("RMSE", f"{rmse:.4f}", "Avg Error", "success")
                with c4: 
                    sig_style = "success" if f_pval < 0.05 else "danger"
                    sig_text = "Significant" if f_pval < 0.05 else "Not Sig."
                    render_metric("Model P-Value", f"{f_pval:.4e}", sig_text, sig_style)

                st.markdown("---")

                c_left, c_right = st.columns([2, 1])
                
                with c_left:
                    preds = model.predict(sm.add_constant(data[feature_cols]))
                    
                    fig_pred = px.scatter(x=data[target_col], y=preds, 
                                          labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                          title=f"Actual vs Predicted: {target_col}")
                    
                    min_val = min(data[target_col].min(), preds.min())
                    max_val = max(data[target_col].max(), preds.max())
                    fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                     line=dict(color="#FFC300", dash="dash"))
                    
                    fig_pred.update_traces(marker=dict(color="#06b6d4", size=8, opacity=0.7))
                    st.plotly_chart(update_chart_theme(fig_pred), use_container_width=True)
                
                with c_right:
                    st.markdown("""
                    <div class="guide-box">
                        <h4 style="color:#FFC300; margin-top:0;">Interpretation Guide</h4>
                        <b>R-Squared:</b><br>
                        • 0.7 - 1.0: Strong relationship<br>
                        • 0.3 - 0.7: Moderate relationship<br>
                        • < 0.3: Weak relationship<br><br>
                        <b>RMSE:</b><br>
                        Lower is better. Represents the standard deviation of prediction errors.
                    </div>
                    """, unsafe_allow_html=True)

            # ================================================================
            # TAB 2: COEFFICIENTS (Original)
            # ================================================================
            with tab2:
                st.markdown("#### Model Equation")
                intercept = model.params['const']
                equation_str = f"{target_col} = {intercept:.4f}"
                
                for feat in feature_cols:
                    coef = model.params[feat]
                    sign = "+" if coef >= 0 else "-"
                    equation_str += f" {sign} ({abs(coef):.4f} × {feat})"
                
                st.markdown(f'<div class="equation-box">{equation_str}</div>', unsafe_allow_html=True)
                
                st.markdown("#### Model Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": model.params.index,
                    "Coefficient": model.params.values,
                    "Std Error": model.bse.values,
                    "t-value": model.tvalues.values,
                    "P-Value": model.pvalues.values,
                    "[0.025": model.conf_int()[0].values,
                    "0.975]": model.conf_int()[1].values
                })
                
                st.dataframe(coef_df.style.format({
                    "Coefficient": "{:.4f}",
                    "Std Error": "{:.4f}",
                    "t-value": "{:.4f}",
                    "P-Value": "{:.4f}",
                    "[0.025": "{:.4f}",
                    "0.975]": "{:.4f}"
                }), use_container_width=True)
                
                st.info("💡 A low P-Value (< 0.05) indicates the feature is a meaningful predictor.")
                
                st.markdown("---")
                st.markdown("#### 🗣 Business Interpretation")
                
                for feat in feature_cols:
                    coef = model.params[feat]
                    std_dev = data[feat].std()
                    impact_1sd = coef * std_dev
                    
                    direction_1 = "increase" if coef > 0 else "decrease"
                    color_1 = "#10b981" if coef > 0 else "#ef4444"
                    direction_sd = "increase" if impact_1sd > 0 else "decrease"
                    
                    is_sig = model.pvalues[feat] < 0.05
                    opacity = "1" if is_sig else "0.6"
                    sig_badge = '<span style="background:#10b981; padding:2px 6px; border-radius:4px; font-size:0.7em; color:black;">Significant</span>' if is_sig else '<span style="background:#888; padding:2px 6px; border-radius:4px; font-size:0.7em; color:black;">Not Sig.</span>'

                    st.markdown(f"""
                    <div class="interpretation-item" style="opacity: {opacity};">
                        <div class="interp-header">
                            <span class="feature-tag">{feat}</span>
                            {sig_badge}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1px 1fr; gap: 1rem; color: #EAEAEA; font-size: 0.95rem;">
                            <div>
                                <div style="color:#888; font-size:0.8rem; margin-bottom:4px;">PER UNIT</div>
                                For every <b>1.0 unit</b> increase in {feat}, 
                                {target_col} <span style="color: {color_1}; font-weight: 700;">{direction_1}s</span> 
                                by <b>{abs(coef):.4f}</b> units.
                            </div>
                            <div style="background:var(--border-color);"></div>
                            <div>
                                <div style="color:#FFC300; font-size:0.8rem; margin-bottom:4px;">PER STD DEV</div>
                                A typical fluctuation of <b>{std_dev:.2f}</b> units in {feat} 
                                is associated with a <b>{abs(impact_1sd):.2f}</b> unit {direction_sd} in {target_col}.
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ================================================================
            # TAB 3: DIAGNOSTICS (Original)
            # ================================================================
            with tab3:
                residuals = model.resid
                
                st.markdown("#### 1. Normality of Residuals")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_hist = px.histogram(residuals, nbins=30, title="Residual Distribution",
                                          labels={'value': 'Residual Error'})
                    fig_hist.update_traces(marker_color='#FFC300')
                    st.plotly_chart(update_chart_theme(fig_hist), use_container_width=True)
                with c2:
                    jb_stat, jb_pval = stats.jarque_bera(residuals)
                    normality_status = "Normal" if jb_pval > 0.05 else "Non-Normal"
                    style = "success" if jb_pval > 0.05 else "danger"
                    render_metric("Jarque-Bera Test", normality_status, f"P-Val: {jb_pval:.4f}", style)

                st.markdown("---")
                
                st.markdown("#### 2. Homoscedasticity")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_res = px.scatter(x=model.fittedvalues, y=residuals,
                                       labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                       title="Residuals vs Predicted")
                    fig_res.add_hline(y=0, line_dash="dash", line_color="#ef4444")
                    fig_res.update_traces(marker=dict(color="#06b6d4", size=7))
                    st.plotly_chart(update_chart_theme(fig_res), use_container_width=True)
                with c2:
                    try:
                        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, sm.add_constant(data[feature_cols]))
                        homo_status = "Homoscedastic" if bp_pval > 0.05 else "Heteroscedastic"
                        style = "success" if bp_pval > 0.05 else "danger"
                        render_metric("Breusch-Pagan", homo_status, f"P-Val: {bp_pval:.4f}", style)
                    except:
                        st.info("Breusch-Pagan test unavailable.")

                st.markdown("---")
                
                st.markdown("#### 3. Autocorrelation")
                dw_stat = durbin_watson(residuals)
                dw_status = "No Autocorr" if 1.5 < dw_stat < 2.5 else "Autocorrelated"
                style = "success" if 1.5 < dw_stat < 2.5 else "warning"
                render_metric("Durbin-Watson", f"{dw_stat:.2f}", dw_status, style)

                st.markdown("---")
                
                st.markdown("#### 4. Multicollinearity (VIF)")
                if len(feature_cols) > 1:
                    try:
                        X_vif = data[feature_cols]
                        vif_data = []
                        for i, col in enumerate(feature_cols):
                            vif = variance_inflation_factor(X_vif.values, i)
                            vif_data.append({'Feature': col, 'VIF': vif})
                        vif_df = pd.DataFrame(vif_data)
                        st.dataframe(vif_df.style.format({'VIF': '{:.2f}'}), use_container_width=True)
                        st.caption("VIF > 5 indicates problematic multicollinearity. VIF > 10 is severe.")
                    except:
                        st.info("VIF calculation failed.")
                else:
                    st.info("VIF requires 2+ features.")

            # ================================================================
            # TAB 4: PREDICTION ANALYSIS (Original)
            # ================================================================
            with tab4:
                preds = model.predict(sm.add_constant(data[feature_cols]))
                analysis_df = data.copy()
                analysis_df['Predicted'] = preds
                analysis_df['Deviation'] = analysis_df[target_col] - analysis_df['Predicted']
                
                st.markdown("#### Deviation Distribution")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_dev = px.histogram(analysis_df['Deviation'], nbins=30, title="Prediction Errors")
                    fig_dev.add_vline(x=0, line_dash="dash", line_color="#FFC300")
                    st.plotly_chart(update_chart_theme(fig_dev), use_container_width=True)
                with c2:
                    mean_dev = analysis_df['Deviation'].mean()
                    style = "info" if abs(mean_dev) < 0.1 else ("success" if mean_dev > 0 else "danger")
                    render_metric("Mean Deviation", f"{mean_dev:.4f}", "Avg Bias", style)

                st.markdown("---")

                # Residuals Over Time (Full Series) - RESTORED FROM ORIGINAL
                st.markdown("#### 📉 Residuals Over Time (Full Series)")
                
                # Determine X-axis for Time Series
                if date_col_option != "None" and date_col_option in analysis_df.columns:
                    ts_df = analysis_df.sort_values(by=date_col_option)
                    x_axis_col = date_col_option
                    x_axis_title = "Date"
                else:
                    ts_df = analysis_df.sort_index()
                    x_axis_col = ts_df.index
                    x_axis_title = "Index / Row"

                # Residuals over time bar chart with color scale
                fig_resid_time = px.bar(
                    ts_df,
                    x=x_axis_col,
                    y='Deviation',
                    title=f"Residuals ({target_col} Actual - Predicted) over {x_axis_title}",
                    color='Deviation',
                    color_continuous_scale=['#FF4B4B', '#FFC300', '#00E396'], 
                    labels={'Deviation': 'Residual (Act - Pred)'}
                )
                fig_resid_time.add_hline(y=0, line_color="#EAEAEA", line_width=1, opacity=0.5)
                fig_resid_time.update_traces(marker_line_width=0) 
                fig_resid_time.update_layout(coloraxis_showscale=False)
                st.plotly_chart(update_chart_theme(fig_resid_time), use_container_width=True)

                st.markdown("---")
                
                st.markdown("#### 🚩 Extreme Misses")
                c_top, c_bot = st.columns(2)
                
                disp_cols = [target_col, 'Predicted', 'Deviation'] + feature_cols
                if date_col_option != "None" and date_col_option in analysis_df.columns:
                    disp_cols.insert(0, date_col_option)

                with c_top:
                    st.markdown("**Top Under-Predictions** (Actual >> Predicted)")
                    st.dataframe(analysis_df.nlargest(5, 'Deviation')[disp_cols], use_container_width=True)
                    
                with c_bot:
                    st.markdown("**Top Over-Predictions** (Actual << Predicted)")
                    st.dataframe(analysis_df.nsmallest(5, 'Deviation')[disp_cols], use_container_width=True)

            # ================================================================
            # TAB 5: DELTA ANALYSIS (Original)
            # ================================================================
            with tab5:
                st.markdown("### 🌊 Delta (Move) Analysis")
                st.markdown("""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">Logic: Analyzing "Changes" instead of "Levels"</h4>
                    This tab calculates the <b>Period-over-Period Change (Delta)</b> for your features.<br>
                    It effectively asks: <i>"Based on the coefficients derived from the main model, did the Target change as expected given the changes in Features?"</i><br>
                    <br>
                    <b>Predicted Move</b> = Σ (Change in Feature × Coefficient)<br>
                    <b>Move Residual</b> = Actual Move - Predicted Move
                </div>
                """, unsafe_allow_html=True)

                # Explicitly Show Calculation Logic
                st.markdown("#### 🧮 Calculation Logic")
                equation_parts = []
                for feat in feature_cols:
                    coef = model.params[feat]
                    equation_parts.append(f"(Δ {feat} × {coef:.4f})")
                
                full_eq = " + ".join(equation_parts)
                st.info(f"**Predicted Move** = {full_eq}")
                
                if date_col_option != "None" and date_col_option in data.columns:
                    delta_df = data.sort_values(by=date_col_option).copy()
                else:
                    delta_df = data.sort_index().copy()

                delta_df['Actual_Move'] = delta_df[target_col].diff()
                delta_df['Predicted_Move'] = 0.0
                for feat in feature_cols:
                    coef = model.params[feat]
                    delta_df['Predicted_Move'] += delta_df[feat].diff() * coef

                delta_df['Move_Residual'] = delta_df['Actual_Move'] - delta_df['Predicted_Move']
                delta_df = delta_df.dropna()

                c1, c2 = st.columns([2, 1])
                
                with c1:
                    fig_delta = px.scatter(delta_df, x='Actual_Move', y='Predicted_Move',
                                          title="Actual vs Predicted Move")
                    min_m = min(delta_df['Actual_Move'].min(), delta_df['Predicted_Move'].min())
                    max_m = max(delta_df['Actual_Move'].max(), delta_df['Predicted_Move'].max())
                    fig_delta.add_shape(type="line", x0=min_m, y0=min_m, x1=max_m, y1=max_m,
                                       line=dict(color="#FFC300", dash="dash"))
                    fig_delta.update_traces(marker=dict(color="#06b6d4", size=8))
                    st.plotly_chart(update_chart_theme(fig_delta), use_container_width=True)

                with c2:
                    move_corr, _ = stats.pearsonr(delta_df['Actual_Move'], delta_df['Predicted_Move'])
                    render_metric("Move Correlation", f"{move_corr:.4f}", "Directional Accuracy", "primary")
                    
                    avg_miss = delta_df['Move_Residual'].abs().mean()
                    render_metric("Avg Move Error", f"{avg_miss:.4f}", "Mean Abs Deviation", "danger")

                st.markdown("---")

                # Move Residuals Over Time - RESTORED FROM ORIGINAL
                st.markdown("#### 📉 Move Residuals Over Time")
                
                if date_col_option != "None" and date_col_option in delta_df.columns:
                    x_axis_move = date_col_option
                else:
                    x_axis_move = delta_df.index

                fig_move_resid = px.bar(
                    delta_df,
                    x=x_axis_move,
                    y='Move_Residual',
                    title="Residual of Moves (Actual Move - Predicted Move)",
                    color='Move_Residual',
                    color_continuous_scale=['#FF4B4B', '#FFC300', '#00E396']
                )
                fig_move_resid.add_hline(y=0, line_color="#EAEAEA", opacity=0.5)
                fig_move_resid.update_traces(marker_line_width=0)
                fig_move_resid.update_layout(coloraxis_showscale=False)
                st.plotly_chart(update_chart_theme(fig_move_resid), use_container_width=True)

            # ================================================================
            # TAB 6: SIMULATION (Original + Enhanced)
            # ================================================================
            with tab6:
                st.markdown("### 🔮 Simulation & Backtesting")
                
                subtab_sim, subtab_bt, subtab_wf, subtab_mc = st.tabs([
                    "🎛️ What-If Simulator", 
                    "🔙 Simple Backtest",
                    "🚶 Walk-Forward",
                    "🎲 Monte Carlo"
                ])
                
                # --- What-If Simulator ---
                with subtab_sim:
                    st.markdown("#### Sensitivity Analysis")
                    
                    c_sim_inputs, c_sim_res = st.columns([1, 1])
                    user_inputs = {}
                    
                    with c_sim_inputs:
                        st.markdown("**Adjust Feature Values:**")
                        for feat in feature_cols:
                            min_val = float(data[feat].min())
                            max_val = float(data[feat].max())
                            mean_val = float(data[feat].mean())
                            step_val = (max_val - min_val) / 100
                            
                            user_inputs[feat] = st.slider(
                                f"{feat}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val,
                                step=step_val,
                                format="%.2f",
                                key=f"sim_{feat}"
                            )
                    
                    sim_pred = model.params['const']
                    for feat, val in user_inputs.items():
                        sim_pred += model.params[feat] * val
                        
                    with c_sim_res:
                        st.markdown("**Simulation Result:**")
                        avg_target = data[target_col].mean()
                        diff_from_avg = sim_pred - avg_target
                        pct_diff = (diff_from_avg / avg_target) * 100 if avg_target != 0 else 0
                        
                        render_metric(f"Predicted {target_col}", f"{sim_pred:.2f}", f"Vs Average: {pct_diff:+.2f}%", "primary")

                # --- Simple Backtest ---
                with subtab_bt:
                    st.markdown("#### Chronological Train/Test Split")
                    
                    split_pct = st.slider("Training Data %", 50, 90, 80, 5, key="bt_split") / 100.0
                    
                    if date_col_option != "None" and date_col_option in data.columns:
                        bt_data = data.sort_values(by=date_col_option).reset_index(drop=True)
                    else:
                        bt_data = data.reset_index(drop=True)

                    split_idx = int(len(bt_data) * split_pct)
                    train_df = bt_data.iloc[:split_idx]
                    test_df = bt_data.iloc[split_idx:]
                    
                    if len(test_df) < 2:
                        st.error("Not enough test data.")
                    else:
                        try:
                            y_train = train_df[target_col]
                            X_train = sm.add_constant(train_df[feature_cols])
                            bt_model = sm.OLS(y_train, X_train).fit()
                            
                            X_test = sm.add_constant(test_df[feature_cols])
                            test_preds = bt_model.predict(X_test)
                            train_preds = bt_model.predict(X_train)
                            
                            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                            test_rmse = np.sqrt(mean_squared_error(test_df[target_col], test_preds))
                            
                            c1, c2 = st.columns([2, 1])
                            
                            with c1:
                                fig_bt = go.Figure()
                                fig_bt.add_trace(go.Scatter(y=bt_data[target_col], mode='lines', 
                                                           name='Actual', line=dict(color='#444')))
                                fig_bt.add_trace(go.Scatter(x=test_df.index, y=test_preds, mode='lines',
                                                           name='Test Prediction', line=dict(color='#00E396')))
                                fig_bt.add_vline(x=split_idx, line_dash="dash", line_color="#FFC300")
                                st.plotly_chart(update_chart_theme(fig_bt), use_container_width=True)
                                
                            with c2:
                                render_metric("Train RMSE", f"{train_rmse:.4f}", "In-Sample", "info")
                                render_metric("Test RMSE", f"{test_rmse:.4f}", "Out-of-Sample", "primary")
                                
                                pct_degrade = ((test_rmse - train_rmse) / train_rmse) * 100
                                if pct_degrade > 20:
                                    st.warning(f"⚠️ Possible overfitting (+{pct_degrade:.0f}%)")
                                else:
                                    st.success(f"✅ Stable model (+{pct_degrade:.0f}%)")
                        except Exception as e:
                            st.error(f"Backtest failed: {e}")

                # --- Walk-Forward (NEW) ---
                with subtab_wf:
                    st.markdown("#### 🚶 Walk-Forward Optimization")
                    st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="guide-box">
                        Rolling retrain: The model is retrained on expanding windows and tested on future data.
                        This simulates real-world trading conditions where you only have historical data.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    n_folds = st.slider("Number of Folds", 3, 10, 5, key="wf_folds")
                    
                    if st.button("Run Walk-Forward Analysis", key="wf_run"):
                        with st.spinner("Running walk-forward analysis..."):
                            wf_results = walk_forward_backtest(data, target_col, feature_cols, 
                                                               n_splits=n_folds, date_col=date_col_option)
                            
                            if len(wf_results) > 0:
                                st.dataframe(wf_results.style.format({
                                    'R²': '{:.4f}',
                                    'RMSE': '{:.4f}',
                                    'MAE': '{:.4f}',
                                    'Dir Accuracy': '{:.2%}'
                                }), use_container_width=True)
                                
                                # Summary metrics
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    render_metric("Avg R²", f"{wf_results['R²'].mean():.4f}", "Across Folds", "gold")
                                with c2:
                                    render_metric("Avg RMSE", f"{wf_results['RMSE'].mean():.4f}", "Test Error", "info")
                                with c3:
                                    render_metric("Avg Dir Acc", f"{wf_results['Dir Accuracy'].mean():.2%}", "Hit Rate", "success")
                                with c4:
                                    stability = wf_results['R²'].std()
                                    render_metric("R² Stability", f"{stability:.4f}", "Std Dev", "purple")
                                
                                # Visualization
                                fig_wf = go.Figure()
                                fig_wf.add_trace(go.Bar(x=wf_results['Fold'], y=wf_results['R²'], name='R²',
                                                       marker_color='#FFC300'))
                                fig_wf.add_trace(go.Scatter(x=wf_results['Fold'], y=wf_results['RMSE'], 
                                                           name='RMSE', yaxis='y2', line=dict(color='#ef4444')))
                                fig_wf.update_layout(
                                    title="Walk-Forward Performance by Fold",
                                    yaxis=dict(title='R²'),
                                    yaxis2=dict(title='RMSE', overlaying='y', side='right')
                                )
                                st.plotly_chart(update_chart_theme(fig_wf), use_container_width=True)
                            else:
                                st.error("Walk-forward analysis failed.")

                # --- Monte Carlo (NEW) ---
                with subtab_mc:
                    st.markdown("#### 🎲 Monte Carlo Confidence Intervals")
                    st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="guide-box">
                        Bootstrap simulation: Resamples the data many times to estimate 
                        prediction confidence intervals. Shows how uncertain our predictions really are.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    n_sims = st.slider("Number of Simulations", 100, 2000, 500, 100, key="mc_sims")
                    
                    if st.button("Run Monte Carlo Simulation", key="mc_run"):
                        with st.spinner(f"Running {n_sims} bootstrap simulations..."):
                            mc_data = data[[target_col] + feature_cols].copy()
                            mc_results = monte_carlo_simulation(model, mc_data, feature_cols, n_simulations=n_sims)
                            
                            if mc_results:
                                actual_vals = data[target_col].values
                                
                                fig_mc = go.Figure()
                                
                                # Confidence bands
                                x_range = list(range(len(actual_vals)))
                                fig_mc.add_trace(go.Scatter(
                                    x=x_range + x_range[::-1],
                                    y=list(mc_results['ci_upper']) + list(mc_results['ci_lower'][::-1]),
                                    fill='toself',
                                    fillcolor='rgba(255, 195, 0, 0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='95% CI'
                                ))
                                
                                fig_mc.add_trace(go.Scatter(
                                    x=x_range + x_range[::-1],
                                    y=list(mc_results['ci_95']) + list(mc_results['ci_5'][::-1]),
                                    fill='toself',
                                    fillcolor='rgba(255, 195, 0, 0.3)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='90% CI'
                                ))
                                
                                # Mean prediction
                                fig_mc.add_trace(go.Scatter(x=x_range, y=mc_results['mean'], 
                                                           mode='lines', name='Mean Prediction',
                                                           line=dict(color='#FFC300', width=2)))
                                
                                # Actual
                                fig_mc.add_trace(go.Scatter(x=x_range, y=actual_vals, 
                                                           mode='markers', name='Actual',
                                                           marker=dict(color='#06b6d4', size=5)))
                                
                                fig_mc.update_layout(title="Monte Carlo Prediction Confidence Bands")
                                st.plotly_chart(update_chart_theme(fig_mc), use_container_width=True)
                                
                                # Coverage statistics
                                in_95_ci = ((actual_vals >= mc_results['ci_lower']) & 
                                           (actual_vals <= mc_results['ci_upper'])).mean()
                                in_90_ci = ((actual_vals >= mc_results['ci_5']) & 
                                           (actual_vals <= mc_results['ci_95'])).mean()
                                
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    render_metric("95% CI Coverage", f"{in_95_ci:.1%}", "Actual in Band", "gold")
                                with c2:
                                    render_metric("90% CI Coverage", f"{in_90_ci:.1%}", "Actual in Band", "info")
                                with c3:
                                    avg_width = (mc_results['ci_upper'] - mc_results['ci_lower']).mean()
                                    render_metric("Avg CI Width", f"{avg_width:.4f}", "Uncertainty", "purple")
                            else:
                                st.error("Monte Carlo simulation failed.")

            # ================================================================
            # TAB 7: ADVANCED REGRESSION TYPES (NEW)
            # ================================================================
            with tab7:
                st.markdown("### ⚙️ Advanced Regression Types")
                st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                
                if not SKLEARN_AVAILABLE:
                    st.error("scikit-learn is required for advanced regression types.")
                else:
                    reg_type = st.selectbox("Select Regression Type", [
                        "Ridge (L2 Regularization)",
                        "Lasso (L1 Regularization)",
                        "Elastic Net (L1 + L2)",
                        "Huber (Robust to Outliers)",
                        "RANSAC (Outlier Removal)",
                        "Quantile Regression"
                    ])
                    
                    X = data[feature_cols].values
                    y = data[target_col].values
                    
                    col_params, col_results = st.columns([1, 2])
                    
                    with col_params:
                        st.markdown("#### Parameters")
                        
                        if "Ridge" in reg_type:
                            alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, 0.1)
                            adv_model, scaler = run_ridge_regression(X, y, alpha)
                            
                        elif "Lasso" in reg_type:
                            alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, 0.1)
                            adv_model, scaler = run_lasso_regression(X, y, alpha)
                            
                        elif "Elastic" in reg_type:
                            alpha = st.slider("Alpha", 0.01, 10.0, 1.0, 0.1)
                            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.1)
                            adv_model, scaler = run_elastic_net(X, y, alpha, l1_ratio)
                            
                        elif "Huber" in reg_type:
                            epsilon = st.slider("Epsilon", 1.0, 2.0, 1.35, 0.05)
                            adv_model, scaler = run_huber_regression(X, y, epsilon)
                            
                        elif "RANSAC" in reg_type:
                            adv_model, scaler = run_ransac_regression(X, y)
                            
                        elif "Quantile" in reg_type:
                            quantile = st.slider("Quantile", 0.1, 0.9, 0.5, 0.1)
                            adv_model = run_quantile_regression(data, target_col, feature_cols, quantile)
                            scaler = None
                    
                    with col_results:
                        st.markdown("#### Results")
                        
                        # Get predictions
                        if "Quantile" in reg_type:
                            X_const = sm.add_constant(data[feature_cols])
                            adv_preds = adv_model.predict(X_const)
                            
                            # Coefficients
                            coef_df = pd.DataFrame({
                                'Feature': ['const'] + feature_cols,
                                'Coefficient': adv_model.params.values
                            })
                        else:
                            if scaler:
                                X_scaled = scaler.transform(X)
                                adv_preds = adv_model.predict(X_scaled)
                            else:
                                adv_preds = adv_model.predict(X)
                            
                            if hasattr(adv_model, 'coef_'):
                                coef_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Coefficient': adv_model.coef_
                                })
                                if hasattr(adv_model, 'intercept_'):
                                    intercept_row = pd.DataFrame({'Feature': ['Intercept'], 
                                                                 'Coefficient': [adv_model.intercept_]})
                                    coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
                            else:
                                coef_df = pd.DataFrame({'Feature': ['N/A'], 'Coefficient': [0]})
                        
                        # Metrics
                        adv_rmse = np.sqrt(mean_squared_error(y, adv_preds))
                        adv_mae = mean_absolute_error(y, adv_preds)
                        
                        ss_res = np.sum((y - adv_preds) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        adv_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            render_metric("R²", f"{adv_r2:.4f}", "Fit Quality", "gold")
                        with c2:
                            render_metric("RMSE", f"{adv_rmse:.4f}", "Error", "info")
                        with c3:
                            render_metric("MAE", f"{adv_mae:.4f}", "Abs Error", "success")
                        
                        st.markdown("**Coefficients:**")
                        st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}), use_container_width=True)
                        
                        # Comparison with OLS
                        st.markdown("---")
                        st.markdown("#### Comparison with OLS")
                        
                        ols_preds = model.predict(sm.add_constant(data[feature_cols]))
                        ols_rmse = np.sqrt(mean_squared_error(y, ols_preds))
                        
                        improvement = ((ols_rmse - adv_rmse) / ols_rmse) * 100
                        
                        if improvement > 0:
                            st.success(f"✅ {reg_type} improved RMSE by {improvement:.2f}% vs OLS")
                        else:
                            st.info(f"ℹ️ OLS performs {abs(improvement):.2f}% better for this data")
                        
                        # Scatter plot
                        fig_adv = px.scatter(x=y, y=adv_preds, 
                                            labels={'x': 'Actual', 'y': 'Predicted'},
                                            title=f"{reg_type}: Actual vs Predicted")
                        min_v, max_v = min(y.min(), adv_preds.min()), max(y.max(), adv_preds.max())
                        fig_adv.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                                         line=dict(color="#FFC300", dash="dash"))
                        fig_adv.update_traces(marker=dict(color="#06b6d4", size=8))
                        st.plotly_chart(update_chart_theme(fig_adv), use_container_width=True)

            # ================================================================
            # TAB 8: ROLLING ANALYSIS (NEW)
            # ================================================================
            with tab8:
                st.markdown("### 📈 Rolling Window Analysis")
                st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="guide-box">
                    <b>Purpose:</b> See how relationships change over time.<br>
                    Rolling regression retrains the model on a moving window to detect regime changes and coefficient instability.
                </div>
                """, unsafe_allow_html=True)
                
                min_window = max(10, len(feature_cols) + 5)
                max_window = len(data) // 2
                
                if max_window <= min_window:
                    st.warning("Not enough data for rolling analysis. Need at least 2x the minimum window size.")
                else:
                    window_size = st.slider("Rolling Window Size", min_window, max_window, 
                                           min(50, max_window), 5)
                    
                    if st.button("Run Rolling Regression", key="rolling_run"):
                        with st.spinner("Calculating rolling regressions..."):
                            rolling_df = rolling_regression(data, target_col, feature_cols, 
                                                           window_size, date_col_option)
                            
                            if len(rolling_df) > 0:
                                # R² over time
                                st.markdown("#### R² Stability Over Time")
                                
                                x_axis = rolling_df['date'] if 'date' in rolling_df.columns else rolling_df['window_end']
                                
                                fig_r2 = px.line(rolling_df, x=x_axis, y='r_squared',
                                                title="Rolling R² (Model Fit Over Time)")
                                fig_r2.add_hline(y=rolling_df['r_squared'].mean(), line_dash="dash", 
                                               line_color="#FFC300", annotation_text="Mean R²")
                                st.plotly_chart(update_chart_theme(fig_r2), use_container_width=True)
                                
                                # Coefficient evolution
                                st.markdown("#### Coefficient Evolution")
                                
                                coef_cols = [c for c in rolling_df.columns if c.startswith('coef_')]
                                
                                fig_coef = go.Figure()
                                colors = ['#FFC300', '#06b6d4', '#10b981', '#ef4444', '#8b5cf6', '#ec4899']
                                
                                for i, col in enumerate(coef_cols):
                                    feat_name = col.replace('coef_', '')
                                    fig_coef.add_trace(go.Scatter(
                                        x=x_axis, y=rolling_df[col],
                                        mode='lines', name=feat_name,
                                        line=dict(color=colors[i % len(colors)])
                                    ))
                                
                                fig_coef.update_layout(title="Rolling Coefficients Over Time")
                                st.plotly_chart(update_chart_theme(fig_coef), use_container_width=True)
                                
                                # Structural break detection
                                st.markdown("#### Structural Break Detection")
                                
                                breaks = detect_structural_breaks(rolling_df, feature_cols)
                                
                                if breaks:
                                    for brk in breaks:
                                        st.warning(f"⚠️ **{brk['feature']}**: Potential regime change detected. "
                                                  f"Volatility ratio: {brk['volatility_ratio']:.2f}x average")
                                else:
                                    st.success("✅ No significant structural breaks detected.")
                                
                                # Summary stats
                                st.markdown("#### Coefficient Summary Statistics")
                                
                                summary_data = []
                                for col in coef_cols:
                                    feat_name = col.replace('coef_', '')
                                    summary_data.append({
                                        'Feature': feat_name,
                                        'Mean': rolling_df[col].mean(),
                                        'Std Dev': rolling_df[col].std(),
                                        'Min': rolling_df[col].min(),
                                        'Max': rolling_df[col].max(),
                                        'Range': rolling_df[col].max() - rolling_df[col].min()
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df.style.format({
                                    'Mean': '{:.4f}', 'Std Dev': '{:.4f}',
                                    'Min': '{:.4f}', 'Max': '{:.4f}', 'Range': '{:.4f}'
                                }), use_container_width=True)
                            else:
                                st.error("Rolling regression failed.")

            # ================================================================
            # TAB 9: FEATURE ENGINEERING (NEW)
            # ================================================================
            with tab9:
                st.markdown("### 🧪 Feature Engineering Module")
                st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="guide-box">
                    Generate new features from your existing data. Select transformations below and 
                    preview the engineered features before using them in your model.
                </div>
                """, unsafe_allow_html=True)
                
                eng_cols = st.multiselect("Select columns to transform", feature_cols, 
                                         default=feature_cols[:2] if len(feature_cols) >= 2 else feature_cols)
                
                if eng_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Transformations")
                        
                        do_lags = st.checkbox("Generate Lags", value=False)
                        if do_lags:
                            max_lag = st.slider("Max Lag Periods", 1, 10, 3)
                        
                        do_diff = st.checkbox("Generate Differences", value=False)
                        if do_diff:
                            diff_periods = st.slider("Difference Periods", 1, 5, 1)
                        
                        do_rolling = st.checkbox("Generate Rolling Stats", value=False)
                        if do_rolling:
                            roll_window = st.slider("Rolling Window", 3, 20, 5)
                        
                        do_interact = st.checkbox("Generate Interactions", value=False)
                        do_poly = st.checkbox("Generate Polynomials", value=False)
                        if do_poly:
                            poly_degree = st.slider("Polynomial Degree", 2, 4, 2)
                        
                        do_log = st.checkbox("Log Transform", value=False)
                        do_winsor = st.checkbox("Winsorize (Outlier Handling)", value=False)
                    
                    with col2:
                        st.markdown("#### Preview")
                        
                        if st.button("Generate Features", key="fe_generate"):
                            eng_data = data.copy()
                            new_features = []
                            
                            if do_lags:
                                eng_data = generate_lags(eng_data, eng_cols, max_lag)
                                for col in eng_cols:
                                    for l in range(1, max_lag + 1):
                                        new_features.append(f'{col}_lag{l}')
                            
                            if do_diff:
                                eng_data = generate_differences(eng_data, eng_cols, diff_periods)
                                for col in eng_cols:
                                    new_features.append(f'{col}_diff{diff_periods}')
                            
                            if do_rolling:
                                eng_data = generate_rolling_stats(eng_data, eng_cols, roll_window)
                                for col in eng_cols:
                                    new_features.append(f'{col}_roll_mean{roll_window}')
                                    new_features.append(f'{col}_roll_std{roll_window}')
                            
                            if do_interact and len(eng_cols) > 1:
                                eng_data = generate_interactions(eng_data, eng_cols)
                                for i, c1 in enumerate(eng_cols):
                                    for c2 in eng_cols[i+1:]:
                                        new_features.append(f'{c1}_x_{c2}')
                            
                            if do_poly:
                                eng_data = generate_polynomial_features(eng_data, eng_cols, poly_degree)
                                for col in eng_cols:
                                    for d in range(2, poly_degree + 1):
                                        new_features.append(f'{col}_pow{d}')
                            
                            if do_log:
                                eng_data = apply_log_transform(eng_data, eng_cols)
                                for col in eng_cols:
                                    new_features.append(f'{col}_log')
                            
                            if do_winsor:
                                eng_data = winsorize_data(eng_data, eng_cols)
                                for col in eng_cols:
                                    new_features.append(f'{col}_winsor')
                            
                            st.session_state['engineered_data'] = eng_data
                            st.session_state['new_features'] = new_features
                            
                            st.success(f"✅ Generated {len(new_features)} new features!")
                            
                            # Preview
                            preview_cols = [target_col] + eng_cols + new_features[:5]
                            preview_cols = [c for c in preview_cols if c in eng_data.columns]
                            st.dataframe(eng_data[preview_cols].head(10), use_container_width=True)
                            
                            st.markdown(f"**New Features:** {', '.join(new_features)}")
                    
                    # PCA Preview
                    st.markdown("---")
                    st.markdown("#### PCA Dimensionality Reduction Preview")
                    
                    if st.button("Run PCA Analysis", key="pca_run"):
                        pca_results = run_pca_preview(data, feature_cols)
                        
                        # Explained variance plot
                        fig_pca = go.Figure()
                        fig_pca.add_trace(go.Bar(
                            x=[f'PC{i+1}' for i in range(len(pca_results['explained_variance_ratio']))],
                            y=pca_results['explained_variance_ratio'],
                            name='Individual',
                            marker_color='#FFC300'
                        ))
                        fig_pca.add_trace(go.Scatter(
                            x=[f'PC{i+1}' for i in range(len(pca_results['cumulative_variance']))],
                            y=pca_results['cumulative_variance'],
                            name='Cumulative',
                            line=dict(color='#06b6d4')
                        ))
                        fig_pca.update_layout(title="PCA Explained Variance")
                        st.plotly_chart(update_chart_theme(fig_pca), use_container_width=True)
                        
                        # Recommendation
                        for i, cum_var in enumerate(pca_results['cumulative_variance']):
                            if cum_var >= 0.95:
                                st.info(f"💡 {i+1} principal components explain 95%+ of variance "
                                       f"(vs {len(feature_cols)} original features)")
                                break

            # ================================================================
            # TAB 10: MODEL COMPARISON (NEW)
            # ================================================================
            with tab10:
                st.markdown("### 🏆 Model Comparison Framework")
                st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="guide-box">
                    Compare multiple models with different feature sets side-by-side.
                    Use AIC/BIC to select the best model that balances fit and complexity.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Define Models to Compare")
                
                # Model 1: Current model
                st.markdown("**Model 1 (Current):** " + ", ".join(feature_cols))
                
                # Model 2: User-defined
                model2_features = st.multiselect("Model 2 Features", available_features, 
                                                default=available_features[:1] if available_features else [],
                                                key="m2_feat")
                
                # Model 3: User-defined
                model3_features = st.multiselect("Model 3 Features", available_features,
                                                default=available_features[:2] if len(available_features) >= 2 else available_features,
                                                key="m3_feat")
                
                if st.button("Compare Models", key="compare_run"):
                    feature_sets = {
                        "Model 1 (Current)": feature_cols,
                    }
                    
                    if model2_features:
                        feature_sets["Model 2"] = model2_features
                    if model3_features:
                        feature_sets["Model 3"] = model3_features
                    
                    if len(feature_sets) < 2:
                        st.warning("Define at least one additional model for comparison.")
                    else:
                        with st.spinner("Comparing models..."):
                            comparison_df = compare_models(df, target_col, feature_sets, date_col_option)
                            
                            if len(comparison_df) > 0:
                                # Find best model by AIC
                                best_aic_idx = comparison_df['AIC'].idxmin()
                                best_bic_idx = comparison_df['BIC'].idxmin()
                                
                                st.markdown("#### Comparison Results")
                                
                                # Highlight best models
                                def highlight_best(row):
                                    styles = [''] * len(row)
                                    if row.name == best_aic_idx:
                                        styles = ['background-color: rgba(16, 185, 129, 0.2)'] * len(row)
                                    return styles
                                
                                st.dataframe(comparison_df.style.apply(highlight_best, axis=1).format({
                                    'R²': '{:.4f}',
                                    'Adj R²': '{:.4f}',
                                    'AIC': '{:.2f}',
                                    'BIC': '{:.2f}',
                                    'RMSE': '{:.4f}',
                                    'MAE': '{:.4f}',
                                    'F-stat': '{:.2f}',
                                    'F p-val': '{:.4e}'
                                }), use_container_width=True)
                                
                                # Winner announcement
                                best_model = comparison_df.loc[best_aic_idx, 'Model']
                                st.success(f"🏆 **Best Model (by AIC):** {best_model}")
                                
                                if best_aic_idx != best_bic_idx:
                                    st.info(f"ℹ️ BIC prefers: {comparison_df.loc[best_bic_idx, 'Model']}")
                                
                                # Visual comparison
                                st.markdown("#### Visual Comparison")
                                
                                fig_comp = make_subplots(rows=1, cols=3, 
                                                        subplot_titles=('R²', 'AIC (lower=better)', 'RMSE'))
                                
                                fig_comp.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['R²'],
                                                        marker_color='#FFC300'), row=1, col=1)
                                fig_comp.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['AIC'],
                                                        marker_color='#06b6d4'), row=1, col=2)
                                fig_comp.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'],
                                                        marker_color='#10b981'), row=1, col=3)
                                
                                fig_comp.update_layout(showlegend=False, height=400)
                                st.plotly_chart(update_chart_theme(fig_comp), use_container_width=True)
                                
                                # Nested F-test (if applicable)
                                st.markdown("---")
                                st.markdown("#### Nested Model F-Test")
                                st.caption("Tests if adding features significantly improves the model.")
                                
                                # Run F-test between smallest and largest model
                                sorted_models = comparison_df.sort_values('Features')
                                if len(sorted_models) >= 2:
                                    restricted_name = sorted_models.iloc[0]['Model']
                                    full_name = sorted_models.iloc[-1]['Model']
                                    
                                    restricted_features = feature_sets[restricted_name]
                                    full_features = feature_sets[full_name]
                                    
                                    # Only test if one is nested in the other
                                    if set(restricted_features).issubset(set(full_features)):
                                        clean_r = clean_data(df, target_col, restricted_features, date_col_option)
                                        clean_f = clean_data(df, target_col, full_features, date_col_option)
                                        
                                        model_r, _ = run_regression(clean_r, target_col, restricted_features)
                                        model_f, _ = run_regression(clean_f, target_col, full_features)
                                        
                                        if model_r and model_f:
                                            f_stat, f_pval = nested_model_f_test(model_r, model_f, len(clean_f))
                                            
                                            if f_stat and f_pval:
                                                c1, c2 = st.columns(2)
                                                with c1:
                                                    render_metric("F-Statistic", f"{f_stat:.4f}", 
                                                                 f"{restricted_name} vs {full_name}", "info")
                                                with c2:
                                                    sig_style = "success" if f_pval < 0.05 else "warning"
                                                    render_metric("P-Value", f"{f_pval:.4e}", 
                                                                 "Significant" if f_pval < 0.05 else "Not Sig.", sig_style)
                                                
                                                if f_pval < 0.05:
                                                    st.success(f"✅ The additional features in {full_name} significantly improve the model.")
                                                else:
                                                    st.warning(f"⚠️ The simpler {restricted_name} may be sufficient (extra features don't help significantly).")
                            else:
                                st.error("Model comparison failed.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Landing page
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2 style="color: #FFC300;">Welcome to Regression Lab Pro</h2>
            <p style="color: #888; font-size: 1.1rem;">
                Upload your data to begin advanced regression analysis.
            </p>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-top: 2rem;">
                <div class="metric-card">
                    <h4>ADVANCED MODELS</h4>
                    <p style="font-size:0.85rem; color:#888;">Ridge, Lasso, Elastic Net, Huber, RANSAC, Quantile</p>
                </div>
                <div class="metric-card">
                    <h4>ROLLING ANALYSIS</h4>
                    <p style="font-size:0.85rem; color:#888;">Time-varying coefficients & regime detection</p>
                </div>
                <div class="metric-card">
                    <h4>FEATURE ENGINEERING</h4>
                    <p style="font-size:0.85rem; color:#888;">Lags, diffs, interactions, polynomials, PCA</p>
                </div>
                <div class="metric-card">
                    <h4>MODEL COMPARISON</h4>
                    <p style="font-size:0.85rem; color:#888;">AIC/BIC selection & nested F-tests</p>
                </div>
                <div class="metric-card">
                    <h4>ENHANCED BACKTEST</h4>
                    <p style="font-size:0.85rem; color:#888;">Walk-forward & Monte Carlo simulation</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Footer ---
    st.markdown(f"""
    <div class="app-footer">
        <p>© {datetime.now().year} Regression Lab Pro | Arthagati Analytics Suite. All data is for informational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
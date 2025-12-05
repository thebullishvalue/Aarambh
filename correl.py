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
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor, LinearRegression
    from sklearn.preprocessing import StandardScaler
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

# ============================================================================
# INTELLIGENT INTERPRETATION SYSTEM
# ============================================================================

def interpret_r_squared(r2, adj_r2):
    """Generate dynamic interpretation of R-squared values."""
    if r2 >= 0.9:
        strength = "excellent"
        emoji = "🎯"
        action = "Your model explains the vast majority of variance. This is production-ready for forecasting."
    elif r2 >= 0.7:
        strength = "strong"
        emoji = "✅"
        action = "Your model captures most of the relationship. Consider if additional predictors could push it higher."
    elif r2 >= 0.5:
        strength = "moderate"
        emoji = "📊"
        action = "The model explains about half the variance. Look for missing variables or non-linear relationships."
    elif r2 >= 0.3:
        strength = "weak"
        emoji = "⚠️"
        action = "Significant unexplained variance exists. Consider adding more predictors or transforming variables."
    else:
        strength = "very weak"
        emoji = "❌"
        action = "The model explains little variance. The predictors may not be relevant, or the relationship is non-linear."
    
    # Check for overfitting
    overfit_gap = r2 - adj_r2
    overfit_warning = ""
    if overfit_gap > 0.05:
        overfit_warning = f" ⚠️ Gap between R² and Adj R² ({overfit_gap:.3f}) suggests possible overfitting — you may have too many predictors."
    
    return f"{emoji} **{strength.title()} Fit** (R² = {r2:.1%}): {action}{overfit_warning}"

def interpret_p_value(p_val, context="model"):
    """Interpret p-value in plain English."""
    if p_val < 0.001:
        return f"Highly significant (p < 0.001) — extremely strong statistical evidence"
    elif p_val < 0.01:
        return f"Very significant (p < 0.01) — very strong statistical evidence"
    elif p_val < 0.05:
        return f"Significant (p < 0.05) — sufficient statistical evidence"
    elif p_val < 0.10:
        return f"Marginally significant (p < 0.10) — weak evidence, interpret with caution"
    else:
        return f"Not significant (p = {p_val:.3f}) — insufficient evidence of a real relationship"

def interpret_coefficient(coef, feat_name, target_name, std_dev, p_val):
    """Generate plain English interpretation of a coefficient."""
    direction = "increases" if coef > 0 else "decreases"
    abs_coef = abs(coef)
    impact_1sd = abs(coef * std_dev)
    
    sig_status = "✅" if p_val < 0.05 else "⚠️ (not statistically significant)"
    
    interpretation = f"""
**{feat_name}** {sig_status}
- For every 1-unit increase in {feat_name}, {target_name} {direction} by **{abs_coef:.4f}**
- A typical fluctuation (±1 std dev = {std_dev:.2f}) moves {target_name} by **±{impact_1sd:.4f}**
"""
    return interpretation

def interpret_residuals(mean_resid, std_resid, pos_pct, drift):
    """Interpret residual patterns and provide actionable insights."""
    insights = []
    actions = []
    
    # Bias check
    if abs(mean_resid) > std_resid * 0.1:
        if mean_resid > 0:
            insights.append("📈 Model tends to **under-predict** (actual values are higher than predicted)")
            actions.append("Consider if there's a missing upward factor not captured by your predictors")
        else:
            insights.append("📉 Model tends to **over-predict** (actual values are lower than predicted)")
            actions.append("Consider if there's a missing downward factor not captured by your predictors")
    else:
        insights.append("✅ Model predictions are **unbiased** on average")
    
    # Balance check
    if pos_pct > 60:
        insights.append(f"⚠️ {pos_pct:.0f}% of residuals are positive — systematic under-prediction")
    elif pos_pct < 40:
        insights.append(f"⚠️ {100-pos_pct:.0f}% of residuals are negative — systematic over-prediction")
    else:
        insights.append("✅ Residuals are well-balanced between positive and negative")
    
    # Drift check
    if abs(drift) > std_resid * 2:
        insights.append("🚨 **Cumulative drift detected** — model bias is growing over time")
        actions.append("The relationship may be changing. Consider rolling regression or regime detection.")
    
    return insights, actions

def interpret_diagnostics(jb_pval, bp_pval, dw_stat, max_vif):
    """Provide actionable diagnostic interpretation."""
    issues = []
    recommendations = []
    
    # Normality
    if jb_pval < 0.05:
        issues.append("❌ Residuals are **not normally distributed**")
        recommendations.append("Consider robust standard errors or transform your target variable (log, sqrt)")
    else:
        issues.append("✅ Residuals are approximately **normally distributed**")
    
    # Heteroscedasticity
    if bp_pval is not None:
        if bp_pval < 0.05:
            issues.append("❌ **Heteroscedasticity detected** — error variance changes with predictions")
            recommendations.append("Use robust standard errors (HC3) or weighted least squares")
        else:
            issues.append("✅ **Homoscedastic** — constant error variance (good)")
    
    # Autocorrelation
    if dw_stat < 1.5:
        issues.append("❌ **Positive autocorrelation** — errors are correlated over time")
        recommendations.append("Add lagged variables or use time-series methods (ARIMA, etc.)")
    elif dw_stat > 2.5:
        issues.append("❌ **Negative autocorrelation** — unusual pattern in errors")
        recommendations.append("Check for over-differencing or model misspecification")
    else:
        issues.append("✅ **No autocorrelation** — errors are independent (good)")
    
    # Multicollinearity
    if max_vif is not None:
        if max_vif > 10:
            issues.append(f"🚨 **Severe multicollinearity** (VIF = {max_vif:.1f})")
            recommendations.append("Remove or combine highly correlated predictors, or use Ridge regression")
        elif max_vif > 5:
            issues.append(f"⚠️ **Moderate multicollinearity** (VIF = {max_vif:.1f})")
            recommendations.append("Consider removing one of the correlated predictors")
        else:
            issues.append("✅ **No multicollinearity issues** (VIF < 5)")
    
    return issues, recommendations

def interpret_backtest(train_rmse, test_rmse, train_r2, test_r2):
    """Interpret backtesting results."""
    rmse_change = ((test_rmse - train_rmse) / train_rmse) * 100
    r2_change = ((test_r2 - train_r2) / train_r2) * 100 if train_r2 > 0 else 0
    
    if rmse_change > 30:
        status = "🚨 **Severe Overfitting**"
        explanation = f"Test error is {rmse_change:.0f}% higher than training. The model memorized the training data."
        action = "Simplify your model: remove predictors, use regularization (Ridge/Lasso), or get more data."
    elif rmse_change > 15:
        status = "⚠️ **Moderate Overfitting**"
        explanation = f"Test error is {rmse_change:.0f}% higher than training. Some overfitting present."
        action = "Consider removing the least significant predictors or using regularization."
    elif rmse_change < -10:
        status = "🤔 **Unusual: Test Better Than Train**"
        explanation = f"Test error is {abs(rmse_change):.0f}% lower than training. This is unusual."
        action = "Check if your test period has lower volatility or if there's data leakage."
    else:
        status = "✅ **Robust Model**"
        explanation = f"Test and training performance are similar (±{abs(rmse_change):.0f}%)."
        action = "Your model generalizes well. Safe to use for forecasting."
    
    return status, explanation, action

def generate_model_summary(model, r2, adj_r2, rmse, f_pval, n_obs, n_features, target_name):
    """Generate an executive summary of the model."""
    
    # Overall verdict
    if r2 >= 0.7 and f_pval < 0.05:
        verdict = "🟢 **STRONG MODEL** — Ready for use"
        verdict_detail = "High explanatory power with statistical significance."
    elif r2 >= 0.5 and f_pval < 0.05:
        verdict = "🟡 **MODERATE MODEL** — Use with caution"
        verdict_detail = "Decent fit but significant unexplained variance remains."
    elif f_pval >= 0.05:
        verdict = "🔴 **WEAK MODEL** — Not recommended"
        verdict_detail = "Model is not statistically significant. Predictors may not be relevant."
    else:
        verdict = "🟠 **LIMITED MODEL** — Needs improvement"
        verdict_detail = "Low explanatory power. Consider different predictors or transformations."
    
    summary = f"""
### 📋 Model Summary

{verdict}
{verdict_detail}

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² | {r2:.1%} | Model explains {r2:.1%} of {target_name} variance |
| Adj R² | {adj_r2:.1%} | Adjusted for {n_features} predictors |
| RMSE | {rmse:.4f} | Average prediction error magnitude |
| Significance | p = {f_pval:.4f} | {interpret_p_value(f_pval)} |
| Sample Size | {n_obs} | {"✅ Adequate" if n_obs > 30 else "⚠️ Small sample"} |

"""
    return summary, verdict

def generate_action_items(model, diagnostics_ok, overfit_risk, significant_features, insignificant_features):
    """Generate prioritized action items based on analysis."""
    actions = []
    
    # Priority 1: Model validity
    if not diagnostics_ok:
        actions.append("🔴 **Fix diagnostic issues first** — Your statistical tests may be unreliable")
    
    # Priority 2: Overfitting
    if overfit_risk:
        actions.append("🟠 **Address overfitting** — Remove weak predictors or use regularization")
    
    # Priority 3: Feature refinement
    if insignificant_features:
        actions.append(f"🟡 **Consider removing**: {', '.join(insignificant_features)} (not statistically significant)")
    
    # Priority 4: Next steps
    if len(significant_features) > 0:
        actions.append(f"🟢 **Key drivers identified**: {', '.join(significant_features)}")
    
    return actions

def render_insight_box(title, content, box_type="info"):
    """Render a styled insight box."""
    colors = {
        "success": ("#10b981", "rgba(16, 185, 129, 0.1)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "danger": ("#ef4444", "rgba(239, 68, 68, 0.1)"),
        "info": ("#06b6d4", "rgba(6, 182, 212, 0.1)"),
        "primary": ("#FFC300", "rgba(255, 195, 0, 0.1)")
    }
    border_color, bg_color = colors.get(box_type, colors["info"])
    
    st.markdown(f"""
    <div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: {border_color}; margin: 0 0 0.5rem 0;">{title}</h4>
        <p style="color: #EAEAEA; margin: 0; font-size: 0.95rem; line-height: 1.6;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def render_bottom_line(title, verdict, explanation, next_steps, verdict_type="info"):
    """Render a comprehensive bottom-line summary box."""
    colors = {
        "success": ("#10b981", "rgba(16, 185, 129, 0.08)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.08)"),
        "danger": ("#ef4444", "rgba(239, 68, 68, 0.08)"),
        "info": ("#06b6d4", "rgba(6, 182, 212, 0.08)"),
        "primary": ("#FFC300", "rgba(255, 195, 0, 0.08)")
    }
    border_color, bg_color = colors.get(verdict_type, colors["info"])
    
    steps_html = "".join([f"<li style='margin-bottom: 0.3rem;'>{step}</li>" for step in next_steps]) if next_steps else ""
    
    st.markdown(f"""
    <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem; margin-right: 0.75rem;">📋</span>
            <h3 style="color: {border_color}; margin: 0; font-size: 1.1rem;">{title}</h3>
        </div>
        <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="color: #EAEAEA; margin: 0; font-size: 1.1rem; font-weight: 600;">{verdict}</p>
        </div>
        <p style="color: #AAAAAA; margin: 0 0 1rem 0; font-size: 0.95rem; line-height: 1.6;">{explanation}</p>
        {"<div style='margin-top: 1rem;'><p style='color: #888; font-size: 0.85rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;'>Recommended Actions:</p><ul style='color: #EAEAEA; margin: 0; padding-left: 1.25rem; font-size: 0.9rem;'>" + steps_html + "</ul></div>" if next_steps else ""}
    </div>
    """, unsafe_allow_html=True)

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

def run_regression_basket(data, target, features):
    """
    Runs multiple regression models and returns results for comparison.
    Returns a dictionary with all model results and recommendations.
    """
    results = {}
    y = data[target].values
    X = data[features].values
    n_samples, n_features = X.shape
    
    # 1. OLS (Baseline)
    try:
        X_ols = sm.add_constant(X)
        ols_model = sm.OLS(y, X_ols).fit()
        ols_preds = ols_model.predict(X_ols)
        ols_rmse = np.sqrt(mean_squared_error(y, ols_preds))
        ols_r2 = ols_model.rsquared
        ols_adj_r2 = ols_model.rsquared_adj
        ols_aic = ols_model.aic
        ols_bic = ols_model.bic
        
        results['OLS'] = {
            'model': ols_model,
            'predictions': ols_preds,
            'r2': ols_r2,
            'adj_r2': ols_adj_r2,
            'rmse': ols_rmse,
            'mae': mean_absolute_error(y, ols_preds),
            'aic': ols_aic,
            'bic': ols_bic,
            'description': 'Ordinary Least Squares - Standard linear regression',
            'best_for': 'Clean data with no multicollinearity or outliers',
            'coefficients': dict(zip(['const'] + features, ols_model.params))
        }
    except Exception as e:
        results['OLS'] = {'error': str(e)}
    
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Ridge Regression
        try:
            from sklearn.linear_model import RidgeCV
            ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
            ridge_cv.fit(X_scaled, y)
            ridge_preds = ridge_cv.predict(X_scaled)
            ridge_rmse = np.sqrt(mean_squared_error(y, ridge_preds))
            ridge_r2 = ridge_cv.score(X_scaled, y)
            
            # Calculate AIC/BIC approximation for Ridge
            n = len(y)
            k = n_features + 1
            rss = np.sum((y - ridge_preds) ** 2)
            ridge_aic = n * np.log(rss / n) + 2 * k
            ridge_bic = n * np.log(rss / n) + k * np.log(n)
            
            results['Ridge'] = {
                'model': ridge_cv,
                'scaler': scaler,
                'predictions': ridge_preds,
                'r2': ridge_r2,
                'adj_r2': 1 - (1 - ridge_r2) * (n - 1) / (n - k - 1),
                'rmse': ridge_rmse,
                'mae': mean_absolute_error(y, ridge_preds),
                'aic': ridge_aic,
                'bic': ridge_bic,
                'alpha': ridge_cv.alpha_,
                'description': 'Ridge (L2) - Handles multicollinearity',
                'best_for': 'High correlation between predictors',
                'coefficients': dict(zip(features, ridge_cv.coef_))
            }
        except Exception as e:
            results['Ridge'] = {'error': str(e)}
        
        # 3. Lasso Regression
        try:
            from sklearn.linear_model import LassoCV
            lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5, max_iter=10000)
            lasso_cv.fit(X_scaled, y)
            lasso_preds = lasso_cv.predict(X_scaled)
            lasso_rmse = np.sqrt(mean_squared_error(y, lasso_preds))
            lasso_r2 = lasso_cv.score(X_scaled, y)
            
            n = len(y)
            k = np.sum(lasso_cv.coef_ != 0) + 1  # Only count non-zero coefficients
            rss = np.sum((y - lasso_preds) ** 2)
            lasso_aic = n * np.log(rss / n) + 2 * k
            lasso_bic = n * np.log(rss / n) + k * np.log(n)
            
            # Count selected features
            selected_features = [f for f, c in zip(features, lasso_cv.coef_) if abs(c) > 1e-6]
            
            results['Lasso'] = {
                'model': lasso_cv,
                'scaler': scaler,
                'predictions': lasso_preds,
                'r2': lasso_r2,
                'adj_r2': 1 - (1 - lasso_r2) * (n - 1) / (n - k - 1) if k > 0 else lasso_r2,
                'rmse': lasso_rmse,
                'mae': mean_absolute_error(y, lasso_preds),
                'aic': lasso_aic,
                'bic': lasso_bic,
                'alpha': lasso_cv.alpha_,
                'selected_features': selected_features,
                'n_selected': len(selected_features),
                'description': 'Lasso (L1) - Feature selection via sparsity',
                'best_for': 'Too many predictors, need automatic selection',
                'coefficients': dict(zip(features, lasso_cv.coef_))
            }
        except Exception as e:
            results['Lasso'] = {'error': str(e)}
        
        # 4. Elastic Net
        try:
            from sklearn.linear_model import ElasticNetCV
            enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95], alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=10000)
            enet_cv.fit(X_scaled, y)
            enet_preds = enet_cv.predict(X_scaled)
            enet_rmse = np.sqrt(mean_squared_error(y, enet_preds))
            enet_r2 = enet_cv.score(X_scaled, y)
            
            n = len(y)
            k = np.sum(enet_cv.coef_ != 0) + 1
            rss = np.sum((y - enet_preds) ** 2)
            enet_aic = n * np.log(rss / n) + 2 * k
            enet_bic = n * np.log(rss / n) + k * np.log(n)
            
            results['Elastic Net'] = {
                'model': enet_cv,
                'scaler': scaler,
                'predictions': enet_preds,
                'r2': enet_r2,
                'adj_r2': 1 - (1 - enet_r2) * (n - 1) / (n - k - 1) if k > 0 else enet_r2,
                'rmse': enet_rmse,
                'mae': mean_absolute_error(y, enet_preds),
                'aic': enet_aic,
                'bic': enet_bic,
                'alpha': enet_cv.alpha_,
                'l1_ratio': enet_cv.l1_ratio_,
                'description': 'Elastic Net - Combines Ridge + Lasso benefits',
                'best_for': 'Correlated predictors + need feature selection',
                'coefficients': dict(zip(features, enet_cv.coef_))
            }
        except Exception as e:
            results['Elastic Net'] = {'error': str(e)}
        
        # 5. Huber Robust Regression
        try:
            huber = HuberRegressor(epsilon=1.35, max_iter=1000)
            huber.fit(X_scaled, y)
            huber_preds = huber.predict(X_scaled)
            huber_rmse = np.sqrt(mean_squared_error(y, huber_preds))
            huber_r2 = huber.score(X_scaled, y)
            
            n = len(y)
            k = n_features + 1
            rss = np.sum((y - huber_preds) ** 2)
            huber_aic = n * np.log(rss / n) + 2 * k
            huber_bic = n * np.log(rss / n) + k * np.log(n)
            
            results['Huber'] = {
                'model': huber,
                'scaler': scaler,
                'predictions': huber_preds,
                'r2': huber_r2,
                'adj_r2': 1 - (1 - huber_r2) * (n - 1) / (n - k - 1),
                'rmse': huber_rmse,
                'mae': mean_absolute_error(y, huber_preds),
                'aic': huber_aic,
                'bic': huber_bic,
                'description': 'Huber - Robust to outliers',
                'best_for': 'Data with outliers or heavy tails',
                'coefficients': dict(zip(features, huber.coef_))
            }
        except Exception as e:
            results['Huber'] = {'error': str(e)}
        
        # 6. RANSAC (outlier-resistant)
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            ransac = RANSACRegressor(estimator=LinearRegression(), random_state=42, max_trials=100)
            ransac.fit(X, y)  # RANSAC doesn't need scaling
            ransac_preds = ransac.predict(X)
            ransac_rmse = np.sqrt(mean_squared_error(y, ransac_preds))
            ransac_r2 = ransac.score(X, y)
            
            n = len(y)
            k = n_features + 1
            rss = np.sum((y - ransac_preds) ** 2)
            ransac_aic = n * np.log(rss / n) + 2 * k
            ransac_bic = n * np.log(rss / n) + k * np.log(n)
            
            # Count inliers
            n_inliers = np.sum(ransac.inlier_mask_)
            
            results['RANSAC'] = {
                'model': ransac,
                'predictions': ransac_preds,
                'r2': ransac_r2,
                'adj_r2': 1 - (1 - ransac_r2) * (n - 1) / (n - k - 1),
                'rmse': ransac_rmse,
                'mae': mean_absolute_error(y, ransac_preds),
                'aic': ransac_aic,
                'bic': ransac_bic,
                'n_inliers': n_inliers,
                'n_outliers': n - n_inliers,
                'outlier_pct': (n - n_inliers) / n * 100,
                'description': 'RANSAC - Automatic outlier removal',
                'best_for': 'Data with significant outliers to ignore',
                'coefficients': dict(zip(features, ransac.estimator_.coef_))
            }
        except Exception as e:
            results['RANSAC'] = {'error': str(e)}
    
    # 7. Quantile Regression (median)
    try:
        X_quant = sm.add_constant(X)
        quant_model = QuantReg(y, X_quant).fit(q=0.5)
        quant_preds = quant_model.predict(X_quant)
        quant_rmse = np.sqrt(mean_squared_error(y, quant_preds))
        
        # Pseudo R2 for quantile regression
        quant_r2 = 1 - quant_model.prsquared if hasattr(quant_model, 'prsquared') else 0
        
        n = len(y)
        k = n_features + 1
        rss = np.sum((y - quant_preds) ** 2)
        quant_aic = n * np.log(rss / n) + 2 * k
        quant_bic = n * np.log(rss / n) + k * np.log(n)
        
        results['Quantile (Median)'] = {
            'model': quant_model,
            'predictions': quant_preds,
            'r2': quant_r2,
            'adj_r2': quant_r2,
            'rmse': quant_rmse,
            'mae': mean_absolute_error(y, quant_preds),
            'aic': quant_aic,
            'bic': quant_bic,
            'quantile': 0.5,
            'description': 'Quantile (Median) - Predicts median instead of mean',
            'best_for': 'Skewed distributions or when median is preferred',
            'coefficients': dict(zip(['const'] + features, quant_model.params))
        }
    except Exception as e:
        results['Quantile (Median)'] = {'error': str(e)}
    
    return results

def select_best_model(basket_results, data, target, features):
    """
    Analyzes data characteristics and recommends the best model.
    Returns recommendation with reasoning.
    """
    y = data[target].values
    X = data[features].values
    n_samples, n_features = X.shape
    
    # Data diagnostics
    diagnostics = {}
    
    # 1. Check for multicollinearity
    if n_features > 1:
        try:
            corr_matrix = np.corrcoef(X.T)
            max_corr = np.max(np.abs(corr_matrix[np.triu_indices(n_features, k=1)]))
            diagnostics['multicollinearity'] = max_corr > 0.7
            diagnostics['max_correlation'] = max_corr
        except:
            diagnostics['multicollinearity'] = False
            diagnostics['max_correlation'] = 0
    else:
        diagnostics['multicollinearity'] = False
        diagnostics['max_correlation'] = 0
    
    # 2. Check for outliers (using IQR method)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outlier_mask = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
    diagnostics['has_outliers'] = np.sum(outlier_mask) > n_samples * 0.05
    diagnostics['outlier_pct'] = np.sum(outlier_mask) / n_samples * 100
    
    # 3. Check for skewness
    diagnostics['skewness'] = stats.skew(y)
    diagnostics['is_skewed'] = abs(diagnostics['skewness']) > 1
    
    # 4. Check sample size vs features
    diagnostics['low_sample'] = n_samples < n_features * 10
    diagnostics['samples_per_feature'] = n_samples / n_features
    
    # 5. Check for heteroscedasticity (simple test)
    if 'OLS' in basket_results and 'error' not in basket_results['OLS']:
        residuals = y - basket_results['OLS']['predictions']
        try:
            _, bp_pval, _, _ = het_breuschpagan(residuals, sm.add_constant(X))
            diagnostics['heteroscedastic'] = bp_pval < 0.05
        except:
            diagnostics['heteroscedastic'] = False
    else:
        diagnostics['heteroscedastic'] = False
    
    # Scoring system for each model
    scores = {}
    reasoning = {}
    
    for model_name, result in basket_results.items():
        if 'error' in result:
            continue
        
        score = 0
        reasons = []
        
        # Base score from R²
        r2 = result.get('r2', 0)
        score += r2 * 30  # Max 30 points for R²
        reasons.append(f"R² = {r2:.3f}")
        
        # Bonus/penalty based on data characteristics
        
        # Multicollinearity handling
        if diagnostics['multicollinearity']:
            if model_name in ['Ridge', 'Elastic Net']:
                score += 15
                reasons.append("+15: Handles multicollinearity well")
            elif model_name == 'OLS':
                score -= 10
                reasons.append("-10: Sensitive to multicollinearity")
        
        # Outlier handling
        if diagnostics['has_outliers']:
            if model_name in ['Huber', 'RANSAC', 'Quantile (Median)']:
                score += 15
                reasons.append("+15: Robust to outliers")
            elif model_name == 'OLS':
                score -= 10
                reasons.append("-10: Sensitive to outliers")
        
        # Feature selection for high dimensionality
        if diagnostics['low_sample']:
            if model_name in ['Lasso', 'Elastic Net']:
                score += 10
                reasons.append("+10: Good for high-dimensional data")
            elif model_name == 'OLS':
                score -= 5
                reasons.append("-5: May overfit with many features")
        
        # Skewness handling
        if diagnostics['is_skewed']:
            if model_name == 'Quantile (Median)':
                score += 10
                reasons.append("+10: Better for skewed distributions")
        
        # BIC bonus (parsimony)
        bic = result.get('bic', float('inf'))
        # Will normalize later
        
        # Interpretability bonus
        if model_name == 'OLS':
            score += 5
            reasons.append("+5: Most interpretable")
        
        scores[model_name] = score
        reasoning[model_name] = reasons
    
    # Normalize and select best
    if scores:
        best_model = max(scores, key=scores.get)
    else:
        best_model = 'OLS'
    
    return {
        'best_model': best_model,
        'scores': scores,
        'reasoning': reasoning,
        'diagnostics': diagnostics
    }

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
# MAIN APPLICATION
# ============================================================================

def load_google_sheet(sheet_url):
    """Load data from a public Google Sheet."""
    try:
        # Extract the sheet ID and gid from the URL
        import re
        
        # Extract sheet ID
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            return None, "Invalid Google Sheets URL"
        
        sheet_id = sheet_id_match.group(1)
        
        # Extract gid if present
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        
        # Construct CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)

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
    
    # Data Source Selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["📤 Upload File", "📊 Google Sheets"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "📤 Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
    
    else:  # Google Sheets
        st.sidebar.markdown("---")
        
        # Default Google Sheet URL
        default_sheet_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155#gid=1738251155"
        
        sheet_url = st.sidebar.text_input(
            "Google Sheet URL",
            value=default_sheet_url,
            help="Paste a Google Sheets URL. The sheet must be publicly accessible (Anyone with link can view)."
        )
        
        if st.sidebar.button("🔄 Load Sheet", type="primary"):
            with st.spinner("Loading Google Sheet..."):
                df, error = load_google_sheet(sheet_url)
                if error:
                    st.error(f"Failed to load sheet: {error}")
                    st.info("Make sure the sheet is set to 'Anyone with the link can view'")
                    return
                else:
                    st.session_state['gsheet_data'] = df
                    st.success("✅ Google Sheet loaded successfully!")
        
        # Use cached data if available
        if 'gsheet_data' in st.session_state:
            df = st.session_state['gsheet_data']
    
    if df is not None:
        try:
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Data must have at least 2 numeric columns.")
                return

            st.sidebar.markdown("---")
            
            # Default selections for Google Sheets
            default_target = "NIFTY50_PE"
            default_predictors = [
                "AD_RATIO", "COUNT", "IN10Y", "IN02Y", "IN30Y", "INIRYY", 
                "REPO", "CRR", "US02Y", "US10Y", "US30Y", "US_FED", 
                "NIFTY50_DY", "NIFTY50_PB"
            ]
            
            # Determine target default index
            if default_target in numeric_cols:
                target_default_idx = numeric_cols.index(default_target)
            else:
                target_default_idx = 0
            
            target_col = st.sidebar.selectbox("🎯 Target (Y)", numeric_cols, index=target_default_idx)
            available_features = [c for c in numeric_cols if c != target_col]
            
            # Determine predictor defaults (only those that exist in data)
            if data_source == "📊 Google Sheets":
                valid_defaults = [p for p in default_predictors if p in available_features]
                feature_default = valid_defaults if valid_defaults else available_features[:1]
            else:
                feature_default = available_features[:1]
            
            feature_cols = st.sidebar.multiselect("📊 Predictors (X)", available_features, default=feature_default)
            
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

            # Run regression basket FIRST to get all models
            basket_results = run_regression_basket(data, target_col, feature_cols)
            model_selection = select_best_model(basket_results, data, target_col, feature_cols)
            
            # Get available models (those without errors)
            available_models = [name for name, result in basket_results.items() if 'error' not in result]
            best_model_name = model_selection['best_model']
            
            # Model Selection in Sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎯 Model Selection")
            
            # Show recommendation
            st.sidebar.markdown(f"""
            <div style="background: rgba(255,195,0,0.1); border: 1px solid #FFC300; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
                <div style="color: #FFC300; font-weight: 600; font-size: 0.85rem;">🏆 RECOMMENDED</div>
                <div style="color: #EAEAEA; font-size: 1rem;">{best_model_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Model selector
            selected_model_name = st.sidebar.selectbox(
                "Active Model",
                available_models,
                index=available_models.index(best_model_name) if best_model_name in available_models else 0,
                help="All analysis tabs will use this model"
            )
            
            # Show quick stats of selected model
            if selected_model_name in basket_results:
                sel_result = basket_results[selected_model_name]
                st.sidebar.markdown(f"""
                <div style="font-size: 0.85rem; color: #888; margin-top: 0.5rem;">
                    R² = {sel_result['r2']:.4f} | RMSE = {sel_result['rmse']:.4f}
                </div>
                """, unsafe_allow_html=True)
                
                if selected_model_name != best_model_name:
                    st.sidebar.warning(f"ℹ️ {best_model_name} is recommended for your data")
            
            # Get the selected model's predictions and details
            selected_result = basket_results[selected_model_name]
            predictions = selected_result['predictions']
            
            # For OLS, we have the full statsmodels object
            # For sklearn models, we need to create a wrapper or use different logic
            is_ols = selected_model_name == 'OLS'
            
            if is_ols:
                model = selected_result['model']  # statsmodels OLS object
            else:
                # For sklearn models, we still need OLS for some diagnostic tests
                # Run OLS as backup for diagnostics
                ols_model, _ = run_regression(data, target_col, feature_cols)
                model = ols_model  # Use OLS for diagnostic tests
                # But use sklearn predictions for residual analysis

            # --- TABS (Streamlined workflow) ---
            # Primary Analysis → Model Understanding → Validation → Model Selection
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📉 Residuals",           # PRIMARY: Time series residual analysis
                "📊 Performance",          # Model fit metrics
                "📐 Equation",             # Coefficients & interpretation
                "🔍 Predictions",          # Actual vs predicted deep dive
                "🌊 Moves",                # Delta/change analysis
                "🛠️ Diagnostics",          # Statistical tests
                "🎯 Model Selection"       # Regression basket with recommendations
            ])
            
            # Show active model banner at the top of analysis
            if selected_model_name != 'OLS':
                st.info(f"📊 **Active Model: {selected_model_name}** — {selected_result['description']}")

            # ================================================================
            # TAB 2: PERFORMANCE
            # ================================================================
            with tab2:
                # Use selected model metrics
                r2 = selected_result['r2']
                adj_r2 = selected_result['adj_r2']
                rmse = selected_result['rmse']
                
                # For p-value, use OLS if available
                if is_ols:
                    f_pval = model.f_pvalue
                    n_obs = int(model.nobs)
                else:
                    f_pval = basket_results['OLS']['model'].f_pvalue if 'OLS' in basket_results and 'error' not in basket_results['OLS'] else 0.05
                    n_obs = len(data)
                
                # Executive Summary at the top
                st.markdown(f"""
                <div class="guide-box" style="border-color: {'#10b981' if r2 >= 0.7 else '#f59e0b' if r2 >= 0.5 else '#ef4444'};">
                    <h4 style="margin-top:0;">📊 {selected_model_name} Performance Summary</h4>
                    <p style="color: #EAEAEA; font-size: 1.1rem;">
                        {'🟢 <b>Strong Model</b>' if r2 >= 0.7 else '🟡 <b>Moderate Model</b>' if r2 >= 0.5 else '🔴 <b>Weak Model</b>'} — 
                        Explains <b>{r2:.1%}</b> of variance in {target_col} with RMSE of <b>{rmse:.4f}</b>
                    </p>
                    <p style="color: #888; font-size: 0.9rem; margin-top: 0.5rem;">
                        {selected_result['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Key Metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1: render_metric("R-Squared", f"{r2:.4f}", f"Explains {r2:.1%} of variance", "gold")
                with c2: render_metric("Adj. R-Squared", f"{adj_r2:.4f}", f"Penalized for {len(feature_cols)} features", "gold")
                with c3: render_metric("RMSE", f"{rmse:.4f}", f"±{rmse:.2f} avg error", "success")
                with c4: 
                    mae = selected_result['mae']
                    render_metric("MAE", f"{mae:.4f}", "Mean Absolute Error", "info")

                st.markdown("---")

                c_left, c_right = st.columns([2, 1])
                
                with c_left:
                    # Use selected model's predictions
                    preds = predictions
                    
                    fig_pred = px.scatter(x=data[target_col], y=preds, 
                                          labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                          title=f"Actual vs Predicted: {target_col} ({selected_model_name})")
                    
                    min_val = min(data[target_col].min(), preds.min())
                    max_val = max(data[target_col].max(), preds.max())
                    fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                     line=dict(color="#FFC300", dash="dash"))
                    
                    fig_pred.update_traces(marker=dict(color="#06b6d4", size=8, opacity=0.7))
                    st.plotly_chart(update_chart_theme(fig_pred), width="stretch")
                
                with c_right:
                    # Dynamic interpretation instead of static guide
                    st.markdown("#### 🎯 What This Means")
                    
                    r2_interpretation = interpret_r_squared(r2, adj_r2)
                    st.markdown(r2_interpretation)
                    
                    st.markdown("")
                    st.markdown(f"**Model Significance:** {interpret_p_value(f_pval)}")
                    
                    st.markdown("")
                    st.markdown(f"**Prediction Accuracy:** On average, predictions are off by ±{rmse:.4f} units of {target_col}")
                    
                    # Identify significant vs insignificant features using OLS p-values
                    ols_result = basket_results.get('OLS', {})
                    if 'error' not in ols_result and 'model' in ols_result:
                        ols_pvals = ols_result['model'].pvalues
                        sig_feats = [f for f in feature_cols if f in ols_pvals.index and ols_pvals[f] < 0.05]
                        insig_feats = [f for f in feature_cols if f in ols_pvals.index and ols_pvals[f] >= 0.05]
                    else:
                        sig_feats = feature_cols  # Assume all significant if no OLS
                        insig_feats = []
                    
                    if sig_feats:
                        st.markdown(f"**✅ Significant predictors:** {', '.join(sig_feats)}")
                    if insig_feats:
                        st.markdown(f"**⚠️ Weak predictors:** {', '.join(insig_feats)}")

            # ================================================================
            # TAB 3: EQUATION & COEFFICIENTS
            # ================================================================
            with tab3:
                st.markdown(f"#### Model Equation ({selected_model_name})")
                
                # Get coefficients from selected model
                coefficients = selected_result.get('coefficients', {})
                intercept = coefficients.get('const', 0)
                
                equation_str = f"{target_col} = {intercept:.4f}"
                
                for feat in feature_cols:
                    coef = coefficients.get(feat, 0)
                    sign = "+" if coef >= 0 else "-"
                    equation_str += f" {sign} ({abs(coef):.4f} × {feat})"
                
                st.markdown(f'<div class="equation-box">{equation_str}</div>', unsafe_allow_html=True)
                
                st.markdown(f"#### Model Coefficients ({selected_model_name})")
                
                # For OLS, we have full stats. For sklearn, we only have coefficients
                if is_ols:
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
                    }), width="stretch")
                    
                    st.info("💡 A low P-Value (< 0.05) indicates the feature is a meaningful predictor.")
                else:
                    # For sklearn models, show simpler coefficient table
                    coef_df = pd.DataFrame({
                        "Feature": list(coefficients.keys()),
                        "Coefficient": list(coefficients.values())
                    })
                    
                    st.dataframe(coef_df.style.format({
                        "Coefficient": "{:.6f}"
                    }), width="stretch")
                    
                    st.info(f"💡 {selected_model_name} coefficients shown. Statistical significance requires OLS model.")
                    
                    # Show Lasso-specific info
                    if selected_model_name == 'Lasso':
                        zero_coefs = [f for f, c in coefficients.items() if abs(c) < 1e-6]
                        if zero_coefs:
                            st.warning(f"🔍 Lasso eliminated {len(zero_coefs)} features: {', '.join(zero_coefs)}")
                
                st.markdown("---")
                st.markdown("#### 🗣 Business Interpretation")
                
                # Use OLS p-values for significance if available, otherwise all are "significant"
                ols_pvalues = basket_results['OLS']['model'].pvalues if 'OLS' in basket_results and 'error' not in basket_results['OLS'] else None
                
                for feat in feature_cols:
                    coef = coefficients.get(feat, 0)
                    std_dev = data[feat].std()
                    impact_1sd = coef * std_dev
                    
                    direction_1 = "increase" if coef > 0 else "decrease"
                    color_1 = "#10b981" if coef > 0 else "#ef4444"
                    direction_sd = "increase" if impact_1sd > 0 else "decrease"
                    
                    # Check significance using OLS p-values
                    is_sig = ols_pvalues[feat] < 0.05 if ols_pvalues is not None and feat in ols_pvalues else True
                    opacity = "1" if is_sig else "0.6"
                    
                    if abs(coef) < 1e-6:  # Zero coefficient (Lasso eliminated)
                        sig_badge = '<span style="background:#666; padding:2px 6px; border-radius:4px; font-size:0.7em; color:white;">Eliminated</span>'
                        opacity = "0.4"
                    elif is_sig:
                        sig_badge = '<span style="background:#10b981; padding:2px 6px; border-radius:4px; font-size:0.7em; color:black;">Significant</span>'
                    else:
                        sig_badge = '<span style="background:#888; padding:2px 6px; border-radius:4px; font-size:0.7em; color:black;">Not Sig.</span>'

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
                
                # BOTTOM LINE SUMMARY
                st.markdown("---")
                
                # Determine key drivers using OLS p-values if available
                if ols_pvalues is not None:
                    sig_features = [(f, coefficients.get(f, 0), ols_pvalues.get(f, 1)) for f in feature_cols if f in ols_pvalues and ols_pvalues[f] < 0.05]
                    insig_features = [f for f in feature_cols if f not in ols_pvalues or ols_pvalues[f] >= 0.05]
                else:
                    sig_features = [(f, coefficients.get(f, 0), 0) for f in feature_cols if abs(coefficients.get(f, 0)) > 1e-6]
                    insig_features = [f for f in feature_cols if abs(coefficients.get(f, 0)) < 1e-6]
                
                sig_features_sorted = sorted(sig_features, key=lambda x: abs(x[1] * data[x[0]].std()), reverse=True)
                
                if sig_features_sorted:
                    top_driver = sig_features_sorted[0][0]
                    top_impact = abs(sig_features_sorted[0][1] * data[top_driver].std())
                    driver_direction = "increases" if sig_features_sorted[0][1] > 0 else "decreases"
                    
                    verdict = f"The strongest driver of {target_col} is {top_driver}"
                    explanation = f"When {top_driver} moves by one standard deviation, {target_col} {driver_direction} by approximately {top_impact:.2f} units. "
                    
                    if len(sig_features_sorted) > 1:
                        other_drivers = [f[0] for f in sig_features_sorted[1:3]]
                        explanation += f"Other significant factors include: {', '.join(other_drivers)}."
                    
                    next_steps = []
                    if insig_features:
                        next_steps.append(f"Consider removing weak predictors ({', '.join(insig_features)}) to simplify the model")
                    next_steps.append(f"Monitor {top_driver} closely as it has the largest impact on {target_col}")
                    if len(sig_features_sorted) >= 2:
                        next_steps.append("Use the equation above to forecast future values based on your predictor assumptions")
                    
                    render_bottom_line("Bottom Line: Key Drivers", verdict, explanation, next_steps, "success")
                else:
                    verdict = "No statistically significant predictors found"
                    explanation = f"None of the selected features show a reliable relationship with {target_col}. The coefficients cannot be trusted for prediction or interpretation."
                    next_steps = [
                        "Try different predictor variables that may have stronger relationships",
                        "Check if the relationship is non-linear (try polynomial features)",
                        "Ensure you have enough data points for reliable estimation"
                    ]
                    render_bottom_line("Bottom Line: No Clear Drivers", verdict, explanation, next_steps, "danger")

            # ================================================================
            # TAB 6: DIAGNOSTICS
            # ================================================================
            with tab6:
                st.markdown("### 🛠️ Model Diagnostics")
                st.markdown("Statistical tests to verify your regression assumptions are met.")
                
                residuals = model.resid
                
                # Run all tests first to generate summary
                jb_stat, jb_pval = stats.jarque_bera(residuals)
                dw_stat = durbin_watson(residuals)
                
                try:
                    bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, sm.add_constant(data[feature_cols]))
                except:
                    bp_pval = None
                
                max_vif = None
                if len(feature_cols) > 1:
                    try:
                        X_vif = data[feature_cols]
                        vifs = [variance_inflation_factor(X_vif.values, i) for i in range(len(feature_cols))]
                        max_vif = max(vifs)
                    except:
                        pass
                
                # Generate interpretation
                issues, recommendations = interpret_diagnostics(jb_pval, bp_pval, dw_stat, max_vif)
                
                # Summary at top
                passed_tests = sum([
                    jb_pval > 0.05,
                    bp_pval > 0.05 if bp_pval else True,
                    1.5 < dw_stat < 2.5,
                    max_vif < 5 if max_vif else True
                ])
                total_tests = 4
                
                if passed_tests == total_tests:
                    st.success(f"✅ **All {total_tests} diagnostic tests passed!** Your model assumptions are valid.")
                elif passed_tests >= total_tests - 1:
                    st.warning(f"⚠️ **{passed_tests}/{total_tests} tests passed.** Minor issues detected — see recommendations below.")
                else:
                    st.error(f"❌ **{passed_tests}/{total_tests} tests passed.** Significant issues detected — results may be unreliable.")
                
                st.markdown("---")
                
                st.markdown("#### 1. Normality of Residuals")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_hist = px.histogram(residuals, nbins=30, title="Residual Distribution",
                                          labels={'value': 'Residual Error'})
                    fig_hist.update_traces(marker_color='#FFC300')
                    st.plotly_chart(update_chart_theme(fig_hist), width="stretch")
                with c2:
                    normality_status = "Normal" if jb_pval > 0.05 else "Non-Normal"
                    style = "success" if jb_pval > 0.05 else "danger"
                    render_metric("Jarque-Bera Test", normality_status, f"P-Val: {jb_pval:.4f}", style)
                    if jb_pval < 0.05:
                        st.caption("⚠️ Non-normal residuals can affect confidence intervals and p-values.")

                st.markdown("---")
                
                st.markdown("#### 2. Homoscedasticity (Constant Variance)")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_res = px.scatter(x=model.fittedvalues, y=residuals,
                                       labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                       title="Residuals vs Predicted")
                    fig_res.add_hline(y=0, line_dash="dash", line_color="#ef4444")
                    fig_res.update_traces(marker=dict(color="#06b6d4", size=7))
                    st.plotly_chart(update_chart_theme(fig_res), width="stretch")
                with c2:
                    if bp_pval is not None:
                        homo_status = "Homoscedastic" if bp_pval > 0.05 else "Heteroscedastic"
                        style = "success" if bp_pval > 0.05 else "danger"
                        render_metric("Breusch-Pagan", homo_status, f"P-Val: {bp_pval:.4f}", style)
                        if bp_pval < 0.05:
                            st.caption("⚠️ Error variance changes with predictions. Consider robust standard errors.")
                    else:
                        st.info("Breusch-Pagan test unavailable.")

                st.markdown("---")
                
                st.markdown("#### 3. Autocorrelation")
                c1, c2 = st.columns([2, 1])
                with c1:
                    dw_status = "No Autocorr" if 1.5 < dw_stat < 2.5 else "Autocorrelated"
                    style = "success" if 1.5 < dw_stat < 2.5 else "warning"
                    render_metric("Durbin-Watson", f"{dw_stat:.2f}", dw_status, style)
                with c2:
                    if dw_stat < 1.5:
                        st.caption("⚠️ Positive autocorrelation: errors follow a pattern. Add lagged variables.")
                    elif dw_stat > 2.5:
                        st.caption("⚠️ Negative autocorrelation: unusual error pattern.")
                    else:
                        st.caption("✅ Errors are independent — good for reliable inference.")

                st.markdown("---")
                
                st.markdown("#### 4. Multicollinearity (VIF)")
                if len(feature_cols) > 1:
                    try:
                        X_vif = data[feature_cols]
                        vif_data = []
                        for i, col in enumerate(feature_cols):
                            vif = variance_inflation_factor(X_vif.values, i)
                            status = "🟢" if vif < 5 else ("🟡" if vif < 10 else "🔴")
                            vif_data.append({'Feature': col, 'VIF': vif, 'Status': status})
                        vif_df = pd.DataFrame(vif_data)
                        st.dataframe(vif_df.style.format({'VIF': '{:.2f}'}), width="stretch")
                        
                        if max_vif and max_vif > 5:
                            st.caption(f"⚠️ High VIF detected ({max_vif:.1f}). Correlated predictors inflate standard errors.")
                        else:
                            st.caption("✅ No multicollinearity issues — predictors are independent.")
                    except:
                        st.info("VIF calculation failed.")
                else:
                    st.info("VIF requires 2+ features.")
                
                # Summary and Recommendations
                st.markdown("---")
                st.markdown("### 🎯 Summary & Recommendations")
                
                for issue in issues:
                    st.markdown(issue)
                
                if recommendations:
                    st.markdown("")
                    st.markdown("**💡 Recommended Actions:**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                
                # BOTTOM LINE SUMMARY
                st.markdown("---")
                
                if passed_tests == total_tests:
                    verdict = "All diagnostic tests passed — model is statistically valid"
                    explanation = "Your regression assumptions (normality, constant variance, no autocorrelation, no multicollinearity) are all satisfied. You can trust the p-values and confidence intervals."
                    next_steps = [
                        "Proceed with confidence — your statistical inference is reliable",
                        "The coefficients and their significance levels are trustworthy",
                        "Use the model for prediction and interpretation"
                    ]
                    verdict_type = "success"
                elif passed_tests >= 3:
                    failed = [k for k, v in [("normality", jb_pval > 0.05), 
                                             ("homoscedasticity", bp_pval > 0.05 if bp_pval else True),
                                             ("no autocorrelation", 1.5 < dw_stat < 2.5),
                                             ("no multicollinearity", max_vif < 5 if max_vif else True)] if not v]
                    verdict = f"Minor diagnostic issues ({', '.join(failed)})"
                    explanation = "Most assumptions are met, but some violations exist. Results are likely still useful but interpret with some caution."
                    next_steps = recommendations + ["Consider using robust standard errors for more reliable inference"]
                    verdict_type = "warning"
                else:
                    verdict = "Significant diagnostic problems detected"
                    explanation = "Multiple regression assumptions are violated. P-values and confidence intervals may be unreliable. Address these issues before drawing conclusions."
                    next_steps = recommendations + ["Do not rely on this model for critical decisions until issues are resolved"]
                    verdict_type = "danger"
                
                render_bottom_line("Bottom Line: Model Validity", verdict, explanation, next_steps, verdict_type)

            # ================================================================
            # TAB 4: PREDICTIONS
            # ================================================================
            with tab4:
                # Use selected model predictions
                preds = predictions
                analysis_df = data.copy()
                analysis_df['Predicted'] = preds
                analysis_df['Deviation'] = analysis_df[target_col] - analysis_df['Predicted']
                
                st.markdown(f"#### Deviation Distribution ({selected_model_name})")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_dev = px.histogram(analysis_df['Deviation'], nbins=30, title=f"Prediction Errors ({selected_model_name})")
                    fig_dev.add_vline(x=0, line_dash="dash", line_color="#FFC300")
                    st.plotly_chart(update_chart_theme(fig_dev), width="stretch")
                with c2:
                    mean_dev = analysis_df['Deviation'].mean()
                    style = "info" if abs(mean_dev) < 0.1 else ("success" if mean_dev > 0 else "danger")
                    render_metric("Mean Deviation", f"{mean_dev:.4f}", "Avg Bias", style)

                st.markdown("---")

                # Residuals Over Time (Full Series)
                st.markdown(f"#### 📉 Residuals Over Time ({selected_model_name})")
                
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
                st.plotly_chart(update_chart_theme(fig_resid_time), width="stretch")

                st.markdown("---")
                
                st.markdown("#### 🚩 Extreme Misses")
                c_top, c_bot = st.columns(2)
                
                disp_cols = [target_col, 'Predicted', 'Deviation'] + feature_cols
                if date_col_option != "None" and date_col_option in analysis_df.columns:
                    disp_cols.insert(0, date_col_option)

                with c_top:
                    st.markdown("**Top Under-Predictions** (Actual >> Predicted)")
                    st.dataframe(analysis_df.nlargest(5, 'Deviation')[disp_cols], width="stretch")
                    
                with c_bot:
                    st.markdown("**Top Over-Predictions** (Actual << Predicted)")
                    st.dataframe(analysis_df.nsmallest(5, 'Deviation')[disp_cols], width="stretch")
                
                # BOTTOM LINE SUMMARY
                st.markdown("---")
                
                mean_dev = analysis_df['Deviation'].mean()
                std_dev = analysis_df['Deviation'].std()
                pos_pct = (analysis_df['Deviation'] > 0).mean() * 100
                max_under = analysis_df['Deviation'].max()
                max_over = abs(analysis_df['Deviation'].min())
                
                # Determine bias pattern
                if abs(mean_dev) < std_dev * 0.1:
                    bias_status = "unbiased"
                    verdict_type = "success"
                elif mean_dev > 0:
                    bias_status = "tends to under-predict"
                    verdict_type = "warning"
                else:
                    bias_status = "tends to over-predict"
                    verdict_type = "warning"
                
                verdict = f"Model is {bias_status} with average error of ±{std_dev:.2f}"
                explanation = f"On average, predictions miss by {std_dev:.2f} units. The largest under-prediction was {max_under:.2f} and largest over-prediction was {max_over:.2f}. About {pos_pct:.0f}% of predictions are too low."
                
                next_steps = []
                if abs(mean_dev) > std_dev * 0.1:
                    if mean_dev > 0:
                        next_steps.append(f"Systematic under-prediction: consider if there's an unmeasured factor pushing {target_col} higher")
                    else:
                        next_steps.append(f"Systematic over-prediction: consider if there's an unmeasured factor pushing {target_col} lower")
                
                # Look at extreme cases
                top_miss = analysis_df.nlargest(1, 'Deviation').iloc[0]
                next_steps.append(f"Investigate the largest miss ({top_miss['Deviation']:.2f}) to understand what the model is missing")
                next_steps.append(f"Use these prediction errors to set realistic confidence bounds (±{std_dev*2:.2f} for 95% of cases)")
                
                render_bottom_line("Bottom Line: Prediction Quality", verdict, explanation, next_steps, verdict_type)

            # ================================================================
            # TAB 5: MOVES (Delta Analysis)
            # ================================================================
            with tab5:
                st.markdown(f"### 🌊 Move Analysis ({selected_model_name})")
                st.markdown(f"""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">Analyzing Changes Instead of Levels</h4>
                    This tab calculates the <b>Period-over-Period Change (Delta)</b> for your features.<br>
                    It asks: <i>"Based on the {selected_model_name} coefficients, did the Target change as expected given the changes in Features?"</i><br>
                    <br>
                    <b>Predicted Move</b> = Σ (Δ Feature × Coefficient)<br>
                    <b>Move Residual</b> = Actual Move − Predicted Move
                </div>
                """, unsafe_allow_html=True)

                # Use selected model coefficients
                coefficients = selected_result.get('coefficients', {})

                # Explicitly Show Calculation Logic
                st.markdown("#### 🧮 Calculation Logic")
                equation_parts = []
                for feat in feature_cols:
                    coef = coefficients.get(feat, 0)
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
                    coef = coefficients.get(feat, 0)
                    delta_df['Predicted_Move'] += delta_df[feat].diff() * coef

                delta_df['Move_Residual'] = delta_df['Actual_Move'] - delta_df['Predicted_Move']
                delta_df = delta_df.dropna()

                c1, c2 = st.columns([2, 1])
                
                with c1:
                    fig_delta = px.scatter(delta_df, x='Actual_Move', y='Predicted_Move',
                                          title=f"Actual vs Predicted Move ({selected_model_name})")
                    min_m = min(delta_df['Actual_Move'].min(), delta_df['Predicted_Move'].min())
                    max_m = max(delta_df['Actual_Move'].max(), delta_df['Predicted_Move'].max())
                    fig_delta.add_shape(type="line", x0=min_m, y0=min_m, x1=max_m, y1=max_m,
                                       line=dict(color="#FFC300", dash="dash"))
                    fig_delta.update_traces(marker=dict(color="#06b6d4", size=8))
                    st.plotly_chart(update_chart_theme(fig_delta), width="stretch")

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
                st.plotly_chart(update_chart_theme(fig_move_resid), width="stretch")
                
                # BOTTOM LINE SUMMARY
                st.markdown("---")
                
                move_corr_val, _ = stats.pearsonr(delta_df['Actual_Move'], delta_df['Predicted_Move'])
                dir_accuracy = ((delta_df['Actual_Move'] * delta_df['Predicted_Move']) > 0).mean() * 100
                avg_move_error = delta_df['Move_Residual'].abs().mean()
                
                if move_corr_val >= 0.7:
                    verdict = f"Model captures {target_col} movements well (correlation: {move_corr_val:.2f})"
                    verdict_type = "success"
                elif move_corr_val >= 0.4:
                    verdict = f"Model partially captures {target_col} movements (correlation: {move_corr_val:.2f})"
                    verdict_type = "warning"
                else:
                    verdict = f"Model struggles to predict {target_col} movements (correlation: {move_corr_val:.2f})"
                    verdict_type = "danger"
                
                explanation = f"The model correctly predicts the direction of change {dir_accuracy:.0f}% of the time. On average, predicted moves differ from actual moves by {avg_move_error:.2f} units."
                
                next_steps = []
                if dir_accuracy >= 60:
                    next_steps.append(f"Direction accuracy ({dir_accuracy:.0f}%) is good for trend-following strategies")
                else:
                    next_steps.append(f"Low direction accuracy ({dir_accuracy:.0f}%) means the model may miss trend changes")
                
                if move_corr_val < 0.5:
                    next_steps.append("Consider adding momentum or lagged features to better capture dynamics")
                
                next_steps.append(f"Use move predictions for period-over-period forecasting rather than absolute levels")
                
                render_bottom_line("Bottom Line: Move Prediction", verdict, explanation, next_steps, verdict_type)

            # ================================================================
            # TAB 1: RESIDUALS (Primary Analysis View)
            # ================================================================
            with tab1:
                st.markdown(f"### 📉 Residual Analysis ({selected_model_name})")
                st.markdown(f"""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">How Prediction Errors Evolve Over Time</h4>
                    Identify periods where the model systematically over or under-predicts,
                    detect regime changes, and spot anomalies in your data.
                    <br><span style="color:#888; font-size:0.85rem;">Using: {selected_result['description']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare data using selected model predictions
                preds_ts = predictions  # Use selected model's predictions
                ts_analysis_df = data.copy()
                ts_analysis_df['Predicted'] = preds_ts
                ts_analysis_df['Deviation'] = ts_analysis_df[target_col] - ts_analysis_df['Predicted']
                
                # Delta data - use selected model's coefficients
                if date_col_option != "None" and date_col_option in data.columns:
                    ts_delta_df = data.sort_values(by=date_col_option).copy()
                else:
                    ts_delta_df = data.sort_index().copy()
                
                ts_delta_df['Actual_Move'] = ts_delta_df[target_col].diff()
                
                # For move predictions, use selected model's coefficients
                ts_delta_df['Predicted_Move'] = 0.0
                coefficients = selected_result.get('coefficients', {})
                for feat in feature_cols:
                    if feat in coefficients:
                        coef = coefficients[feat]
                        ts_delta_df['Predicted_Move'] += ts_delta_df[feat].diff() * coef
                
                ts_delta_df['Move_Residual'] = ts_delta_df['Actual_Move'] - ts_delta_df['Predicted_Move']
                ts_delta_df = ts_delta_df.dropna()
                
                # ---- CHART 1: Residuals Over Time (Full Series) ----
                st.markdown("#### 📊 Residuals Over Time (Full Series)")
                st.caption(f"Shows how {selected_model_name} prediction errors (Actual - Predicted) vary across time. Persistent positive/negative values indicate systematic bias.")
                
                if date_col_option != "None" and date_col_option in ts_analysis_df.columns:
                    ts_df_sorted = ts_analysis_df.sort_values(by=date_col_option)
                    x_axis_resid = date_col_option
                    x_title = "Date"
                else:
                    ts_df_sorted = ts_analysis_df.sort_index()
                    x_axis_resid = ts_df_sorted.index
                    x_title = "Index / Row"

                fig_resid_ts = px.bar(
                    ts_df_sorted,
                    x=x_axis_resid,
                    y='Deviation',
                    title=f"Level Residuals ({target_col} Actual - Predicted) over {x_title}",
                    color='Deviation',
                    color_continuous_scale=['#FF4B4B', '#FFC300', '#00E396'],
                    labels={'Deviation': 'Residual (Act - Pred)'}
                )
                fig_resid_ts.add_hline(y=0, line_color="#EAEAEA", line_width=1, opacity=0.5)
                fig_resid_ts.update_traces(marker_line_width=0)
                fig_resid_ts.update_layout(coloraxis_showscale=False, height=400)
                st.plotly_chart(update_chart_theme(fig_resid_ts), width="stretch")
                
                # Stats for Level Residuals
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    render_metric("Mean Residual", f"{ts_df_sorted['Deviation'].mean():.4f}", "Avg Bias", "info")
                with c2:
                    render_metric("Std Dev", f"{ts_df_sorted['Deviation'].std():.4f}", "Volatility", "gold")
                with c3:
                    pos_pct = (ts_df_sorted['Deviation'] > 0).mean() * 100
                    render_metric("% Positive", f"{pos_pct:.1f}%", "Under-predictions", "success")
                with c4:
                    neg_pct = (ts_df_sorted['Deviation'] < 0).mean() * 100
                    render_metric("% Negative", f"{neg_pct:.1f}%", "Over-predictions", "danger")
                
                st.markdown("---")
                
                # ---- CHART 2: Move Residuals Over Time ----
                st.markdown("#### 🌊 Move Residuals Over Time")
                st.caption("Shows how well the model predicts period-over-period changes. Large bars indicate the model missed significant moves.")
                
                if date_col_option != "None" and date_col_option in ts_delta_df.columns:
                    x_axis_move_ts = date_col_option
                    x_title_move = "Date"
                else:
                    x_axis_move_ts = ts_delta_df.index
                    x_title_move = "Index"

                fig_move_ts = px.bar(
                    ts_delta_df,
                    x=x_axis_move_ts,
                    y='Move_Residual',
                    title=f"Move Residuals (Actual Move - Predicted Move) over {x_title_move}",
                    color='Move_Residual',
                    color_continuous_scale=['#FF4B4B', '#FFC300', '#00E396']
                )
                fig_move_ts.add_hline(y=0, line_color="#EAEAEA", opacity=0.5)
                fig_move_ts.update_traces(marker_line_width=0)
                fig_move_ts.update_layout(coloraxis_showscale=False, height=400)
                st.plotly_chart(update_chart_theme(fig_move_ts), width="stretch")
                
                # Stats for Move Residuals
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    render_metric("Mean Move Error", f"{ts_delta_df['Move_Residual'].mean():.4f}", "Avg Bias", "info")
                with c2:
                    render_metric("Std Dev", f"{ts_delta_df['Move_Residual'].std():.4f}", "Volatility", "gold")
                with c3:
                    move_corr, _ = stats.pearsonr(ts_delta_df['Actual_Move'], ts_delta_df['Predicted_Move'])
                    render_metric("Move Correlation", f"{move_corr:.4f}", "Directional", "purple")
                with c4:
                    hit_rate = ((ts_delta_df['Actual_Move'] * ts_delta_df['Predicted_Move']) > 0).mean() * 100
                    render_metric("Direction Hit Rate", f"{hit_rate:.1f}%", "Sign Match", "success")
                
                st.markdown("---")
                
                # ---- Cumulative Residual Chart ----
                st.markdown("#### 📈 Cumulative Residual (Drift Detection)")
                st.caption("Cumulative sum of residuals. A trending line indicates persistent model bias over time.")
                
                ts_df_sorted['Cumulative_Residual'] = ts_df_sorted['Deviation'].cumsum()
                
                fig_cumul = px.line(
                    ts_df_sorted,
                    x=x_axis_resid,
                    y='Cumulative_Residual',
                    title="Cumulative Residual Over Time"
                )
                fig_cumul.add_hline(y=0, line_dash="dash", line_color="#FFC300", opacity=0.7)
                fig_cumul.update_traces(line=dict(color='#06b6d4', width=2))
                fig_cumul.update_layout(height=350)
                st.plotly_chart(update_chart_theme(fig_cumul), width="stretch")
                
                st.markdown("---")
                
                # ---- NEW: Correlation Analysis Section ----
                st.markdown("#### 🔗 Residual Correlation Analysis")
                st.caption("Deep dive into the relationship between target values and prediction errors. Ideally, residuals should be uncorrelated with the target.")
                
                # Calculate correlations
                level_resid_corr, level_resid_pval = stats.pearsonr(ts_df_sorted[target_col], ts_df_sorted['Deviation'])
                move_resid_corr, move_resid_pval = stats.pearsonr(ts_delta_df['Actual_Move'], ts_delta_df['Move_Residual'])
                
                # Also calculate correlation between target level and move residual (cross-analysis)
                # Need to align the data
                ts_df_sorted_aligned = ts_df_sorted.copy()
                ts_df_sorted_aligned['Move_Residual'] = ts_delta_df['Move_Residual'].reindex(ts_df_sorted_aligned.index)
                ts_df_sorted_aligned = ts_df_sorted_aligned.dropna()
                
                if len(ts_df_sorted_aligned) > 5:
                    cross_corr, cross_pval = stats.pearsonr(ts_df_sorted_aligned[target_col], ts_df_sorted_aligned['Move_Residual'])
                else:
                    cross_corr, cross_pval = 0, 1
                
                # Two column layout for scatter plots
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("**Target vs Level Residuals**")
                    fig_corr1 = px.scatter(
                        ts_df_sorted,
                        x=target_col,
                        y='Deviation',
                        title=f"Correlation: {level_resid_corr:.3f}",
                        trendline="ols",
                        labels={'Deviation': 'Level Residual (Act - Pred)'}
                    )
                    fig_corr1.update_traces(marker=dict(color='#06b6d4', size=6, opacity=0.6))
                    fig_corr1.add_hline(y=0, line_dash="dash", line_color="#FFC300", opacity=0.5)
                    fig_corr1.update_layout(height=350)
                    st.plotly_chart(update_chart_theme(fig_corr1), width="stretch")
                    
                    # Interpretation
                    if abs(level_resid_corr) < 0.1:
                        st.success(f"✅ **No correlation** (r = {level_resid_corr:.3f}): Residuals are independent of {target_col} level. This is ideal.")
                    elif abs(level_resid_corr) < 0.3:
                        st.info(f"ℹ️ **Weak correlation** (r = {level_resid_corr:.3f}): Slight pattern exists but may not be significant.")
                    elif level_resid_corr > 0.3:
                        st.warning(f"⚠️ **Positive correlation** (r = {level_resid_corr:.3f}): Model under-predicts more at higher {target_col} values. Consider non-linear terms.")
                    else:
                        st.warning(f"⚠️ **Negative correlation** (r = {level_resid_corr:.3f}): Model over-predicts more at higher {target_col} values. Consider non-linear terms.")
                
                with col_right:
                    st.markdown("**Target Move vs Move Residuals**")
                    fig_corr2 = px.scatter(
                        ts_delta_df,
                        x='Actual_Move',
                        y='Move_Residual',
                        title=f"Correlation: {move_resid_corr:.3f}",
                        trendline="ols",
                        labels={'Actual_Move': f'{target_col} Actual Move', 'Move_Residual': 'Move Residual'}
                    )
                    fig_corr2.update_traces(marker=dict(color='#10b981', size=6, opacity=0.6))
                    fig_corr2.add_hline(y=0, line_dash="dash", line_color="#FFC300", opacity=0.5)
                    fig_corr2.add_vline(x=0, line_dash="dash", line_color="#FFC300", opacity=0.5)
                    fig_corr2.update_layout(height=350)
                    st.plotly_chart(update_chart_theme(fig_corr2), width="stretch")
                    
                    # Interpretation
                    if abs(move_resid_corr) < 0.1:
                        st.success(f"✅ **No correlation** (r = {move_resid_corr:.3f}): Move residuals are independent of move size. Model captures dynamics well.")
                    elif abs(move_resid_corr) < 0.3:
                        st.info(f"ℹ️ **Weak correlation** (r = {move_resid_corr:.3f}): Minor pattern in move prediction errors.")
                    elif move_resid_corr > 0.3:
                        st.warning(f"⚠️ **Positive correlation** (r = {move_resid_corr:.3f}): Model underestimates large positive moves. Consider momentum features.")
                    else:
                        st.warning(f"⚠️ **Negative correlation** (r = {move_resid_corr:.3f}): Model overestimates moves. May be too reactive to changes.")
                
                # Summary metrics row
                st.markdown("")
                c1, c2, c3 = st.columns(3)
                with c1:
                    style1 = "success" if abs(level_resid_corr) < 0.2 else "warning"
                    render_metric("Level Correlation", f"{level_resid_corr:.3f}", f"p = {level_resid_pval:.4f}", style1)
                with c2:
                    style2 = "success" if abs(move_resid_corr) < 0.2 else "warning"
                    render_metric("Move Correlation", f"{move_resid_corr:.3f}", f"p = {move_resid_pval:.4f}", style2)
                with c3:
                    style3 = "success" if abs(cross_corr) < 0.2 else "warning"
                    render_metric("Cross Correlation", f"{cross_corr:.3f}", f"Level vs Move Resid", style3)
                
                # Correlation Analysis Bottom Line
                corr_issues = []
                corr_actions = []
                
                if abs(level_resid_corr) >= 0.3:
                    corr_issues.append(f"Level residuals correlated with {target_col}")
                    if level_resid_corr > 0:
                        corr_actions.append(f"Add squared or log terms of {target_col} as a feature")
                    else:
                        corr_actions.append("Consider a different functional form (log, sqrt transformation)")
                
                if abs(move_resid_corr) >= 0.3:
                    corr_issues.append("Move residuals correlated with move magnitude")
                    if move_resid_corr > 0:
                        corr_actions.append("Add momentum or lagged features to capture large moves")
                    else:
                        corr_actions.append("Model may be over-reactive; consider smoothing predictors")
                
                if not corr_issues:
                    st.markdown("""
                    <div class="guide-box" style="border-color: #10b981;">
                        <h4 style="color:#10b981; margin-top:0;">✅ Correlation Check Passed</h4>
                        Residuals show no significant correlation with target values. This indicates:
                        <ul>
                            <li>The model captures the linear relationship well</li>
                            <li>No obvious non-linear patterns are being missed</li>
                            <li>Predictions are equally reliable across all target values</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    issues_text = ", ".join(corr_issues)
                    st.markdown(f"""
                    <div class="guide-box" style="border-color: #f59e0b;">
                        <h4 style="color:#f59e0b; margin-top:0;">⚠️ Correlation Issues Detected</h4>
                        <b>Problems:</b> {issues_text}<br><br>
                        <b>What this means:</b> The model's accuracy varies depending on the value of {target_col}. 
                        Predictions may be less reliable in certain ranges.<br><br>
                        <b>Recommended fixes:</b>
                        <ul>
                            {"".join([f"<li>{a}</li>" for a in corr_actions])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ---- BUSINESS INTERPRETATION OF CORRELATIONS ----
                st.markdown("#### 🗣️ Business Interpretation")
                
                # Level Correlation Business Interpretation
                level_corr_abs = abs(level_resid_corr)
                level_direction = "positive" if level_resid_corr > 0 else "negative"
                
                # Determine practical impact
                target_range = ts_df_sorted[target_col].max() - ts_df_sorted[target_col].min()
                resid_range = ts_df_sorted['Deviation'].max() - ts_df_sorted['Deviation'].min()
                
                # Calculate how much residual changes per unit of target (regression slope)
                if ts_df_sorted[target_col].std() > 0:
                    level_slope = level_resid_corr * (ts_df_sorted['Deviation'].std() / ts_df_sorted[target_col].std())
                else:
                    level_slope = 0
                
                # Level correlation interpretation
                if level_corr_abs < 0.1:
                    level_sig_badge = '<span style="background:#10b981; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">IDEAL</span>'
                    level_opacity = "1"
                elif level_corr_abs < 0.3:
                    level_sig_badge = '<span style="background:#f59e0b; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">MINOR</span>'
                    level_opacity = "0.9"
                else:
                    level_sig_badge = '<span style="background:#ef4444; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">CONCERN</span>'
                    level_opacity = "1"
                
                st.markdown(f"""
                <div class="interpretation-item" style="opacity: {level_opacity};">
                    <div class="interp-header">
                        <span class="feature-tag">Target vs Level Residuals</span>
                        {level_sig_badge}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1px 1fr; gap: 1rem; color: #EAEAEA; font-size: 0.95rem;">
                        <div>
                            <div style="color:#888; font-size:0.8rem; margin-bottom:4px;">CORRELATION STRENGTH</div>
                            The correlation coefficient is <b>r = {level_resid_corr:.3f}</b> ({level_direction}).<br>
                            This means <b>{level_corr_abs*100:.1f}%</b> of the variation in residuals 
                            can be explained by the {target_col} level itself.
                        </div>
                        <div style="background:var(--border-color);"></div>
                        <div>
                            <div style="color:#FFC300; font-size:0.8rem; margin-bottom:4px;">PRACTICAL IMPACT</div>
                            For every <b>1 unit</b> increase in {target_col}, 
                            the prediction error {"increases" if level_resid_corr > 0 else "decreases"} by approximately <b>{abs(level_slope):.4f}</b> units.
                            {"<br><span style='color:#10b981;'>✓ Negligible impact on predictions.</span>" if level_corr_abs < 0.1 else "<br><span style='color:#f59e0b;'>⚠ Model accuracy varies by target level.</span>" if level_corr_abs < 0.3 else "<br><span style='color:#ef4444;'>⚠ Significant bias at extreme values.</span>"}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Move Correlation Business Interpretation
                move_corr_abs = abs(move_resid_corr)
                move_direction = "positive" if move_resid_corr > 0 else "negative"
                
                # Calculate slope for moves
                if ts_delta_df['Actual_Move'].std() > 0:
                    move_slope = move_resid_corr * (ts_delta_df['Move_Residual'].std() / ts_delta_df['Actual_Move'].std())
                else:
                    move_slope = 0
                
                # Typical move size for context
                typical_move = ts_delta_df['Actual_Move'].std()
                move_error_at_typical = abs(move_slope * typical_move)
                
                if move_corr_abs < 0.1:
                    move_sig_badge = '<span style="background:#10b981; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">IDEAL</span>'
                    move_opacity = "1"
                elif move_corr_abs < 0.3:
                    move_sig_badge = '<span style="background:#f59e0b; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">MINOR</span>'
                    move_opacity = "0.9"
                else:
                    move_sig_badge = '<span style="background:#ef4444; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">CONCERN</span>'
                    move_opacity = "1"
                
                st.markdown(f"""
                <div class="interpretation-item" style="opacity: {move_opacity};">
                    <div class="interp-header">
                        <span class="feature-tag">Target Move vs Move Residuals</span>
                        {move_sig_badge}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1px 1fr; gap: 1rem; color: #EAEAEA; font-size: 0.95rem;">
                        <div>
                            <div style="color:#888; font-size:0.8rem; margin-bottom:4px;">CORRELATION STRENGTH</div>
                            The correlation coefficient is <b>r = {move_resid_corr:.3f}</b> ({move_direction}).<br>
                            {"<b>Positive correlation:</b> Model underestimates large upward moves and overestimates large downward moves." if move_resid_corr > 0.1 else "<b>Negative correlation:</b> Model overestimates large upward moves and underestimates large downward moves." if move_resid_corr < -0.1 else "<b>No pattern:</b> Model errors are independent of move size."}
                        </div>
                        <div style="background:var(--border-color);"></div>
                        <div>
                            <div style="color:#FFC300; font-size:0.8rem; margin-bottom:4px;">PRACTICAL IMPACT</div>
                            A typical move of <b>±{typical_move:.2f}</b> units in {target_col} 
                            is associated with an additional prediction error of <b>±{move_error_at_typical:.4f}</b> units.
                            {"<br><span style='color:#10b981;'>✓ Model captures moves accurately.</span>" if move_corr_abs < 0.1 else "<br><span style='color:#f59e0b;'>⚠ Large moves are harder to predict.</span>" if move_corr_abs < 0.3 else "<br><span style='color:#ef4444;'>⚠ Model systematically misses big moves.</span>"}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Cross Correlation Business Interpretation
                cross_corr_abs = abs(cross_corr)
                cross_direction = "positive" if cross_corr > 0 else "negative"
                
                if cross_corr_abs < 0.1:
                    cross_sig_badge = '<span style="background:#10b981; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">IDEAL</span>'
                    cross_opacity = "1"
                elif cross_corr_abs < 0.3:
                    cross_sig_badge = '<span style="background:#f59e0b; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">MINOR</span>'
                    cross_opacity = "0.9"
                else:
                    cross_sig_badge = '<span style="background:#ef4444; padding:2px 8px; border-radius:4px; font-size:0.75em; color:white;">CONCERN</span>'
                    cross_opacity = "1"
                
                st.markdown(f"""
                <div class="interpretation-item" style="opacity: {cross_opacity};">
                    <div class="interp-header">
                        <span class="feature-tag">Target Level vs Move Residuals (Cross)</span>
                        {cross_sig_badge}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1px 1fr; gap: 1rem; color: #EAEAEA; font-size: 0.95rem;">
                        <div>
                            <div style="color:#888; font-size:0.8rem; margin-bottom:4px;">WHAT THIS MEASURES</div>
                            This cross-correlation tests if the <b>current level</b> of {target_col} 
                            affects how well we predict <b>future changes</b>.<br>
                            Correlation: <b>r = {cross_corr:.3f}</b> ({cross_direction})
                        </div>
                        <div style="background:var(--border-color);"></div>
                        <div>
                            <div style="color:#FFC300; font-size:0.8rem; margin-bottom:4px;">BUSINESS MEANING</div>
                            {"<span style='color:#10b981;'>✓ Move predictions work equally well at all {target_col} levels.</span>".format(target_col=target_col) if cross_corr_abs < 0.1 else "<span style='color:#f59e0b;'>⚠ When {target_col} is {'high' if cross_corr > 0 else 'low'}, move predictions tend to {'underestimate' if cross_corr > 0 else 'overestimate'}.</span>".format(target_col=target_col) if cross_corr_abs < 0.3 else "<span style='color:#ef4444;'>⚠ Strong regime dependency: model behaves differently at different {target_col} levels. Consider regime-switching models.</span>".format(target_col=target_col)}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bottom Line for Correlation Analysis
                st.markdown("---")
                
                total_corr_issues = sum([level_corr_abs >= 0.3, move_corr_abs >= 0.3, cross_corr_abs >= 0.3])
                
                if total_corr_issues == 0:
                    corr_verdict = "Residual correlations are healthy — model is well-specified"
                    corr_explain = f"All three correlation tests show no significant relationship between {target_col} values and prediction errors. This means your model's accuracy is consistent regardless of whether {target_col} is high, low, rising, or falling."
                    corr_next = [
                        "No correlation-based adjustments needed",
                        "Model is suitable for predictions across all value ranges",
                        "Proceed to backtest and deployment"
                    ]
                    corr_type = "success"
                elif total_corr_issues == 1:
                    corr_verdict = "Minor correlation pattern detected — model is acceptable with caveats"
                    corr_explain = f"One correlation test shows a pattern. The model may be slightly less accurate in certain conditions, but overall performance should be acceptable for most use cases."
                    corr_next = corr_actions + ["Monitor prediction accuracy at extreme {target_col} values".format(target_col=target_col)]
                    corr_type = "warning"
                else:
                    corr_verdict = "Multiple correlation issues — model specification may need revision"
                    corr_explain = f"Multiple correlations suggest the linear model is missing important patterns. Prediction accuracy varies significantly depending on {target_col} values."
                    corr_next = corr_actions + [
                        "Consider non-linear transformations (log, polynomial)",
                        "Try regime-switching or rolling regression approaches",
                        "Add interaction terms between features and target level"
                    ]
                    corr_type = "danger"
                
                render_bottom_line("Bottom Line: Correlation Health", corr_verdict, corr_explain, corr_next, corr_type)
                
                st.markdown("---")
                
                # Calculate all residual statistics
                mean_resid = ts_df_sorted['Deviation'].mean()
                std_resid = ts_df_sorted['Deviation'].std()
                pos_pct = (ts_df_sorted['Deviation'] > 0).mean() * 100
                final_cumul = ts_df_sorted['Cumulative_Residual'].iloc[-1]
                move_corr_val, _ = stats.pearsonr(ts_delta_df['Actual_Move'], ts_delta_df['Predicted_Move'])
                hit_rate_val = ((ts_delta_df['Actual_Move'] * ts_delta_df['Predicted_Move']) > 0).mean() * 100
                
                # Dynamic interpretation
                insights, actions = interpret_residuals(mean_resid, std_resid, pos_pct, final_cumul)
                
                st.markdown("---")
                st.markdown("### 🎯 Summary & Recommendations")
                
                # Insights
                for insight in insights:
                    st.markdown(insight)
                
                st.markdown("")
                
                # Key findings box
                if hit_rate_val >= 60:
                    direction_verdict = f"✅ Model correctly predicts direction **{hit_rate_val:.0f}%** of the time — useful for directional trades"
                elif hit_rate_val >= 50:
                    direction_verdict = f"⚠️ Model predicts direction **{hit_rate_val:.0f}%** of the time — barely better than random"
                else:
                    direction_verdict = f"❌ Model predicts direction only **{hit_rate_val:.0f}%** of the time — worse than a coin flip"
                
                st.markdown(direction_verdict)
                
                # Actions
                if actions:
                    st.markdown("")
                    st.markdown("**💡 Recommended Actions:**")
                    for action in actions:
                        st.markdown(f"- {action}")
                
                # COMPREHENSIVE BOTTOM LINE
                st.markdown("---")
                
                # Overall model health assessment
                health_score = 0
                health_issues = []
                
                if abs(mean_resid) < std_resid * 0.1:
                    health_score += 25
                else:
                    health_issues.append("systematic bias")
                
                if 40 <= pos_pct <= 60:
                    health_score += 25
                else:
                    health_issues.append("imbalanced residuals")
                
                if abs(final_cumul) < std_resid * 2:
                    health_score += 25
                else:
                    health_issues.append("drift over time")
                
                if hit_rate_val >= 55:
                    health_score += 25
                else:
                    health_issues.append("poor directional accuracy")
                
                if health_score >= 75:
                    overall_verdict = f"Model residuals look healthy (Score: {health_score}/100)"
                    verdict_type = "success"
                    main_message = f"The model's predictions are unbiased and consistent over time. Direction accuracy of {hit_rate_val:.0f}% is acceptable for forecasting."
                elif health_score >= 50:
                    overall_verdict = f"Model has some issues to address (Score: {health_score}/100)"
                    verdict_type = "warning"
                    main_message = f"Issues detected: {', '.join(health_issues)}. The model may still be useful but results should be interpreted with caution."
                else:
                    overall_verdict = f"Model residuals indicate problems (Score: {health_score}/100)"
                    verdict_type = "danger"
                    main_message = f"Significant issues: {', '.join(health_issues)}. Consider revising your model before relying on predictions."
                
                next_steps_final = []
                if "systematic bias" in health_issues:
                    next_steps_final.append("Add features to capture the systematic pattern in residuals")
                if "drift over time" in health_issues:
                    next_steps_final.append("The relationship may be changing — use rolling regression to adapt")
                if "poor directional accuracy" in health_issues:
                    next_steps_final.append("Add momentum/lagged features to improve direction prediction")
                if health_score >= 75:
                    next_steps_final.append("Model is ready for deployment — monitor residuals going forward")
                
                render_bottom_line("Bottom Line: Residual Health Check", overall_verdict, main_message, next_steps_final, verdict_type)


            # ================================================================
            # TAB 7: MODEL SELECTION (Regression Basket with Recommendations)
            # ================================================================
            with tab7:
                st.markdown("### 🎯 Model Selection")
                st.markdown("""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">Intelligent Regression Model Selection</h4>
                    We automatically run <b>7 different regression models</b> on your data and analyze which one is best suited 
                    based on your data characteristics (outliers, multicollinearity, sample size, skewness).
                </div>
                """, unsafe_allow_html=True)
                
                # Data Diagnostics Summary
                st.markdown("---")
                st.markdown("#### 📋 Data Diagnostics")
                
                diag = model_selection['diagnostics']
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    mc_style = "danger" if diag['multicollinearity'] else "success"
                    mc_text = f"High ({diag['max_correlation']:.2f})" if diag['multicollinearity'] else f"OK ({diag['max_correlation']:.2f})"
                    render_metric("Multicollinearity", mc_text, "Max feature correlation", mc_style)
                with c2:
                    out_style = "warning" if diag['has_outliers'] else "success"
                    out_text = f"Yes ({diag['outlier_pct']:.1f}%)" if diag['has_outliers'] else "No"
                    render_metric("Outliers", out_text, "IQR method", out_style)
                with c3:
                    skew_style = "warning" if diag['is_skewed'] else "success"
                    skew_text = f"Yes ({diag['skewness']:.2f})" if diag['is_skewed'] else f"No ({diag['skewness']:.2f})"
                    render_metric("Skewness", skew_text, "Distribution shape", skew_style)
                with c4:
                    sample_style = "warning" if diag['low_sample'] else "success"
                    sample_text = f"Low ({diag['samples_per_feature']:.0f}:1)" if diag['low_sample'] else f"OK ({diag['samples_per_feature']:.0f}:1)"
                    render_metric("Sample/Features", sample_text, "Samples per predictor", sample_style)
                
                # Recommendation Box
                st.markdown("---")
                best_model = model_selection['best_model']
                best_score = model_selection['scores'].get(best_model, 0)
                
                st.markdown(f"""
                <div style="background: rgba(255, 195, 0, 0.1); border: 2px solid #FFC300; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 2rem; margin-right: 1rem;">🏆</span>
                        <div>
                            <h3 style="color: #FFC300; margin: 0;">Recommended Model: {best_model}</h3>
                            <p style="color: #888; margin: 0.25rem 0 0 0;">Score: {best_score:.1f} points</p>
                        </div>
                    </div>
                    <p style="color: #EAEAEA; margin: 0; font-size: 0.95rem;">
                        {basket_results[best_model]['description'] if best_model in basket_results and 'error' not in basket_results[best_model] else 'N/A'}
                    </p>
                    <p style="color: #888; margin: 0.5rem 0 0 0; font-size: 0.85rem;">
                        <b>Best for:</b> {basket_results[best_model]['best_for'] if best_model in basket_results and 'error' not in basket_results[best_model] else 'N/A'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Why this model?
                if best_model in model_selection['reasoning']:
                    st.markdown("**Why this model?**")
                    for reason in model_selection['reasoning'][best_model]:
                        st.markdown(f"- {reason}")
                
                st.markdown("---")
                st.markdown("#### 📊 Model Comparison")
                
                # Build comparison dataframe
                comparison_data = []
                for model_name, result in basket_results.items():
                    if 'error' not in result:
                        comparison_data.append({
                            'Model': model_name,
                            'R²': result['r2'],
                            'Adj R²': result['adj_r2'],
                            'RMSE': result['rmse'],
                            'MAE': result['mae'],
                            'AIC': result['aic'],
                            'BIC': result['bic'],
                            'Score': model_selection['scores'].get(model_name, 0)
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.sort_values('Score', ascending=False)
                    
                    # Highlight best model
                    def highlight_best(row):
                        if row['Model'] == best_model:
                            return ['background-color: rgba(255, 195, 0, 0.2)'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        comp_df.style.apply(highlight_best, axis=1).format({
                            'R²': '{:.4f}',
                            'Adj R²': '{:.4f}',
                            'RMSE': '{:.4f}',
                            'MAE': '{:.4f}',
                            'AIC': '{:.1f}',
                            'BIC': '{:.1f}',
                            'Score': '{:.1f}'
                        }),
                        width="stretch"
                    )
                    
                    # Visual comparison
                    st.markdown("---")
                    st.markdown("#### 📈 Visual Comparison")
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        fig_r2 = px.bar(
                            comp_df.sort_values('R²', ascending=True),
                            x='R²',
                            y='Model',
                            orientation='h',
                            title='R² by Model (Higher is Better)',
                            color='R²',
                            color_continuous_scale=['#ef4444', '#f59e0b', '#10b981']
                        )
                        fig_r2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
                        st.plotly_chart(update_chart_theme(fig_r2), width="stretch")
                    
                    with c2:
                        fig_rmse = px.bar(
                            comp_df.sort_values('RMSE', ascending=False),
                            x='RMSE',
                            y='Model',
                            orientation='h',
                            title='RMSE by Model (Lower is Better)',
                            color='RMSE',
                            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444']
                        )
                        fig_rmse.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
                        st.plotly_chart(update_chart_theme(fig_rmse), width="stretch")
                
                # Model Details Expanders
                st.markdown("---")
                st.markdown("#### 🔍 Model Details")
                
                for model_name, result in basket_results.items():
                    if 'error' not in result:
                        is_best = model_name == best_model
                        icon = "🏆 " if is_best else ""
                        
                        with st.expander(f"{icon}{model_name} — R²: {result['r2']:.4f}, RMSE: {result['rmse']:.4f}"):
                            st.markdown(f"**Description:** {result['description']}")
                            st.markdown(f"**Best For:** {result['best_for']}")
                            
                            # Model-specific info
                            if 'alpha' in result:
                                st.markdown(f"**Optimal Alpha:** {result['alpha']:.4f}")
                            if 'l1_ratio' in result:
                                st.markdown(f"**L1 Ratio:** {result['l1_ratio']:.2f}")
                            if 'selected_features' in result:
                                st.markdown(f"**Selected Features ({result['n_selected']}):** {', '.join(result['selected_features']) if result['selected_features'] else 'None'}")
                            if 'n_outliers' in result:
                                st.markdown(f"**Outliers Removed:** {result['n_outliers']} ({result['outlier_pct']:.1f}%)")
                            
                            # Coefficients
                            st.markdown("**Coefficients:**")
                            coef_df = pd.DataFrame({
                                'Feature': list(result['coefficients'].keys()),
                                'Coefficient': list(result['coefficients'].values())
                            })
                            st.dataframe(coef_df.style.format({'Coefficient': '{:.6f}'}), width="stretch")
                    else:
                        with st.expander(f"❌ {model_name} — Error"):
                            st.error(result['error'])
                
                # Bottom Line
                st.markdown("---")
                
                # Determine verdict based on diagnostics
                issues = []
                if diag['multicollinearity']:
                    issues.append("multicollinearity")
                if diag['has_outliers']:
                    issues.append("outliers")
                if diag['is_skewed']:
                    issues.append("skewed distribution")
                if diag['low_sample']:
                    issues.append("limited sample size")
                
                if not issues:
                    verdict = f"Your data is clean — {best_model} is recommended for optimal performance"
                    verdict_type = "success"
                    explanation = f"No significant data quality issues detected. The {best_model} model achieves R² = {basket_results[best_model]['r2']:.3f} and RMSE = {basket_results[best_model]['rmse']:.4f}."
                    next_steps = [
                        f"Use {best_model} for your predictions",
                        "The model results in other tabs are based on OLS — consider if {best_model} coefficients differ significantly".format(best_model=best_model),
                        "Monitor performance on new data"
                    ]
                else:
                    verdict = f"Data has challenges ({', '.join(issues)}) — {best_model} is specifically chosen to handle them"
                    verdict_type = "warning"
                    explanation = f"Your data has some characteristics that standard OLS may not handle optimally. The {best_model} model is designed to address these issues while achieving R² = {basket_results[best_model]['r2']:.3f}."
                    next_steps = [
                        f"Use {best_model} instead of OLS for more reliable predictions",
                        "Review the coefficient differences between models",
                        "Consider addressing the underlying data issues if possible"
                    ]
                    if diag['multicollinearity']:
                        next_steps.append("High correlation between predictors — consider removing redundant features")
                    if diag['has_outliers']:
                        next_steps.append("Outliers detected — investigate if they are errors or genuine extreme values")
                
                render_bottom_line("Bottom Line: Model Recommendation", verdict, explanation, next_steps, verdict_type)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Landing page - Tabbed with Cards
        
        # Landing page tabs (no redundant header needed - main header already shows)
        landing_tab1, landing_tab2, landing_tab3 = st.tabs(["🚀 Get Started", "📊 Modules", "📖 About"])
        
        # TAB 1: GET STARTED
        with landing_tab1:
            st.markdown("")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="metric-card" style="border-left: 3px solid #FFC300; min-height: 200px;">
                        <h4 style="color: #FFC300;">📤 UPLOAD FILE</h4>
                        <h2 style="font-size: 1.3rem; color: #EAEAEA; margin-bottom: 1rem;">Local Data</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6;">
                            Upload CSV or Excel files directly from your computer. 
                            Supports any tabular data with numeric columns.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card" style="border-left: 3px solid #10b981; min-height: 200px;">
                        <h4 style="color: #10b981;">📊 GOOGLE SHEETS</h4>
                        <h2 style="font-size: 1.3rem; color: #EAEAEA; margin-bottom: 1rem;">Cloud Data</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6;">
                            Connect directly via URL. Pre-configured with NIFTY50 PE analysis 
                            including interest rates, yields, and market ratios.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            st.markdown("")
            
            st.markdown("""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">Quick Start</h4>
                    <b>1.</b> Select data source in sidebar → 
                    <b>2.</b> Choose Target (Y) → 
                    <b>3.</b> Select Predictors (X) → 
                    <b>4.</b> Explore analysis tabs
                </div>
            """, unsafe_allow_html=True)
        
        # TAB 2: MODULES
        with landing_tab2:
            st.markdown("")
            
            # Primary Analysis Row
            st.markdown("##### Primary Analysis")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("""
                    <div class="metric-card">
                        <h4>📉 RESIDUALS</h4>
                        <h2 style="font-size: 1.1rem;">Time Series</h2>
                        <p class="sub-metric">Prediction errors, correlation analysis, drift detection</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("""
                    <div class="metric-card">
                        <h4>📊 PERFORMANCE</h4>
                        <h2 style="font-size: 1.1rem;">Model Fit</h2>
                        <p class="sub-metric">R², Adjusted R², RMSE, F-statistic, significance</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown("""
                    <div class="metric-card">
                        <h4>📐 EQUATION</h4>
                        <h2 style="font-size: 1.1rem;">Coefficients</h2>
                        <p class="sub-metric">Model equation, business interpretation, drivers</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Validation Row
            st.markdown("##### Validation & Diagnostics")
            c4, c5, c6 = st.columns(3)
            
            with c4:
                st.markdown("""
                    <div class="metric-card">
                        <h4>🔍 PREDICTIONS</h4>
                        <h2 style="font-size: 1.1rem;">Actual vs Predicted</h2>
                        <p class="sub-metric">Deviation analysis, extreme misses, bias</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c5:
                st.markdown("""
                    <div class="metric-card">
                        <h4>🌊 MOVES</h4>
                        <h2 style="font-size: 1.1rem;">Delta Analysis</h2>
                        <p class="sub-metric">Period-over-period changes, directional accuracy</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c6:
                st.markdown("""
                    <div class="metric-card">
                        <h4>🛠️ DIAGNOSTICS</h4>
                        <h2 style="font-size: 1.1rem;">Statistical Tests</h2>
                        <p class="sub-metric">Normality, heteroscedasticity, autocorrelation, VIF</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Model Selection Row
            st.markdown("##### Intelligent Model Selection")
            c7, c8, c9 = st.columns([1, 2, 1])
            
            with c8:
                st.markdown("""
                    <div class="metric-card" style="border-color: #FFC300;">
                        <h4 style="color: #FFC300;">🎯 MODEL SELECTION</h4>
                        <h2 style="font-size: 1.1rem;">Regression Basket</h2>
                        <p class="sub-metric">Automatically compares 7 models (OLS, Ridge, Lasso, Elastic Net, Huber, RANSAC, Quantile) and recommends the best one based on your data characteristics</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # TAB 3: ABOUT
        with landing_tab3:
            st.markdown("")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="metric-card" style="min-height: 180px;">
                        <h4>🎯 PURPOSE</h4>
                        <h2 style="font-size: 1.2rem; color: #EAEAEA;">Quantitative Analysis</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6; margin-top: 1rem;">
                            Professional-grade regression analysis for financial modeling, 
                            research, and data-driven decision making.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card" style="min-height: 180px;">
                        <h4>⚡ FEATURES</h4>
                        <h2 style="font-size: 1.2rem; color: #EAEAEA;">7 Analysis Modules</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6; margin-top: 1rem;">
                            Comprehensive regression analysis with intelligent model selection 
                            that automatically compares 7 different regression algorithms.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("""
                    <div class="metric-card" style="min-height: 180px;">
                        <h4>📈 DEFAULT DATA</h4>
                        <h2 style="font-size: 1.2rem; color: #EAEAEA;">NIFTY50 PE Analysis</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6; margin-top: 1rem;">
                            Pre-loaded dataset with Indian & US interest rates, 
                            yield curves, and market valuation metrics.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                    <div class="metric-card" style="min-height: 180px;">
                        <h4>🏛️ SUITE</h4>
                        <h2 style="font-size: 1.2rem; color: #EAEAEA;">Arthagati Analytics</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6; margin-top: 1rem;">
                            Part of the Arthagati quantitative analytics suite 
                            for institutional-grade financial analysis.
                        </p>
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

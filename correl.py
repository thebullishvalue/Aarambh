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

            model, err = run_regression(data, target_col, feature_cols)
            
            if err:
                st.error(f"Model Error: {err}")
                return

            # --- TABS (Organized by workflow) ---
            # Group 1: Primary Analysis (What you came here for)
            # Group 2: Model Understanding (How the model works)
            # Group 3: Validation & Testing (Is the model reliable?)
            # Group 4: Advanced Tools (Power user features)
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
                "📉 Residuals",           # PRIMARY: Time series residual analysis
                "📊 Performance",          # Model fit metrics
                "📐 Equation",             # Coefficients & interpretation
                "🔍 Predictions",          # Actual vs predicted deep dive
                "🌊 Moves",                # Delta/change analysis
                "🛠️ Diagnostics",          # Statistical tests
                "🔮 Simulate",             # What-if & backtesting
                "⚙️ Models",               # Advanced regression types
                "📈 Rolling",              # Time-varying analysis
                "🧪 Features",             # Feature engineering
                "🏆 Compare"               # Model comparison
            ])

            # ================================================================
            # TAB 2: PERFORMANCE
            # ================================================================
            with tab2:
                r2 = model.rsquared
                adj_r2 = model.rsquared_adj
                rmse = np.sqrt(model.mse_resid)
                f_pval = model.f_pvalue
                n_obs = int(model.nobs)
                
                # Executive Summary at the top
                summary_text, verdict = generate_model_summary(model, r2, adj_r2, rmse, f_pval, n_obs, len(feature_cols), target_col)
                st.markdown(summary_text)
                
                st.markdown("---")
                
                # Key Metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1: render_metric("R-Squared", f"{r2:.4f}", f"Explains {r2:.1%} of variance", "gold")
                with c2: render_metric("Adj. R-Squared", f"{adj_r2:.4f}", f"Penalized for {len(feature_cols)} features", "gold")
                with c3: render_metric("RMSE", f"{rmse:.4f}", f"±{rmse:.2f} avg error", "success")
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
                    
                    # Identify significant vs insignificant features
                    sig_feats = [f for f in feature_cols if model.pvalues[f] < 0.05]
                    insig_feats = [f for f in feature_cols if model.pvalues[f] >= 0.05]
                    
                    if sig_feats:
                        st.markdown(f"**✅ Significant predictors:** {', '.join(sig_feats)}")
                    if insig_feats:
                        st.markdown(f"**⚠️ Weak predictors:** {', '.join(insig_feats)}")

            # ================================================================
            # TAB 3: EQUATION & COEFFICIENTS
            # ================================================================
            with tab3:
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
                }), width="stretch")
                
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
                
                # BOTTOM LINE SUMMARY
                st.markdown("---")
                
                # Determine key drivers
                sig_features = [(f, model.params[f], model.pvalues[f]) for f in feature_cols if model.pvalues[f] < 0.05]
                insig_features = [f for f in feature_cols if model.pvalues[f] >= 0.05]
                
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
                preds = model.predict(sm.add_constant(data[feature_cols]))
                analysis_df = data.copy()
                analysis_df['Predicted'] = preds
                analysis_df['Deviation'] = analysis_df[target_col] - analysis_df['Predicted']
                
                st.markdown("#### Deviation Distribution")
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_dev = px.histogram(analysis_df['Deviation'], nbins=30, title="Prediction Errors")
                    fig_dev.add_vline(x=0, line_dash="dash", line_color="#FFC300")
                    st.plotly_chart(update_chart_theme(fig_dev), width="stretch")
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
                st.markdown("### 🌊 Move Analysis")
                st.markdown("""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">Analyzing Changes Instead of Levels</h4>
                    This tab calculates the <b>Period-over-Period Change (Delta)</b> for your features.<br>
                    It asks: <i>"Based on the model coefficients, did the Target change as expected given the changes in Features?"</i><br>
                    <br>
                    <b>Predicted Move</b> = Σ (Δ Feature × Coefficient)<br>
                    <b>Move Residual</b> = Actual Move − Predicted Move
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
                st.markdown("### 📉 Residual Analysis")
                st.markdown("""
                <div class="guide-box">
                    <h4 style="color:#FFC300; margin-top:0;">How Prediction Errors Evolve Over Time</h4>
                    Identify periods where the model systematically over or under-predicts,
                    detect regime changes, and spot anomalies in your data.
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare data for both charts
                preds_ts = model.predict(sm.add_constant(data[feature_cols]))
                ts_analysis_df = data.copy()
                ts_analysis_df['Predicted'] = preds_ts
                ts_analysis_df['Deviation'] = ts_analysis_df[target_col] - ts_analysis_df['Predicted']
                
                # Delta data
                if date_col_option != "None" and date_col_option in data.columns:
                    ts_delta_df = data.sort_values(by=date_col_option).copy()
                else:
                    ts_delta_df = data.sort_index().copy()
                
                ts_delta_df['Actual_Move'] = ts_delta_df[target_col].diff()
                ts_delta_df['Predicted_Move'] = 0.0
                for feat in feature_cols:
                    coef = model.params[feat]
                    ts_delta_df['Predicted_Move'] += ts_delta_df[feat].diff() * coef
                ts_delta_df['Move_Residual'] = ts_delta_df['Actual_Move'] - ts_delta_df['Predicted_Move']
                ts_delta_df = ts_delta_df.dropna()
                
                # ---- CHART 1: Residuals Over Time (Full Series) ----
                st.markdown("#### 📊 Residuals Over Time (Full Series)")
                st.caption("Shows how prediction errors (Actual - Predicted) vary across time. Persistent positive/negative values indicate systematic bias.")
                
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
            # TAB 7: SIMULATE
            # ================================================================
            with tab7:
                st.markdown("### 🔮 Simulation & Backtesting")
                
                subtab_sim, subtab_bt, subtab_wf, subtab_mc = st.tabs([
                    "🎛️ What-If", 
                    "🔙 Backtest",
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
                            train_r2 = bt_model.rsquared
                            
                            # Calculate test R²
                            ss_res = np.sum((test_df[target_col] - test_preds) ** 2)
                            ss_tot = np.sum((test_df[target_col] - test_df[target_col].mean()) ** 2)
                            test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                            
                            c1, c2 = st.columns([2, 1])
                            
                            with c1:
                                fig_bt = go.Figure()
                                fig_bt.add_trace(go.Scatter(y=bt_data[target_col], mode='lines', 
                                                           name='Actual', line=dict(color='#444')))
                                fig_bt.add_trace(go.Scatter(x=test_df.index, y=test_preds, mode='lines',
                                                           name='Test Prediction', line=dict(color='#00E396')))
                                fig_bt.add_vline(x=split_idx, line_dash="dash", line_color="#FFC300")
                                st.plotly_chart(update_chart_theme(fig_bt), width="stretch")
                                
                            with c2:
                                render_metric("Train RMSE", f"{train_rmse:.4f}", "In-Sample", "info")
                                render_metric("Test RMSE", f"{test_rmse:.4f}", "Out-of-Sample", "primary")
                            
                            # Intelligent interpretation
                            st.markdown("---")
                            st.markdown("#### 🎯 Backtest Verdict")
                            
                            status, explanation, action = interpret_backtest(train_rmse, test_rmse, train_r2, test_r2)
                            st.markdown(f"**{status}**")
                            st.markdown(explanation)
                            st.markdown(f"**💡 Recommendation:** {action}")
                            
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
                                }), width="stretch")
                                
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
                                st.plotly_chart(update_chart_theme(fig_wf), width="stretch")
                                
                                # BOTTOM LINE for Walk-Forward
                                avg_r2 = wf_results['R²'].mean()
                                avg_dir = wf_results['Dir Accuracy'].mean()
                                r2_stability = wf_results['R²'].std()
                                
                                if avg_r2 >= 0.5 and r2_stability < 0.1:
                                    wf_verdict = f"Model is stable across time periods (Avg R²: {avg_r2:.2f}, Stability: {r2_stability:.3f})"
                                    wf_type = "success"
                                    wf_explain = "Performance is consistent across different time windows. The model should perform reliably on future data."
                                elif avg_r2 >= 0.3:
                                    wf_verdict = f"Model has moderate but variable performance (Avg R²: {avg_r2:.2f})"
                                    wf_type = "warning"
                                    wf_explain = "Performance varies across time periods. Be cautious about relying on predictions during volatile periods."
                                else:
                                    wf_verdict = f"Model performs poorly out-of-sample (Avg R²: {avg_r2:.2f})"
                                    wf_type = "danger"
                                    wf_explain = "The model fails to generalize to new data. It may be overfit to the training period."
                                
                                wf_steps = []
                                if avg_dir >= 0.55:
                                    wf_steps.append(f"Direction accuracy ({avg_dir:.0%}) is useful for trend-following strategies")
                                if r2_stability > 0.1:
                                    wf_steps.append("High variability across folds — consider regime-specific models")
                                wf_steps.append("Use the worst-fold performance as a conservative estimate")
                                
                                render_bottom_line("Bottom Line: Walk-Forward Validation", wf_verdict, wf_explain, wf_steps, wf_type)
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
                                st.plotly_chart(update_chart_theme(fig_mc), width="stretch")
                                
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
            # TAB 8: MODELS (Advanced Regression Types)
            # ================================================================
            with tab8:
                st.markdown("### ⚙️ Advanced Models")
                
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
                        st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}), width="stretch")
                        
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
                        st.plotly_chart(update_chart_theme(fig_adv), width="stretch")

            # ================================================================
            # TAB 9: ROLLING
            # ================================================================
            with tab9:
                st.markdown("### 📈 Rolling Analysis")
                
                st.markdown("""
                <div class="guide-box">
                    <b>Time-Varying Coefficients:</b> See how relationships change over time.<br>
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
                                st.plotly_chart(update_chart_theme(fig_r2), width="stretch")
                                
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
                                st.plotly_chart(update_chart_theme(fig_coef), width="stretch")
                                
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
                                }), width="stretch")
                            else:
                                st.error("Rolling regression failed.")

            # ================================================================
            # TAB 10: FEATURES
            # ================================================================
            with tab10:
                st.markdown("### 🧪 Feature Engineering")
                
                st.markdown("""
                <div class="guide-box">
                    <b>Generate New Features:</b> Create lags, differences, interactions, polynomials, and more.
                    Preview engineered features before using them in your model.
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
                            st.dataframe(eng_data[preview_cols].head(10), width="stretch")
                            
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
                        st.plotly_chart(update_chart_theme(fig_pca), width="stretch")
                        
                        # Recommendation
                        for i, cum_var in enumerate(pca_results['cumulative_variance']):
                            if cum_var >= 0.95:
                                st.info(f"💡 {i+1} principal components explain 95%+ of variance "
                                       f"(vs {len(feature_cols)} original features)")
                                break

            # ================================================================
            # TAB 11: COMPARE
            # ================================================================
            with tab11:
                st.markdown("### 🏆 Model Comparison")
                
                st.markdown("""
                <div class="guide-box">
                    <b>Compare Models:</b> Test different feature sets side-by-side.
                    Use AIC/BIC criteria to select the best model that balances fit and complexity.
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
                                }), width="stretch")
                                
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
                                st.plotly_chart(update_chart_theme(fig_comp), width="stretch")
                                
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
                        <p class="sub-metric">Prediction errors over time, drift detection, move residuals</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("""
                    <div class="metric-card">
                        <h4>📊 PERFORMANCE</h4>
                        <h2 style="font-size: 1.1rem;">Model Fit</h2>
                        <p class="sub-metric">R², Adjusted R², RMSE, F-statistic, significance tests</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown("""
                    <div class="metric-card">
                        <h4>📐 EQUATION</h4>
                        <h2 style="font-size: 1.1rem;">Coefficients</h2>
                        <p class="sub-metric">Model equation, p-values, confidence intervals, interpretation</p>
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
                        <p class="sub-metric">Deviation analysis, extreme misses, bias detection</p>
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
            
            # Advanced Row
            st.markdown("##### Advanced Tools")
            c7, c8, c9 = st.columns(3)
            
            with c7:
                st.markdown("""
                    <div class="metric-card">
                        <h4>🔮 SIMULATE</h4>
                        <h2 style="font-size: 1.1rem;">Backtesting</h2>
                        <p class="sub-metric">What-if scenarios, walk-forward, Monte Carlo</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c8:
                st.markdown("""
                    <div class="metric-card">
                        <h4>⚙️ MODELS</h4>
                        <h2 style="font-size: 1.1rem;">Advanced Regression</h2>
                        <p class="sub-metric">Ridge, Lasso, Elastic Net, Huber, RANSAC, Quantile</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c9:
                st.markdown("""
                    <div class="metric-card">
                        <h4>📈 ROLLING</h4>
                        <h2 style="font-size: 1.1rem;">Time-Varying</h2>
                        <p class="sub-metric">Rolling coefficients, regime detection, stability</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Tools Row
            c10, c11, c12 = st.columns(3)
            
            with c10:
                st.markdown("""
                    <div class="metric-card">
                        <h4>🧪 FEATURES</h4>
                        <h2 style="font-size: 1.1rem;">Engineering</h2>
                        <p class="sub-metric">Lags, differences, interactions, polynomials, PCA</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c11:
                st.markdown("""
                    <div class="metric-card">
                        <h4>🏆 COMPARE</h4>
                        <h2 style="font-size: 1.1rem;">Model Selection</h2>
                        <p class="sub-metric">AIC/BIC criteria, nested F-tests, side-by-side</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with c12:
                pass
        
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
                        <h2 style="font-size: 1.2rem; color: #EAEAEA;">11 Analysis Modules</h2>
                        <p style="color: #888; font-size: 0.9rem; line-height: 1.6; margin-top: 1rem;">
                            From basic OLS to advanced regularization, rolling analysis, 
                            Monte Carlo simulation, and model comparison.
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

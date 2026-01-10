"""
Aarambh - Residual Oscillator Prediction System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fair value model with mean-reverting residual oscillator.
Multi-lookback breadth analysis for market top/bottom detection.
Inspired by UMA's Zone Trends and Regime Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import argrelextrema
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Dependencies ---
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import RidgeCV, HuberRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Page Config ---
st.set_page_config(
    page_title="Aarambh | Residual Oscillator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed" # Changed from "expanded" to "collapsed"
)

# --- Premium CSS ---
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
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    /* Updated to 100% width for maximum canvas size */
    .block-container { padding-top: 1rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
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
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    
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
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    
    .metric-card h4 { 
        color: var(--text-muted); 
        font-size: 0.75rem; 
        margin-bottom: 0.5rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        min-height: 30px;
        display: flex;
        align-items: center;
    }
    
    .metric-card h3 { color: var(--text-primary); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-card p { color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }
    
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.purple h2 { color: var(--purple); }
    .metric-card.neutral h2 { color: var(--neutral); }

    .signal-card {
        background: var(--bg-card);
        border-radius: 16px;
        border: 2px solid var(--border-color);
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card.overvalued { border-color: var(--danger-red); box-shadow: 0 0 30px rgba(239, 68, 68, 0.15); }
    .signal-card.undervalued { border-color: var(--success-green); box-shadow: 0 0 30px rgba(16, 185, 129, 0.15); }
    .signal-card.fair { border-color: var(--primary-color); box-shadow: 0 0 30px rgba(255, 195, 0, 0.15); }
    
    .signal-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }
    .signal-card .value { font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .signal-card .subtext { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.75rem; }
    .signal-card.overvalued .value { color: var(--danger-red); }
    .signal-card.undervalued .value { color: var(--success-green); }
    .signal-card.fair .value { color: var(--primary-color); }

    .guide-box {
        background: rgba(var(--primary-rgb), 0.05);
        border-left: 3px solid var(--primary-color);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    .guide-box.success { background: rgba(16, 185, 129, 0.05); border-left-color: var(--success-green); }
    .guide-box.danger { background: rgba(239, 68, 68, 0.05); border-left-color: var(--danger-red); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 0px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }

    .oscillator-gauge {
        background: var(--bg-elevated);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .gauge-bar {
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, var(--success-green), var(--primary-color), var(--danger-red));
        position: relative;
        margin: 1rem 0;
    }
    
    .gauge-marker {
        position: absolute;
        top: -6px;
        width: 24px;
        height: 24px;
        background: white;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 10px rgba(255,255,255,0.5);
        border: 3px solid var(--bg-card);
    }
    
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .conviction-meter {
        background: var(--bg-elevated);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .conviction-bar {
        height: 8px;
        border-radius: 4px;
        background: var(--border-color);
        overflow: hidden;
    }
    
    .conviction-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .section-divider {
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }

    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover { background: var(--primary-color); color: #1A1A1A; }
    
    .app-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
        text-align: center;
        color: var(--text-muted);
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MULTI-LOOKBACK RESIDUAL OSCILLATOR ENGINE
# ============================================================================

class ResidualOscillatorPro:
    """
    Enhanced residual oscillator with multi-lookback breadth analysis.
    
    Like UMA tracks multiple stocks, we track multiple lookback periods
    to create "breadth" signals for market top/bottom detection.
    """
    
    LOOKBACKS = [5, 10, 20, 50, 100]  # Multiple timeframes like multiple stocks
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def fit(self, X, y, feature_names=None):
        """Fit fair value model and compute multi-lookback analytics."""
        self.feature_names = feature_names or [f'X{i}' for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()
        self.X = X.copy()
        
        # Default behavior if libs are missing
        all_preds = []
        self.models = {}

        if SKLEARN_AVAILABLE:
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit ensemble model
            try:
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
                ridge.fit(X_scaled, y)
                self.models['ridge'] = ridge
                all_preds.append(ridge.predict(X_scaled))
            except: pass
            
            try:
                huber = HuberRegressor(epsilon=1.35, max_iter=1000)
                huber.fit(X_scaled, y)
                self.models['huber'] = huber
                all_preds.append(huber.predict(X_scaled))
            except: pass

        if STATSMODELS_AVAILABLE:
            try:
                X_ols = sm.add_constant(X)
                ols = sm.OLS(y, X_ols).fit()
                self.models['ols'] = ols
                all_preds.append(ols.predict(X_ols))
            except: pass
        
        # Fallback if no models worked
        if all_preds:
            self.predictions = np.mean(all_preds, axis=0)
        else:
            # Simple fallback if no libs: Mean
            self.predictions = np.full_like(y, np.mean(y))

        self.residuals = y - self.predictions
        
        # Compute all analytics
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        self._find_pivots()        
        self._compute_divergences()
        self._compute_forward_returns()
        
        return self
    
    def _compute_multi_lookback_signals(self):
        """Compute z-scores and zones for each lookback period."""
        r = self.residuals
        n = len(r)
        
        self.lookback_data = {}
        
        for lb in self.LOOKBACKS:
            if n < lb:
                continue
                
            # Rolling z-score
            rolling_mean = pd.Series(r).rolling(lb).mean().values
            rolling_std = pd.Series(r).rolling(lb).std().values
            z_scores = np.where(rolling_std > 0, (r - rolling_mean) / rolling_std, 0)
            
            # Zone classification
            zones = []
            for z in z_scores:
                if pd.isna(z):
                    zones.append('N/A')
                elif z > 2:
                    zones.append('Extreme Over')
                elif z > 1:
                    zones.append('Overvalued')
                elif z > -1:
                    zones.append('Fair Value')
                elif z > -2:
                    zones.append('Undervalued')
                else:
                    zones.append('Extreme Under')
            
            # Buy/Sell signals (threshold crossings)
            buy_signals = np.zeros(n, dtype=bool)
            sell_signals = np.zeros(n, dtype=bool)
            
            for i in range(1, n):
                if not pd.isna(z_scores[i]) and not pd.isna(z_scores[i-1]):
                    if z_scores[i] < -1 and z_scores[i-1] >= -1:
                        buy_signals[i] = True
                    if z_scores[i] > 1 and z_scores[i-1] <= 1:
                        sell_signals[i] = True
            
            self.lookback_data[lb] = {
                'z_scores': z_scores,
                'zones': np.array(zones),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
            }
        
        # Create master time series DataFrame
        self.ts_data = pd.DataFrame({
            'Actual': self.y,
            'FairValue': self.predictions,
            'Residual': self.residuals,
        })
        
        # Add each lookback's data
        for lb in self.lookback_data:
            self.ts_data[f'Z_{lb}'] = self.lookback_data[lb]['z_scores']
            self.ts_data[f'Zone_{lb}'] = self.lookback_data[lb]['zones']
            self.ts_data[f'Buy_{lb}'] = self.lookback_data[lb]['buy_signals']
            self.ts_data[f'Sell_{lb}'] = self.lookback_data[lb]['sell_signals']
    
    def _compute_breadth_metrics(self):
        """Compute breadth metrics - like % of stocks in each zone."""
        n = len(self.ts_data)
        valid_lookbacks = [lb for lb in self.LOOKBACKS if lb in self.lookback_data]
        
        # For each observation, count how many lookbacks are in each zone
        oversold_count = np.zeros(n)
        overbought_count = np.zeros(n)
        extreme_oversold = np.zeros(n)
        extreme_overbought = np.zeros(n)
        buy_signal_count = np.zeros(n)
        sell_signal_count = np.zeros(n)
        avg_z = np.zeros(n)
        
        for i in range(n):
            z_values = []
            for lb in valid_lookbacks:
                zone = self.lookback_data[lb]['zones'][i]
                z = self.lookback_data[lb]['z_scores'][i]
                
                if zone == 'Extreme Under':
                    extreme_oversold[i] += 1
                    oversold_count[i] += 1
                elif zone == 'Undervalued':
                    oversold_count[i] += 1
                elif zone == 'Extreme Over':
                    extreme_overbought[i] += 1
                    overbought_count[i] += 1
                elif zone == 'Overvalued':
                    overbought_count[i] += 1
                
                if self.lookback_data[lb]['buy_signals'][i]:
                    buy_signal_count[i] += 1
                if self.lookback_data[lb]['sell_signals'][i]:
                    sell_signal_count[i] += 1
                
                if not pd.isna(z):
                    z_values.append(z)
            
            if z_values:
                avg_z[i] = np.mean(z_values)
        
        num_lookbacks = len(valid_lookbacks)
        
        self.ts_data['OversoldBreadth'] = (oversold_count / num_lookbacks * 100) if num_lookbacks > 0 else 0
        self.ts_data['OverboughtBreadth'] = (overbought_count / num_lookbacks * 100) if num_lookbacks > 0 else 0
        self.ts_data['ExtremeOversold'] = (extreme_oversold / num_lookbacks * 100) if num_lookbacks > 0 else 0
        self.ts_data['ExtremeOverbought'] = (extreme_overbought / num_lookbacks * 100) if num_lookbacks > 0 else 0
        self.ts_data['BuySignalBreadth'] = buy_signal_count
        self.ts_data['SellSignalBreadth'] = sell_signal_count
        self.ts_data['AvgZ'] = avg_z
        
        # Composite conviction score (-100 to +100)
        # Negative = Oversold conviction, Positive = Overbought conviction
        self.ts_data['ConvictionScore'] = (
            self.ts_data['OverboughtBreadth'] - self.ts_data['OversoldBreadth'] +
            (self.ts_data['ExtremeOverbought'] - self.ts_data['ExtremeOversold']) * 0.5
        )
        
        # Regime classification
        regimes = []
        for score in self.ts_data['ConvictionScore']:
            if score < -40:
                regimes.append('STRONGLY OVERSOLD')
            elif score < -20:
                regimes.append('OVERSOLD')
            elif score > 40:
                regimes.append('STRONGLY OVERBOUGHT')
            elif score > 20:
                regimes.append('OVERBOUGHT')
            else:
                regimes.append('NEUTRAL')
        
        self.ts_data['Regime'] = regimes
    
    def _compute_divergences(self):
        """Detect price vs residual divergences."""
        n = len(self.ts_data)
        lookback = 5
        
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        
        price = self.y
        residual = self.residuals
        
        for i in range(lookback, n):
            price_change = price[i] - price[i - lookback]
            residual_change = residual[i] - residual[i - lookback]
            
            # Bullish divergence: price falling but residual rising (less negative)
            if price_change < 0 and residual_change > 0 and residual[i] < -self.residual_stats['std']:
                bull_div[i] = True
            
            # Bearish divergence: price rising but residual falling (less positive)
            if price_change > 0 and residual_change < 0 and residual[i] > self.residual_stats['std']:
                bear_div[i] = True
        
        self.ts_data['BullishDiv'] = bull_div
        self.ts_data['BearishDiv'] = bear_div
    
    def _find_pivots(self, order=5):
        """Find pivot points in residuals."""
        r = self.residuals
        
        max_idx = argrelextrema(r, np.greater, order=order)[0]
        min_idx = argrelextrema(r, np.less, order=order)[0]
        
        self.pivots = {
            'tops': max_idx,
            'bottoms': min_idx,
            'avg_top': np.mean(r[max_idx]) if len(max_idx) > 0 else np.mean(r) + np.std(r),
            'avg_bottom': np.mean(r[min_idx]) if len(min_idx) > 0 else np.mean(r) - np.std(r),
            'std_top': np.std(r[max_idx]) if len(max_idx) > 1 else np.std(r),
            'std_bottom': np.std(r[min_idx]) if len(min_idx) > 1 else np.std(r),
        }
        
        self.ts_data['IsPivotTop'] = False
        self.ts_data['IsPivotBottom'] = False
        if len(max_idx) > 0:
            self.ts_data.loc[max_idx, 'IsPivotTop'] = True
        if len(min_idx) > 0:
            self.ts_data.loc[min_idx, 'IsPivotBottom'] = True
        
        # Global residual stats
        self.residual_stats = {
            'mean': np.mean(r),
            'std': np.std(r),
            'current': r[-1],
            'current_zscore': (r[-1] - np.mean(r)) / np.std(r) if np.std(r) > 0 else 0,
            'percentile': stats.percentileofscore(r, r[-1]),
            'min': np.min(r),
            'max': np.max(r),
        }
    
    def _compute_forward_returns(self):
        """Compute forward returns for signal validation."""
        n = len(self.ts_data)
        y = self.y
        
        for period in [5, 10, 20]:
            fwd_ret = np.full(n, np.nan)
            for i in range(n - period):
                fwd_ret[i] = (y[i + period] - y[i]) / y[i] * 100
            self.ts_data[f'FwdRet_{period}'] = fwd_ret
    
    def get_current_signal(self):
        """Get comprehensive current signal."""
        ts = self.ts_data
        current = ts.iloc[-1]
        
        conviction = current['ConvictionScore']
        regime = current['Regime']
        oversold_breadth = current['OversoldBreadth']
        overbought_breadth = current['OverboughtBreadth']
        
        # Signal determination
        if conviction < -60:
            signal, strength = 'BUY', 'STRONG'
        elif conviction < -40:
            signal, strength = 'BUY', 'MODERATE'
        elif conviction < -20:
            signal, strength = 'BUY', 'WEAK'
        elif conviction > 60:
            signal, strength = 'SELL', 'STRONG'
        elif conviction > 40:
            signal, strength = 'SELL', 'MODERATE'
        elif conviction > 20:
            signal, strength = 'SELL', 'WEAK'
        else:
            signal, strength = 'HOLD', 'NEUTRAL'
        
        # Confidence based on breadth agreement
        if signal == 'BUY':
            confidence = 'HIGH' if oversold_breadth >= 80 else 'MEDIUM' if oversold_breadth >= 60 else 'LOW'
        elif signal == 'SELL':
            confidence = 'HIGH' if overbought_breadth >= 80 else 'MEDIUM' if overbought_breadth >= 60 else 'LOW'
        else:
            confidence = 'N/A'
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'conviction_score': conviction,
            'regime': regime,
            'oversold_breadth': oversold_breadth,
            'overbought_breadth': overbought_breadth,
            'residual': current['Residual'],
            'fair_value': current['FairValue'],
            'actual': current['Actual'],
            'avg_z': current['AvgZ'],
            'has_bullish_div': current['BullishDiv'],
            'has_bearish_div': current['BearishDiv'],
        }
    
    def get_model_stats(self):
        """Get model fit statistics."""
        if SKLEARN_AVAILABLE:
            r2 = r2_score(self.y, self.predictions)
            rmse = np.sqrt(mean_squared_error(self.y, self.predictions))
            mae = mean_absolute_error(self.y, self.predictions)
        else:
            r2, rmse, mae = 0, 0, 0 # Placeholders if sklearn missing
            
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_obs': len(self.y),
            'n_features': len(self.feature_names),
        }
    
    def get_regime_stats(self):
        """Get regime distribution statistics."""
        ts = self.ts_data
        n = len(ts)
        
        regime_counts = ts['Regime'].value_counts()
        
        return {
            'strongly_oversold': regime_counts.get('STRONGLY OVERSOLD', 0),
            'oversold': regime_counts.get('OVERSOLD', 0),
            'neutral': regime_counts.get('NEUTRAL', 0),
            'overbought': regime_counts.get('OVERBOUGHT', 0),
            'strongly_overbought': regime_counts.get('STRONGLY OVERBOUGHT', 0),
            'current_regime': ts['Regime'].iloc[-1],
            'total_buy_signals': sum(ts['BuySignalBreadth']),
            'total_sell_signals': sum(ts['SellSignalBreadth']),
            'total_bull_div': ts['BullishDiv'].sum(),
            'total_bear_div': ts['BearishDiv'].sum(),
            'total_pivot_tops': ts['IsPivotTop'].sum(),
            'total_pivot_bottoms': ts['IsPivotBottom'].sum(),
        }
    
    def get_signal_performance(self):
        """Analyze historical signal performance."""
        ts = self.ts_data
        
        results = {}
        
        for period in [5, 10, 20]:
            buy_returns = []
            sell_returns = []
            
            for i in range(len(ts) - period):
                if ts['ConvictionScore'].iloc[i] < -40:  # Oversold
                    fwd = ts[f'FwdRet_{period}'].iloc[i]
                    if not pd.isna(fwd):
                        buy_returns.append(fwd)
                
                if ts['ConvictionScore'].iloc[i] > 40:  # Overbought
                    fwd = ts[f'FwdRet_{period}'].iloc[i]
                    if not pd.isna(fwd):
                        sell_returns.append(-fwd)  # Invert for sell
            
            results[period] = {
                'buy_avg': np.mean(buy_returns) if buy_returns else 0,
                'buy_hit': np.mean([r > 0 for r in buy_returns]) if buy_returns else 0,
                'buy_count': len(buy_returns),
                'sell_avg': np.mean(sell_returns) if sell_returns else 0,
                'sell_hit': np.mean([r > 0 for r in sell_returns]) if sell_returns else 0,
                'sell_count': len(sell_returns),
            }
        
        return results


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_google_sheet(sheet_url):
    try:
        import re
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            return None, "Invalid URL"
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)


def clean_data(df, target, features, date_col=None):
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
            data = data.sort_values(date_col)
        except: pass
    return data.reset_index(drop=True)


def update_chart_theme(fig, date_range=None):
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
        font=dict(family="Inter", color="#EAEAEA"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor="#2A2A2A", font_size=12)
    )
    return fig


def render_landing_page():
    """Renders the landing/home page content when no data is loaded."""
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card purple' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🎯 Fair Value Engine</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Uses advanced ensemble regression (Ridge, Huber, OLS) to determine the theoretical "Fair Value" of an asset based on macro drivers.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Methodology:</strong><br>
                • Linear & Robust Regression<br>
                • Feature Scaling<br>
                • Macro-Factor Attribution<br>
                • Dynamic Re-calibration
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>📉 Residual Oscillator</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Isolates the "noise" or premium/discount by subtracting Fair Value from Actual Price. This residual is strongly mean-reverting.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Key Metrics:</strong><br>
                • Z-Score Analysis<br>
                • Mean Reversion Speed<br>
                • Pivot Detection<br>
                • Volatility Normalization
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>📊 Breadth Conviction</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Analyzes the residual across 5 distinct timeframes (5d to 100d). When all timeframes agree, high-conviction signals are generated.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Lookback Periods:</strong><br>
                • Short: 5D, 10D<br>
                • Medium: 20D, 50D<br>
                • Long: 100D<br>
                • 0-100 Conviction Score
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analysis methodology section
    st.markdown("### 📐 Signal Logic")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("""
        <div class='signal-card fair' style='padding: 1.5rem;'>
            <h4 style='color: var(--primary-color); margin-bottom: 0.5rem;'>The Anchor (Fair Value)</h4>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.7;'>
                Prices deviate in the short term, but are anchored by fundamentals in the long term. 
                We calculate this anchor dynamically using a basket of predictors (Yields, Currency, PE, etc.).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("""
        <div class='signal-card undervalued' style='padding: 1.5rem; border-color: var(--text-muted); box-shadow: none;'>
            <h4 style='color: var(--text-primary); margin-bottom: 0.5rem;'>The Elastic (Residuals)</h4>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.7;'>
                The distance between Price and Value stretches like a rubber band. 
                We measure this tension. Extreme tension (±2σ) across multiple timeframes signals a likely snap-back.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal interpretation
    st.markdown("### 🎯 Signal Interpretation")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 0.75rem;'>🟢 Undervalued Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Conviction < -40</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Price is significantly below Fair Value across most lookbacks. High probability of reversion UP.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        st.markdown("""
        <div style='background: rgba(136, 136, 136, 0.1); border: 1px solid var(--neutral); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--neutral); margin-bottom: 0.75rem;'>⚪ Neutral Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Conviction -20 to +20</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Price is aligned with fundamentals. No strong statistical edge in either direction.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s3:
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger-red); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--danger-red); margin-bottom: 0.75rem;'>🔴 Overvalued Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Conviction > 40</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Price is significantly above Fair Value across most lookbacks. High probability of reversion DOWN.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <p style='color: var(--text-muted); line-height: 1.7;'>
            Use the <strong>Sidebar</strong> to load your dataset. You can either upload a CSV/Excel file or connect a Google Sheet.
            <br>Once loaded, select your <strong>Target</strong> (what to predict) and <strong>Predictors</strong> (macro factors) to generate the model.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown("""
    <div class="premium-header">
        <h1>Aarambh <span style="color:#FFC300;">Oscillator</span></h1>
        <div class="tagline">Multi-Lookback Residual Analysis for Market Top/Bottom Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar ---
    st.sidebar.markdown("### 📁 Data Setup")
    
    data_source = st.sidebar.radio("Source", ["📤 Upload", "📊 Google Sheets"], horizontal=True)
    
    df = None
    
    if data_source == "📤 Upload":
        uploaded_file = st.sidebar.file_uploader("CSV/Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error: {e}")
                return
    else:
        default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155#gid=1738251155"
        sheet_url = st.sidebar.text_input("Sheet URL", value=default_url)
        if st.sidebar.button("🔄 Load", type="primary"):
            with st.spinner("Loading..."):
                df, error = load_google_sheet(sheet_url)
                if error:
                    st.error(f"Failed: {error}")
                    return
                # Force reset of cache and oscillator when new data is loaded
                if 'oscillator' in st.session_state:
                    del st.session_state.oscillator
                if 'osc_cache' in st.session_state:
                    del st.session_state.osc_cache
                st.session_state['data'] = df
                # Use toast for ephemeral notification
                st.toast("Loaded!", icon="✅")
        if 'data' in st.session_state:
            df = st.session_state['data']
    
    if df is None:
        # Render the rich landing page instead of the simple guide box
        render_landing_page()
        return
    
    # --- Config ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return
    
    st.sidebar.markdown("---")
    
    default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
    default_preds = ["AD_RATIO", "COUNT", "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO", "CRR", "US02Y", "US10Y", "US30Y", "US_FED", "NIFTY50_DY", "NIFTY50_PB"]
    
    target_col = st.sidebar.selectbox("🎯 Target", numeric_cols, 
                                       index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)
    
    available = [c for c in numeric_cols if c != target_col]
    valid_defaults = [p for p in default_preds if p in available]
    
    feature_cols = st.sidebar.multiselect("📊 Predictors", available, default=valid_defaults or available[:3])
    
    if not feature_cols:
        st.info("👈 Select predictors")
        return
    
    st.sidebar.markdown("---")
    date_candidates = [c for c in all_cols if 'date' in c.lower()]
    date_col = st.sidebar.selectbox("📅 Date", ["None"] + all_cols,
                                     index=all_cols.index(date_candidates[0]) + 1 if date_candidates else 0)
    
    # Clean
    data = clean_data(df, target_col, feature_cols, date_col if date_col != "None" else None)
    
    if len(data) < 50:
        st.error("Need 50+ data points for multi-lookback analysis.")
        return
    
    X = data[feature_cols].values
    y = data[target_col].values
    
    # Fit
    cache_key = f"{target_col}_{'-'.join(sorted(feature_cols))}_{len(data)}"
    
    if 'osc_cache' not in st.session_state or st.session_state.osc_cache != cache_key:
        with st.spinner("Computing multi-lookback analysis..."):
            osc = ResidualOscillatorPro()
            osc.fit(X, y, feature_names=feature_cols)
            st.session_state.oscillator = osc
            st.session_state.osc_cache = cache_key
    
    osc = st.session_state.oscillator
    signal = osc.get_current_signal()
    model_stats = osc.get_model_stats()
    regime_stats = osc.get_regime_stats()
    ts = osc.ts_data.copy()
    
    # Date processing
    if date_col != "None" and date_col in data.columns:
        ts['Date'] = pd.to_datetime(data[date_col].values)
    else:
        ts['Date'] = np.arange(len(ts))

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY METRICS (UMA-style header)
    # ═══════════════════════════════════════════════════════════════════════
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Adjusted layout: Removed Observations, gave more space to Signal & Regime
    # Old: c1-c6 (equal)
    # New: 5 cols with weighted distribution
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
    
    with c1:
        os_color = "success" if signal['oversold_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {os_color}"><h4>Oversold</h4><h2>{signal["oversold_breadth"]:.0f}%</h2><div class="sub-metric">Lookbacks in Zone</div></div>', unsafe_allow_html=True)
    
    with c2:
        ob_color = "danger" if signal['overbought_breadth'] > 60 else "neutral"
        st.markdown(f'<div class="metric-card {ob_color}"><h4>Overbought</h4><h2>{signal["overbought_breadth"]:.0f}%</h2><div class="sub-metric">Lookbacks in Zone</div></div>', unsafe_allow_html=True)
    
    with c3:
        conv_color = "success" if signal['conviction_score'] < -40 else "danger" if signal['conviction_score'] > 40 else "neutral"
        st.markdown(f'<div class="metric-card {conv_color}"><h4>Conviction</h4><h2>{signal["conviction_score"]:+.0f}</h2><div class="sub-metric">-100 to +100</div></div>', unsafe_allow_html=True)
    
    with c4:
        sig_color = "success" if signal['signal'] == 'BUY' else "danger" if signal['signal'] == 'SELL' else "primary"
        st.markdown(f'<div class="metric-card {sig_color}"><h4>Signal</h4><h2>{signal["signal"]}</h2><div class="sub-metric">{signal["strength"]}</div></div>', unsafe_allow_html=True)
    
    with c5:
        reg_color = "success" if 'OVERSOLD' in signal['regime'] else "danger" if 'OVERBOUGHT' in signal['regime'] else "NEUTRAL"
        st.markdown(f'<div class="metric-card {reg_color}"><h4>Regime</h4><h2>{signal["regime"]}</h2><div class="sub-metric">Current State</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # GLOBAL TIMEFRAME FILTER
    # ═══════════════════════════════════════════════════════════════════════
    
    tf_col1, tf_col2 = st.columns([1, 6])
    with tf_col1:
        st.markdown("##### ⏱️ View Period")
    with tf_col2:
        time_filters = ["1M", "6M", "1Y", "2Y", "ALL"]
        # Default to 1Y (index 2)
        selected_tf = st.radio("Select Timeframe", time_filters, index=2, horizontal=True, label_visibility="collapsed")
    
    # Apply filtering
    ts_filtered = ts.copy()
    
    if selected_tf != "ALL":
        if date_col != "None" and not ts['Date'].empty and pd.api.types.is_datetime64_any_dtype(ts['Date']):
            max_date = ts['Date'].max()
            if selected_tf == "1M":
                cutoff = max_date - pd.DateOffset(months=1)
            elif selected_tf == "6M":
                cutoff = max_date - pd.DateOffset(months=6)
            elif selected_tf == "1Y":
                cutoff = max_date - pd.DateOffset(years=1)
            elif selected_tf == "2Y":
                cutoff = max_date - pd.DateOffset(years=2)
            ts_filtered = ts[ts['Date'] >= cutoff]
        else:
            # Fallback for index-based or non-datetime
            n = len(ts)
            if selected_tf == "1M": cutoff_idx = max(0, n - 21)
            elif selected_tf == "6M": cutoff_idx = max(0, n - 126)
            elif selected_tf == "1Y": cutoff_idx = max(0, n - 252)
            elif selected_tf == "2Y": cutoff_idx = max(0, n - 504)
            ts_filtered = ts.iloc[cutoff_idx:]
    
    x_axis = ts_filtered['Date']
    x_title = "Date" if date_col != "None" else "Index"
    
    # ═══════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "**📊 Signal Dashboard**",
        "**📈 Zone Trends**",
        "**📉 Signal Trends**",
        "**🎯 Regime Analysis**",
        "**📋 Data Table**"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: SIGNAL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("##### Current Signal Analysis")
            
            signal_class = 'undervalued' if signal['signal'] == 'BUY' else 'overvalued' if signal['signal'] == 'SELL' else 'fair'
            signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div class="label">MULTI-LOOKBACK SIGNAL</div>
                <div class="value">{signal_emoji} {signal['signal']}</div>
                <div class="subtext">{signal['strength']} Strength • {signal['confidence']} Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Conviction meter
            conv_pct = (signal['conviction_score'] + 100) / 2  # 0-100 scale
            conv_color = '#10b981' if signal['conviction_score'] < -20 else '#ef4444' if signal['conviction_score'] > 20 else '#FFC300'
            
            st.markdown(f"""
            <div class="conviction-meter">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #10b981; font-size: 0.75rem;">OVERSOLD</span>
                    <span style="color: #888; font-size: 0.75rem;">Conviction: {signal['conviction_score']:+.0f}</span>
                    <span style="color: #ef4444; font-size: 0.75rem;">OVERBOUGHT</span>
                </div>
                <div class="conviction-bar">
                    <div class="conviction-fill" style="width: {conv_pct}%; background: {conv_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Divergence alerts
            if signal['has_bullish_div']:
                st.markdown('<span class="status-badge buy">🔔 BULLISH DIVERGENCE DETECTED</span>', unsafe_allow_html=True)
            if signal['has_bearish_div']:
                st.markdown('<span class="status-badge sell">🔔 BEARISH DIVERGENCE DETECTED</span>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown("##### Lookback Breakdown")
            
            # Show each lookback's current status
            for lb in osc.LOOKBACKS:
                if lb not in osc.lookback_data:
                    continue
                z = osc.lookback_data[lb]['z_scores'][-1]
                zone = osc.lookback_data[lb]['zones'][-1]
                
                zone_color = '#10b981' if 'Under' in zone else '#ef4444' if 'Over' in zone else '#888'
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
                    <span style="color: #888;">{lb}-Day</span>
                    <span style="color: {zone_color}; font-weight: 600;">{zone} ({z:+.2f}σ)</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Price vs Fair Value chart
        st.markdown("##### Actual vs Fair Value")
        
        fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True, vertical_spacing=0.05)
        
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['Actual'], mode='lines', name='Actual', line=dict(color='#FFC300', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered['FairValue'], mode='lines', name='Fair Value', line=dict(color='#06b6d4', width=2, dash='dash')), row=1, col=1)
        
        # Residual
        colors = ['#10b981' if r < 0 else '#ef4444' for r in ts_filtered['Residual']]
        fig.add_trace(go.Bar(x=x_axis, y=ts_filtered['Residual'], name='Residual', marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_color="#FFC300", line_width=1, row=2, col=1)
        
        fig.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        update_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: ZONE TRENDS
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("##### Overbought / Oversold Breadth Over Time")
        st.markdown('<p style="color: #888;">Shows what % of lookback periods are in oversold/overbought zones</p>', unsafe_allow_html=True)
        
        fig_zones = go.Figure()
        
        fig_zones.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['OversoldBreadth'], mode='lines', name='Oversold %',
            fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.3)', line=dict(color='#10b981', width=2)
        ))
        
        fig_zones.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['OverboughtBreadth'], mode='lines', name='Overbought %',
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)', line=dict(color='#ef4444', width=2)
        ))
        
        # Add threshold lines
        fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.5)", annotation_text="60% Threshold")
        fig_zones.add_hline(y=80, line_dash="dash", line_color="rgba(255,195,0,0.8)", annotation_text="80% High Conviction")
        
        fig_zones.update_layout(title="Zone Breadth", height=400, xaxis_title=x_title, yaxis_title="% of Lookbacks",
                               yaxis=dict(range=[0, 105]))
        update_chart_theme(fig_zones)
        st.plotly_chart(fig_zones, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Extreme Zone Breadth")
        
        fig_extreme = go.Figure()
        
        fig_extreme.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ExtremeOversold'], mode='lines', name='Extreme Oversold %',
            fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.5)', line=dict(color='#10b981', width=2)
        ))
        
        fig_extreme.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ExtremeOverbought'], mode='lines', name='Extreme Overbought %',
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.5)', line=dict(color='#ef4444', width=2)
        ))
        
        fig_extreme.update_layout(title="Extreme Zones (±2σ)", height=300, xaxis_title=x_title, yaxis_title="%")
        update_chart_theme(fig_extreme)
        st.plotly_chart(fig_extreme, use_container_width=True)
        
        # Key insight
        if ts['OversoldBreadth'].iloc[-1] > 60:
            st.markdown(f"""
            <div class="guide-box success">
                <strong>🟢 High Oversold Breadth ({ts['OversoldBreadth'].iloc[-1]:.0f}%)</strong><br>
                Multiple lookback periods agree on oversold conditions — historically a bullish setup.
            </div>
            """, unsafe_allow_html=True)
        elif ts['OverboughtBreadth'].iloc[-1] > 60:
            st.markdown(f"""
            <div class="guide-box danger">
                <strong>🔴 High Overbought Breadth ({ts['OverboughtBreadth'].iloc[-1]:.0f}%)</strong><br>
                Multiple lookback periods agree on overbought conditions — historically a bearish setup.
            </div>
            """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: SIGNAL TRENDS
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("##### Buy / Sell Signal Generation Over Time")
        st.markdown('<p style="color: #888;">Count of lookback periods generating signals at each point</p>', unsafe_allow_html=True)
        
        fig_signals = go.Figure()
        
        fig_signals.add_trace(go.Bar(
            x=x_axis, y=ts_filtered['BuySignalBreadth'], name='Buy Signals',
            marker=dict(color='#10b981')
        ))
        
        fig_signals.add_trace(go.Bar(
            x=x_axis, y=-ts_filtered['SellSignalBreadth'], name='Sell Signals',
            marker=dict(color='#ef4444')
        ))
        
        fig_signals.update_layout(title="Signal Count by Period", height=350, xaxis_title=x_title, yaxis_title="Signal Count",
                                  barmode='relative')
        update_chart_theme(fig_signals)
        st.plotly_chart(fig_signals, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Divergence Signals Over Time")
        
        fig_div = go.Figure()
        
        fig_div.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['BullishDiv'].astype(int).cumsum(), mode='lines', name='Cumulative Bullish Div',
            line=dict(color='#FFC300', width=2)
        ))
        
        fig_div.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['BearishDiv'].astype(int).cumsum(), mode='lines', name='Cumulative Bearish Div',
            line=dict(color='#06b6d4', width=2)
        ))
        
        fig_div.update_layout(title="Cumulative Divergences", height=300, xaxis_title=x_title)
        update_chart_theme(fig_div)
        
        # Signal stats
        st.markdown("---")
        st.markdown("##### Signal Statistics")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card success"><h4>Total Buy Signals</h4><h2>{regime_stats["total_buy_signals"]:.0f}</h2><div class="sub-metric">Across Lookbacks</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card danger"><h4>Total Sell Signals</h4><h2>{regime_stats["total_sell_signals"]:.0f}</h2><div class="sub-metric">Across Lookbacks</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card primary"><h4>Bullish Divergences</h4><h2>{regime_stats["total_bull_div"]}</h2><div class="sub-metric">Detected</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card info"><h4>Bearish Divergences</h4><h2>{regime_stats["total_bear_div"]}</h2><div class="sub-metric">Detected</div></div>', unsafe_allow_html=True)
        
        # Performance analysis
        st.markdown("---")
        st.markdown("##### Signal Performance Analysis")
        
        perf = osc.get_signal_performance()
        
        perf_data = []
        for period in [5, 10, 20]:
            p = perf[period]
            perf_data.append({
                'Holding Period': f'{period} Days',
                'Buy Hit Rate': f"{p['buy_hit']*100:.1f}%" if p['buy_count'] > 0 else 'N/A',
                'Buy Avg Return': f"{p['buy_avg']:.2f}%" if p['buy_count'] > 0 else 'N/A',
                'Buy Count': p['buy_count'],
                'Sell Hit Rate': f"{p['sell_hit']*100:.1f}%" if p['sell_count'] > 0 else 'N/A',
                'Sell Avg Return': f"{p['sell_avg']:.2f}%" if p['sell_count'] > 0 else 'N/A',
                'Sell Count': p['sell_count'],
            })
        
        st.dataframe(pd.DataFrame(perf_data), width='stretch', hide_index=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4: REGIME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("##### Conviction Score Over Time")
        st.markdown('<p style="color: #888;">Negative = Oversold bias | Positive = Overbought bias</p>', unsafe_allow_html=True)
        
        fig_conv = go.Figure()
        
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ConvictionScore'].clip(lower=0),
            fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
            line=dict(width=0), showlegend=False
        ))
        
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ConvictionScore'].clip(upper=0),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.15)',
            line=dict(width=0), showlegend=False
        ))
        
        conv_colors = ['#10b981' if c < -40 else '#ef4444' if c > 40 else '#888' for c in ts_filtered['ConvictionScore']]
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered['ConvictionScore'], mode='lines+markers', name='Conviction',
            line=dict(color='#FFC300', width=2), marker=dict(size=4, color=conv_colors)
        ))
        
        fig_conv.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.5)")
        fig_conv.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.5)")
        fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        
        fig_conv.update_layout(title="Conviction Score", height=400, xaxis_title=x_title, yaxis_title="Score",
                               yaxis=dict(range=[-100, 100]))
        update_chart_theme(fig_conv)
        st.plotly_chart(fig_conv, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Regime Distribution")
            
            regime_data = {
                "Regime": ["🟢 Strongly Oversold", "🔵 Oversold", "⚪ Neutral", "🟠 Overbought", "🔴 Strongly Overbought"],
                "Count": [
                    regime_stats['strongly_oversold'],
                    regime_stats['oversold'],
                    regime_stats['neutral'],
                    regime_stats['overbought'],
                    regime_stats['strongly_overbought']
                ],
                "Pct": [
                    f"{regime_stats['strongly_oversold']/len(ts)*100:.1f}%",
                    f"{regime_stats['oversold']/len(ts)*100:.1f}%",
                    f"{regime_stats['neutral']/len(ts)*100:.1f}%",
                    f"{regime_stats['overbought']/len(ts)*100:.1f}%",
                    f"{regime_stats['strongly_overbought']/len(ts)*100:.1f}%"
                ]
            }
            st.dataframe(pd.DataFrame(regime_data), width='stretch', hide_index=True)
        
        with col2:
            st.markdown("##### Current Regime Analysis")
            
            curr_regime = signal['regime']
            regime_box_class = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else ""
            
            st.markdown(f"""
            <div class="guide-box {regime_box_class}">
                <strong>Current: {curr_regime}</strong><br><br>
                {'Multiple timeframes showing oversold conditions — historically a buying opportunity.' if 'OVERSOLD' in curr_regime else 
                 'Multiple timeframes showing overbought conditions — historically a selling opportunity.' if 'OVERBOUGHT' in curr_regime else
                 'No strong directional bias across timeframes.'}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Model Fit")
            st.markdown(f"R²: **{model_stats['r2']:.4f}** | RMSE: **{model_stats['rmse']:.4f}**")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5: DATA TABLE
    # ═══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown(f"##### Filtered Time Series Data ({len(ts_filtered)} observations)")
        
        # Select columns
        display_cols = ['Date', 'Actual', 'FairValue', 'Residual', 'AvgZ', 'OversoldBreadth', 'OverboughtBreadth', 
                       'ConvictionScore', 'Regime', 'BullishDiv', 'BearishDiv']
        
        display_ts = ts_filtered[display_cols].copy()
        display_ts['Residual'] = display_ts['Residual'].round(4)
        display_ts['AvgZ'] = display_ts['AvgZ'].round(3)
        display_ts['FairValue'] = display_ts['FairValue'].round(2)
        display_ts['OversoldBreadth'] = display_ts['OversoldBreadth'].round(1)
        display_ts['OverboughtBreadth'] = display_ts['OverboughtBreadth'].round(1)
        display_ts['ConvictionScore'] = display_ts['ConvictionScore'].round(1)
        display_ts['BullishDiv'] = display_ts['BullishDiv'].apply(lambda x: '🟢' if x else '')
        display_ts['BearishDiv'] = display_ts['BearishDiv'].apply(lambda x: '🔴' if x else '')
        
        display_ts.columns = ['Date', 'Actual', 'Fair Value', 'Residual', 'Avg Z', 'Oversold %', 'Overbought %',
                             'Conviction', 'Regime', 'Bull Div', 'Bear Div']
        
        st.dataframe(display_ts, width='stretch', hide_index=True, height=500)
        
        csv_data = ts.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full CSV", csv_data, f"aarambh_{target_col}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # Footer
    st.markdown(f'<div class="app-footer">© {datetime.now().year} Aarambh Oscillator | Arthagati Analytics</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

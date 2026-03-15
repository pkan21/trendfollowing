#!/usr/bin/env python3
"""
Trend Following Signal System — Streamlit App
==============================================
5-component composite trend signal for US equities.
  1. Time-Series Momentum (30%)  — Moskowitz/Ooi/Pedersen 12m-1m
  2. Vol-Adjusted Slope (25%)    — Clenow exponential regression / vol
  3. Moving Average Regime (20%) — Faber 10m SMA + 50/200 EMA
  4. ADX Trend Strength (15%)    — Wilder 14-period ADX
  5. 52-Week High Proximity (10%)— Behavioral anchoring
"""

import warnings
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Trend Signal System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    /* Dark finance theme */
    .stApp { background-color: #0d1117; }

    .signal-buy  { color: #3fb950; font-size: 2.4rem; font-weight: 800; letter-spacing: 0.05em; }
    .signal-hold { color: #d29922; font-size: 2.4rem; font-weight: 800; letter-spacing: 0.05em; }
    .signal-avoid{ color: #f85149; font-size: 2.4rem; font-weight: 800; letter-spacing: 0.05em; }

    .score-big {
        font-size: 3.6rem; font-weight: 900; line-height: 1;
        background: linear-gradient(135deg, #58a6ff, #7ee787);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .metric-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 16px 20px; text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 4px; }
    .metric-value { color: #f0f6fc; font-size: 1.25rem; font-weight: 700; }
    .metric-sub   { color: #8b949e; font-size: 0.75rem; margin-top: 2px; }

    .component-row {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 8px;
    }
    .comp-name   { color: #c9d1d9; font-weight: 700; font-size: 0.95rem; }
    .comp-weight { color: #8b949e; font-size: 0.8rem; }
    .comp-interp { color: #8b949e; font-size: 0.82rem; margin-top: 4px; }

    /* Progress bar colours */
    .bar-bg { background: #21262d; border-radius: 6px; height: 10px; width: 100%; }
    .bar-fill { height: 10px; border-radius: 6px; transition: width 0.4s ease; }
    .bar-green  { background: linear-gradient(90deg, #238636, #3fb950); }
    .bar-yellow { background: linear-gradient(90deg, #9e6a03, #d29922); }
    .bar-red    { background: linear-gradient(90deg, #b62324, #f85149); }

    div[data-testid="stSidebar"] { background-color: #161b22; }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] label { color: #c9d1d9; }

    .key-level-row { display: flex; justify-content: space-between; padding: 6px 0;
                     border-bottom: 1px solid #21262d; }
    .kl-label { color: #8b949e; font-size: 0.85rem; }
    .kl-value { color: #f0f6fc; font-weight: 700; font-size: 0.9rem; }
    .kl-arrow-up   { color: #3fb950; font-size: 0.85rem; }
    .kl-arrow-down { color: #f85149; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA ACQUISITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(ticker: str, years: int = 2) -> pd.DataFrame:
    end = dt.datetime.now()
    start = end - dt.timedelta(days=int(years * 365 + 365))
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) < 252:
            raise ValueError(f"Insufficient data for {ticker}: only {len(df)} days (need 252+)")
        return df
    except Exception as e_yf:
        try:
            from pandas_datareader import data as pdr
            df = pdr.get_data_yahoo(ticker, start=start, end=end)
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
            if "adj_close" in df.columns:
                ratio = df["adj_close"] / df["close"]
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c] * ratio
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            if len(df) < 252:
                raise ValueError(f"Insufficient data for {ticker}")
            return df
        except Exception as e_pdr:
            raise RuntimeError(
                f"All data sources failed for {ticker}.\nyfinance: {e_yf}\npandas_datareader: {e_pdr}"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL COMPONENTS (unchanged computation logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_ts_momentum(close: pd.Series) -> pd.Series:
    ret_12m = close / close.shift(252) - 1
    ret_1m = close / close.shift(21) - 1
    mom = ret_12m - ret_1m
    score = mom.rolling(252, min_periods=126).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1], kind="rank")
        if len(x.dropna()) > 10 else 50, raw=False)
    return score.clip(0, 100)


def _exp_regression_slope(log_prices: np.ndarray) -> float:
    n = len(log_prices)
    if n < 20:
        return np.nan
    x = np.arange(n)
    slope, _, r_value, _, _ = stats.linregress(x, log_prices)
    return (np.exp(slope * 252) - 1) * (r_value ** 2)


def compute_vol_adjusted_momentum(close: pd.Series) -> pd.Series:
    log_close = np.log(close)
    daily_ret = close.pct_change()
    window = 125
    slopes = log_close.rolling(window, min_periods=60).apply(
        lambda x: _exp_regression_slope(x.values), raw=False)
    ann_vol = daily_ret.rolling(window, min_periods=60).std() * np.sqrt(252)
    adj_slope = slopes / ann_vol.replace(0, np.nan)
    score = adj_slope.rolling(252, min_periods=126).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1], kind="rank")
        if len(x.dropna()) > 10 else 50, raw=False)
    return score.clip(0, 100)


def compute_ma_regime(close: pd.Series) -> tuple:
    sma_10m = close.rolling(210, min_periods=150).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    cond1 = (close > sma_10m).astype(float)
    cond2 = (ema_50 > ema_200).astype(float)
    raw_score = (cond1 + cond2) / 2 * 100
    return raw_score, sma_10m, ema_50, ema_200


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def compute_adx_score(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
    adx = compute_adx(high, low, close)
    score = ((adx - 10) / 40 * 100).clip(0, 100)
    discount = pd.Series(1.0, index=adx.index)
    discount = discount.where(adx >= 25, np.nan)
    discount = discount.fillna(
        adx.where((adx >= 20) & (adx < 25)).apply(
            lambda x: 0.5 + 0.5 * (x - 20) / 5 if pd.notna(x) else np.nan))
    discount = discount.fillna(0.5)
    return score, discount, adx


def compute_52w_high_proximity(close: pd.Series) -> tuple:
    high_52w = close.rolling(252, min_periods=126).max()
    ratio = close / high_52w
    score = ((ratio - 0.7) / 0.3 * 100).clip(0, 100)
    return score, high_52w


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    ts_mom = compute_ts_momentum(close)
    vol_mom = compute_vol_adjusted_momentum(close)
    ma_score, sma_10m, ema_50, ema_200 = compute_ma_regime(close)
    adx_score, adx_discount, adx_raw = compute_adx_score(high, low, close)
    hw_score, high_52w = compute_52w_high_proximity(close)

    composite = (
        0.30 * ts_mom * adx_discount +
        0.25 * vol_mom * adx_discount +
        0.20 * ma_score * (0.5 + 0.5 * adx_discount) +
        0.15 * adx_score +
        0.10 * hw_score * adx_discount
    )
    c_min = composite.rolling(504, min_periods=126).quantile(0.02)
    c_max = composite.rolling(504, min_periods=126).quantile(0.98)
    c_range = (c_max - c_min).replace(0, 1)
    composite_norm = ((composite - c_min) / c_range * 100).clip(0, 100)

    result = df.copy()
    result["ts_momentum"] = ts_mom
    result["vol_adj_momentum"] = vol_mom
    result["ma_regime"] = ma_score
    result["adx_score"] = adx_score
    result["adx_raw"] = adx_raw
    result["adx_discount"] = adx_discount
    result["hw_proximity"] = hw_score
    result["composite_raw"] = composite
    result["composite"] = composite_norm
    result["sma_10m"] = sma_10m
    result["ema_50"] = ema_50
    result["ema_200"] = ema_200
    result["high_52w"] = high_52w
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERPRETATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def interpret_signal(row: pd.Series) -> dict:
    score = row["composite"]
    adx = row["adx_raw"]

    if score >= 70:
        signal, css = "BUY", "signal-buy"
    elif score >= 40:
        signal, css = "HOLD", "signal-hold"
    else:
        signal, css = "AVOID", "signal-avoid"

    if adx >= 25 and (score >= 80 or score <= 20):
        confidence = "HIGH"
    elif adx >= 20 and (score >= 60 or score <= 30):
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    def i_mom(v):
        if v >= 70: return "Strong bullish momentum"
        if v >= 50: return "Moderate bullish momentum"
        if v >= 30: return "Weakening momentum"
        return "Bearish momentum"

    def i_vol(v):
        if v >= 70: return "Clean, strong uptrend (high R²)"
        if v >= 50: return "Moderate trend quality"
        if v >= 30: return "Noisy, weak trend"
        return "Deteriorating or negative trend"

    def i_ma(v):
        if v >= 75: return "Fully bullish — price > 10m SMA, golden cross"
        if v >= 25: return "Mixed — one MA bullish, one not"
        return "Fully bearish — below 10m SMA, death cross"

    def i_adx(v, r):
        regime = "TRENDING" if r >= 25 else ("MARGINAL" if r >= 20 else "CHOPPY")
        if v >= 60: return f"Strong trend ({regime}, ADX {r:.1f})"
        if v >= 30: return f"Moderate trend ({regime}, ADX {r:.1f})"
        return f"Weak / no trend ({regime}, ADX {r:.1f})"

    def i_52w(v, p, h):
        pct = p / h * 100 if h > 0 else 0
        if v >= 80: return f"Near 52w high ({pct:.1f}% of peak)"
        if v >= 50: return f"Moderate pullback ({pct:.1f}% of peak)"
        return f"Significant drawdown ({pct:.1f}% of peak)"

    components = [
        {"key": "ts_momentum",      "name": "Time-Series Momentum", "weight": "30%",
         "score": row["ts_momentum"], "interp": i_mom(row["ts_momentum"])},
        {"key": "vol_adj_momentum", "name": "Vol-Adjusted Slope",   "weight": "25%",
         "score": row["vol_adj_momentum"], "interp": i_vol(row["vol_adj_momentum"])},
        {"key": "ma_regime",        "name": "MA Regime",            "weight": "20%",
         "score": row["ma_regime"], "interp": i_ma(row["ma_regime"])},
        {"key": "adx_score",        "name": "ADX Trend Strength",   "weight": "15%",
         "score": row["adx_score"], "interp": i_adx(row["adx_score"], row["adx_raw"])},
        {"key": "hw_proximity",     "name": "52-Week High Prox.",   "weight": "10%",
         "score": row["hw_proximity"], "interp": i_52w(row["hw_proximity"], row["close"], row["high_52w"])},
    ]

    return dict(signal=signal, css=css, score=score, confidence=confidence,
                components=components, price=row["close"], ema_50=row["ema_50"],
                ema_200=row["ema_200"], sma_10m=row["sma_10m"],
                high_52w=row["high_52w"], adx=adx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOTLY CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    chart_days = min(380, len(df) - 1)
    cdf = df.iloc[-chart_days:].copy()
    dates = cdf.index

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    # ── Panel 1: Price + MAs ──
    # Background shading by MA regime
    regime = cdf["ma_regime"].copy()
    # Find contiguous regime blocks
    regime_cat = regime.apply(lambda v: "bull" if v >= 75 else ("mixed" if v >= 25 else "bear"))
    blocks = []
    prev = None
    block_start = dates[0]
    for i, (d, r) in enumerate(zip(dates, regime_cat)):
        if r != prev and prev is not None:
            blocks.append((block_start, dates[i - 1], prev))
            block_start = d
        prev = r
    blocks.append((block_start, dates[-1], prev))

    color_map = {"bull": "rgba(13,74,46,0.35)", "mixed": "rgba(61,56,0,0.25)",
                 "bear": "rgba(74,13,13,0.35)"}
    for s, e, regime_type in blocks:
        fig.add_vrect(x0=s, x1=e, fillcolor=color_map[regime_type],
                      line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=cdf["close"], name="Price",
                             line=dict(color="#58a6ff", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=cdf["ema_50"], name="50d EMA",
                             line=dict(color="#f0883e", width=1.2, dash="dash"),
                             opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=cdf["ema_200"], name="200d EMA",
                             line=dict(color="#f778ba", width=1.2, dash="dash"),
                             opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=cdf["sma_10m"], name="10m SMA",
                             line=dict(color="#7ee787", width=1.2, dash="dashdot"),
                             opacity=0.85), row=1, col=1)

    # ── Panel 2: Composite score ──
    score_s = cdf["composite"].dropna()
    if len(score_s) > 0:
        colors = ["#3fb950" if v >= 70 else ("#d29922" if v >= 40 else "#f85149")
                  for v in score_s]
        fig.add_trace(go.Bar(x=score_s.index, y=score_s.values, name="Score",
                             marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#3fb950", line_width=1,
                      annotation_text="BUY 70", annotation_font_color="#3fb950",
                      row=2, col=1)
        fig.add_hline(y=40, line_dash="dash", line_color="#f85149", line_width=1,
                      annotation_text="AVOID 40", annotation_font_color="#f85149",
                      row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        title=dict(text=f"{ticker} — Trend Signal", font=dict(size=18, color="#f0f6fc")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color="#c9d1d9"), bgcolor="rgba(0,0,0,0)"),
        height=620, margin=dict(l=50, r=20, t=60, b=30),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor="#21262d",
                     tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title_text="Score", row=2, col=1, gridcolor="#21262d",
                     range=[-5, 105])
    fig.update_xaxes(gridcolor="#21262d")
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bar_html(score: float) -> str:
    if score >= 70:
        cls = "bar-green"
    elif score >= 40:
        cls = "bar-yellow"
    else:
        cls = "bar-red"
    pct = max(1, min(100, score))
    return (f'<div class="bar-bg">'
            f'<div class="bar-fill {cls}" style="width:{pct:.0f}%"></div></div>')


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{sub_html}</div>')


def level_row(label: str, value: float, price: float, is_price: bool = True) -> str:
    if is_price:
        arrow = ('<span class="kl-arrow-up"> ▲ above</span>' if price > value
                 else '<span class="kl-arrow-down"> ▼ below</span>')
        return (f'<div class="key-level-row">'
                f'<span class="kl-label">{label}</span>'
                f'<span class="kl-value">${value:,.2f}{arrow}</span></div>')
    return (f'<div class="key-level-row">'
            f'<span class="kl-label">{label}</span>'
            f'<span class="kl-value">{value:.1f}</span></div>')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 📈 Trend Signal System")
        st.markdown('<p style="color:#8b949e;font-size:0.82rem;">'
                    '5-component composite trend signal for US equities</p>',
                    unsafe_allow_html=True)
        st.markdown("---")
        ticker = st.text_input("Ticker", value="NVDA", max_chars=10,
                               help="Any US equity ticker (e.g. NVDA, AAPL, SPY, TSLA)")
        ticker = ticker.upper().strip()

        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div style="color:#8b949e;font-size:0.75rem;line-height:1.5;">
        <b>Signal Thresholds</b><br>
        🟢 BUY — Score 70–100<br>
        🟡 HOLD — Score 40–69<br>
        🔴 AVOID — Score 0–39<br><br>
        <b>Components</b><br>
        1. 12m-1m Momentum (30%)<br>
        2. Vol-Adj Slope (25%)<br>
        3. MA Regime (20%)<br>
        4. ADX Strength (15%)<br>
        5. 52w High Prox. (10%)<br><br>
        Data: yfinance · 2yr+ history
        </div>
        """, unsafe_allow_html=True)

    # ── Main area ──
    if not analyze_btn and "result" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;margin-top:120px;">
            <div style="font-size:3rem;">📈</div>
            <div style="color:#c9d1d9;font-size:1.3rem;font-weight:700;margin-top:12px;">
                Trend Following Signal System</div>
            <div style="color:#8b949e;font-size:0.95rem;margin-top:8px;">
                Enter a US equity ticker in the sidebar and click <b>Analyze</b></div>
        </div>
        """, unsafe_allow_html=True)
        return

    if analyze_btn:
        with st.spinner(f"Fetching data & computing signals for {ticker}..."):
            try:
                df = fetch_data(ticker)
                result_df = compute_composite(df)
                latest = result_df.dropna(subset=["composite"]).iloc[-1]
                result = interpret_signal(latest)
                result["as_of"] = latest.name.strftime("%Y-%m-%d")
                st.session_state["result"] = result
                st.session_state["result_df"] = result_df
                st.session_state["ticker"] = ticker
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {e}")
                return

    # Pull from session
    result = st.session_state["result"]
    result_df = st.session_state["result_df"]
    tkr = st.session_state["ticker"]

    # ── Header row: signal + score + confidence ──
    h1, h2, h3 = st.columns([1, 1, 1])
    with h1:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-label">Signal</div>'
                    f'<div class="{result["css"]}">{result["signal"]}</div></div>',
                    unsafe_allow_html=True)
    with h2:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-label">Composite Score</div>'
                    f'<div class="score-big">{result["score"]:.1f}</div>'
                    f'<div class="metric-sub">out of 100</div></div>',
                    unsafe_allow_html=True)
    with h3:
        conf_color = {"HIGH": "#3fb950", "MEDIUM": "#d29922", "LOW": "#8b949e"}[result["confidence"]]
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-label">6-Month Confidence</div>'
                    f'<div class="metric-value" style="color:{conf_color};font-size:1.6rem;">'
                    f'{result["confidence"]}</div>'
                    f'<div class="metric-sub">{tkr} · {result["as_of"]}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Chart ──
    fig = build_chart(tkr, result_df)
    st.plotly_chart(fig, use_container_width=True)

    # ── Two-column layout: components + key levels ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Signal Breakdown")
        for comp in result["components"]:
            s = comp["score"]
            st.markdown(
                f'<div class="component-row">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span class="comp-name">{comp["name"]}</span>'
                f'<span class="comp-weight">{comp["weight"]} · {s:.1f}/100</span></div>'
                f'{bar_html(s)}'
                f'<div class="comp-interp">{comp["interp"]}</div></div>',
                unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Key Levels")
        price = result["price"]
        st.markdown(
            f'<div class="metric-card" style="margin-bottom:12px;">'
            f'<div class="metric-label">Current Price</div>'
            f'<div class="metric-value" style="font-size:1.8rem;">${price:,.2f}</div></div>',
            unsafe_allow_html=True)

        levels_html = (
            level_row("50-day EMA", result["ema_50"], price) +
            level_row("200-day EMA", result["ema_200"], price) +
            level_row("10-month SMA", result["sma_10m"], price) +
            level_row("52-week High", result["high_52w"], price) +
            level_row("ADX (14)", result["adx"], price, is_price=False)
        )
        st.markdown(f'<div class="metric-card">{levels_html}</div>', unsafe_allow_html=True)

        # Distance to 52w high
        dist = (price / result["high_52w"] - 1) * 100
        dist_str = f"{dist:+.1f}% from 52w high"
        st.markdown(f'<div class="metric-card" style="margin-top:12px;">'
                    f'<div class="metric-label">Distance to Peak</div>'
                    f'<div class="metric-value">{dist_str}</div></div>',
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()

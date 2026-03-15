#!/usr/bin/env python3
"""
Trend Following Signal System v3 — Streamlit + Polygon.io
==========================================================
7-component composite trend signal for US equities.
Based on: Clenow, Antonacci, Moskowitz / AQR, Carver.
Data: Polygon.io REST API  (POLYGON_API_KEY env var).

Components
----------
1. 12m Absolute Momentum (25%)  — Moskowitz / Antonacci 12m-skip-1m vs T-bill
2. Clenow Adjusted Slope (30%)  — 90-day exp-regression slope × R²
3. MA Regime Triple-Check (20%) — price>100 SMA, price>200 SMA, 50 EMA>200 EMA
4. ADX Trend Strength (10%)     — Wilder 14-period ADX, choppy discount
5. 52-Week High Proximity (15%) — current / 52w high × 100

Overlays
--------
6. Market Regime Filter — SPY > 200d SMA gate (caps score at 35 if risk-off)
7. Gap Filter — any 15 %+ single-day gap in 90 days → −20 pt penalty

Score 70-100 → BUY | 45-69 → HOLD | 0-44 → AVOID
"""

from __future__ import annotations

import os
import warnings
import datetime as dt
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from polygon import RESTClient

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Trend Signal System",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THEME CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CSS = """
<style>
/* ── Base ── */
.stApp { background: #0a0e14; }
section[data-testid="stSidebar"] { background: #12161e; }

/* ── Signal badges ── */
.signal-buy   { color:#3fb950; font-size:2.6rem; font-weight:900; letter-spacing:.06em; }
.signal-hold  { color:#d29922; font-size:2.6rem; font-weight:900; letter-spacing:.06em; }
.signal-avoid { color:#f85149; font-size:2.6rem; font-weight:900; letter-spacing:.06em; }

/* ── Score big number ── */
.score-big {
    font-size:3.8rem; font-weight:900; line-height:1;
    background:linear-gradient(135deg,#58a6ff,#7ee787);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── Cards ── */
.card {
    background:#12161e; border:1px solid #1e2531; border-radius:12px;
    padding:16px 20px; text-align:center;
}
.card-label {
    color:#6e7a8a; font-size:.72rem; text-transform:uppercase;
    letter-spacing:.09em; margin-bottom:4px;
}
.card-value  { color:#e6edf3; font-size:1.25rem; font-weight:700; }
.card-sub    { color:#6e7a8a; font-size:.73rem; margin-top:2px; }

/* ── Component rows ── */
.comp-row {
    background:#12161e; border:1px solid #1e2531; border-radius:10px;
    padding:14px 18px; margin-bottom:8px;
}
.comp-name   { color:#c9d1d9; font-weight:700; font-size:.92rem; }
.comp-wt     { color:#6e7a8a; font-size:.78rem; }
.comp-raw    { color:#58a6ff; font-size:.78rem; font-family:monospace; }
.comp-interp { color:#6e7a8a; font-size:.80rem; margin-top:4px; }

/* ── Progress bars ── */
.bar-bg   { background:#1a1f2b; border-radius:6px; height:10px; width:100%; margin-top:6px; }
.bar-fill { height:10px; border-radius:6px; }
.bar-g  { background:linear-gradient(90deg,#238636,#3fb950); }
.bar-y  { background:linear-gradient(90deg,#9e6a03,#d29922); }
.bar-r  { background:linear-gradient(90deg,#b62324,#f85149); }

/* ── Key-levels table ── */
.kl-row {
    display:flex; justify-content:space-between; padding:6px 0;
    border-bottom:1px solid #1a1f2b;
}
.kl-label { color:#6e7a8a; font-size:.84rem; }
.kl-val   { color:#e6edf3; font-weight:700; font-size:.88rem; }
.kl-up    { color:#3fb950; font-size:.82rem; }
.kl-dn    { color:#f85149; font-size:.82rem; }

/* ── Banners ── */
.banner-red {
    background:#2d1215; border:1px solid #f8514980; border-radius:10px;
    padding:12px 18px; color:#f85149; font-weight:700; font-size:.95rem;
    margin-bottom:12px;
}
.banner-amber {
    background:#2d2610; border:1px solid #d2992280; border-radius:10px;
    padding:12px 18px; color:#d29922; font-weight:700; font-size:.95rem;
    margin-bottom:12px;
}

/* ── Rationale box ── */
.rationale {
    background:#12161e; border:1px solid #1e2531; border-radius:10px;
    padding:18px 22px; color:#a0aab5; font-size:.88rem; line-height:1.65;
    margin-top:16px;
}
.rationale b { color:#c9d1d9; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POLYGON DATA LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _polygon_client() -> RESTClient:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError(
            "POLYGON_API_KEY is not set. "
            "Run: export POLYGON_API_KEY=your_key"
        )
    return RESTClient(api_key=key)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, years: int = 2) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV from Polygon /v2/aggs/ticker/ endpoint."""
    ticker = ticker.upper().strip()
    client = _polygon_client()
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365 + 400))  # extra buffer

    aggs = list(client.list_aggs(
        ticker=ticker, multiplier=1, timespan="day",
        from_=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"),
        adjusted=True, sort="asc", limit=50_000,
    ))
    if not aggs:
        raise ValueError(f"No data returned for '{ticker}'. Check the ticker symbol.")

    rows = [{
        "date": pd.Timestamp(a.timestamp, unit="ms"),
        "open": a.open, "high": a.high, "low": a.low,
        "close": a.close, "volume": a.volume,
    } for a in aggs]

    df = (pd.DataFrame(rows)
          .set_index("date").sort_index()
          [["open", "high", "low", "close", "volume"]]
          .dropna())
    if len(df) < 252:
        raise ValueError(f"Only {len(df)} trading days for {ticker} (need 252+).")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 1 — 12m ABSOLUTE MOMENTUM (25 %)
# Moskowitz / Antonacci: return t-252 → t-21 vs T-bill proxy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TBILL_ANNUAL = 0.05                        # 5 % annual proxy
TBILL_12M    = TBILL_ANNUAL                # 12-month T-bill return


def calc_abs_momentum(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (score 0|100 series, raw 12m-1m return series)."""
    price_12m_ago = close.shift(252)
    price_1m_ago  = close.shift(21)
    ret = (price_1m_ago / price_12m_ago) - 1   # skip last month
    score = (ret > TBILL_12M).astype(float) * 100
    return score, ret


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 2 — CLENOW ADJUSTED SLOPE (30 %)
# 90-day exp-regression slope × R², z-scored over 252 days
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _slope_r2(log_prices: np.ndarray) -> tuple[float, float]:
    n = len(log_prices)
    if n < 30:
        return np.nan, np.nan
    x = np.arange(n)
    slope, _, r_value, _, _ = stats.linregress(x, log_prices)
    ann_slope = np.exp(slope * 252) - 1          # annualised
    r2 = r_value ** 2
    return ann_slope * r2, r2                     # adjusted slope, R²


def calc_clenow_slope(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (score 0-100, raw adjusted_slope series, R² series)."""
    log_c = np.log(close)
    window = 90

    raw_vals, r2_vals = [], []
    for i in range(len(log_c)):
        if i < window - 1:
            raw_vals.append(np.nan); r2_vals.append(np.nan)
        else:
            chunk = log_c.iloc[i - window + 1: i + 1].values
            adj, r2 = _slope_r2(chunk)
            raw_vals.append(adj); r2_vals.append(r2)

    raw = pd.Series(raw_vals, index=close.index, dtype=float)
    r2  = pd.Series(r2_vals, index=close.index, dtype=float)

    # Z-score over trailing 252 days → percentile → 0-100
    score = raw.rolling(252, min_periods=90).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1], kind="rank")
        if len(x.dropna()) > 20 else 50, raw=False,
    ).clip(0, 100)

    return score, raw, r2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 3 — MA REGIME TRIPLE CHECK (20 %)
# price>100 SMA, price>200 SMA, 50 EMA>200 EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_ma_regime(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (score, sma100, sma200, ema50)."""
    sma100 = close.rolling(100, min_periods=80).mean()
    sma200 = close.rolling(200, min_periods=150).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    cond_a = (close > sma100).astype(float)
    cond_b = (close > sma200).astype(float)
    cond_c = (ema50 > ema200).astype(float)
    score  = ((cond_a + cond_b + cond_c) * 33.33).clip(0, 100)
    return score, sma100, sma200, ema50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 4 — ADX TREND STRENGTH (10 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _wilder_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series,
             close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (score 0-100, discount factor, raw ADX)."""
    adx = _wilder_adx(high, low, close)

    score = pd.Series(25.0, index=adx.index)
    score = score.where(adx < 20, np.nan)
    score = score.fillna(adx.where((adx >= 20) & (adx < 25)).apply(lambda x: 50.0 if pd.notna(x) else np.nan))
    score = score.fillna(adx.where((adx >= 25) & (adx < 30)).apply(lambda x: 75.0 if pd.notna(x) else np.nan))
    score = score.fillna(adx.where(adx >= 30).apply(lambda x: 100.0 if pd.notna(x) else np.nan))
    score = score.fillna(25.0)

    discount = pd.Series(1.0, index=adx.index)
    discount = discount.where(adx >= 20, 0.6)

    return score, discount, adx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 5 — 52-WEEK HIGH PROXIMITY (15 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_52w_high(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (score 0-100, 52w high, 52w low)."""
    h52 = close.rolling(252, min_periods=126).max()
    l52 = close.rolling(252, min_periods=126).min()
    score = (close / h52 * 100).clip(0, 100)
    return score, h52, l52


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERLAY 6 — MARKET REGIME (SPY GATE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_market_regime() -> tuple[bool, float, float]:
    """Return (is_risk_on, spy_price, spy_200sma)."""
    spy = fetch_ohlcv("SPY")
    sma200 = spy["close"].rolling(200, min_periods=150).mean()
    last_close = spy["close"].iloc[-1]
    last_sma   = sma200.iloc[-1]
    return (last_close > last_sma), last_close, last_sma


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERLAY 7 — GAP FILTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_gap(df: pd.DataFrame) -> tuple[bool, float | None, str | None]:
    """Return (has_gap, worst_gap_pct, gap_date_str)."""
    recent = df.iloc[-90:]
    gap = (recent["open"] / recent["close"].shift(1) - 1).abs()
    mask = gap > 0.15
    if mask.any():
        worst_idx = gap[mask].idxmax()
        return True, float(gap.loc[worst_idx] * 100), worst_idx.strftime("%Y-%m-%d")
    return False, None, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ATR(20)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 20) -> pd.Series:
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPOSITE ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_all(df: pd.DataFrame, risk_on: bool, has_gap: bool) -> pd.DataFrame:
    """Attach all components + composite to the dataframe."""
    c, h, lo = df["close"], df["high"], df["low"]

    # Components
    abs_mom_score, abs_mom_raw         = calc_abs_momentum(c)
    clenow_score, clenow_raw, clenow_r2 = calc_clenow_slope(c)
    ma_score, sma100, sma200, ema50    = calc_ma_regime(c)
    adx_score, adx_discount, adx_raw   = calc_adx(h, lo, c)
    hw_score, h52, l52                 = calc_52w_high(c)
    atr20                              = calc_atr(h, lo, c)

    # Weighted composite (before overlays)
    raw_comp = (
        0.25 * abs_mom_score
      + 0.30 * clenow_score
      + 0.20 * ma_score
      + 0.10 * adx_score
      + 0.15 * hw_score
    )

    # ADX choppy discount: when ADX < 20, multiply all *other* scores by 0.6
    # Recompute using discounted inputs
    comp = (
        0.25 * abs_mom_score * adx_discount
      + 0.30 * clenow_score * adx_discount
      + 0.20 * ma_score * adx_discount
      + 0.10 * adx_score               # ADX doesn't discount itself
      + 0.15 * hw_score * adx_discount
    )

    # Gap penalty (−20 pts)
    if has_gap:
        comp = (comp - 20).clip(0, 100)

    # Market regime cap
    if not risk_on:
        comp = comp.clip(upper=35)

    comp = comp.clip(0, 100)

    r = df.copy()
    r["abs_mom_score"]   = abs_mom_score
    r["abs_mom_raw"]     = abs_mom_raw
    r["clenow_score"]    = clenow_score
    r["clenow_raw"]      = clenow_raw
    r["clenow_r2"]       = clenow_r2
    r["ma_score"]        = ma_score
    r["adx_score"]       = adx_score
    r["adx_raw"]         = adx_raw
    r["adx_discount"]    = adx_discount
    r["hw_score"]        = hw_score
    r["composite"]       = comp
    r["composite_raw"]   = raw_comp
    r["sma100"]          = sma100
    r["sma200"]          = sma200
    r["ema50"]           = ema50
    r["high_52w"]        = h52
    r["low_52w"]         = l52
    r["atr20"]           = atr20
    return r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERPRET LATEST ROW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _interp_abs(score, raw):
    if pd.isna(raw):
        return "Insufficient history", ""
    r_str = f"{raw * 100:+.1f}%"
    if score >= 100:
        return f"12m-1m return ({r_str}) beats T-bill ({TBILL_ANNUAL*100:.0f}%)", r_str
    return f"12m-1m return ({r_str}) below T-bill hurdle", r_str

def _interp_clenow(score, raw, r2):
    if pd.isna(raw):
        return "Insufficient history", ""
    r_str = f"slope\u00d7R\u00b2={raw:.3f}"
    if score >= 70: return f"Clean, strong trend ({r_str}, R\u00b2={r2:.2f})", r_str
    if score >= 40: return f"Moderate trend quality ({r_str})", r_str
    return f"Weak or deteriorating trend ({r_str})", r_str

def _interp_ma(score):
    n = int(round(score / 33.33))
    if n == 3: return "All 3 MA conditions bullish", f"{n}/3"
    if n == 2: return "2 of 3 MA conditions bullish", f"{n}/3"
    if n == 1: return "Only 1 of 3 MA conditions met", f"{n}/3"
    return "No MA conditions met \u2014 fully bearish", "0/3"

def _interp_adx(score, raw):
    r_str = f"ADX={raw:.1f}"
    if raw >= 30: return f"Strong directional trend ({r_str})", r_str
    if raw >= 25: return f"Moderate trend ({r_str})", r_str
    if raw >= 20: return f"Marginal trend ({r_str})", r_str
    return f"Choppy / no trend ({r_str}) \u2014 scores discounted 40%", r_str

def _interp_52w(score, price, h52):
    pct = price / h52 * 100 if h52 > 0 else 0
    if pct >= 95: return f"Near 52w high ({pct:.1f}%)", f"{pct:.1f}%"
    if pct >= 80: return f"Healthy pullback from peak ({pct:.1f}%)", f"{pct:.1f}%"
    return f"Deep drawdown from peak ({pct:.1f}%)", f"{pct:.1f}%"


def interpret(row: pd.Series, risk_on: bool, has_gap: bool,
              gap_pct: float | None, gap_date: str | None) -> dict:
    score = row["composite"]

    if score >= 70:   sig, css = "BUY",   "signal-buy"
    elif score >= 45: sig, css = "HOLD",  "signal-hold"
    else:             sig, css = "AVOID", "signal-avoid"

    # Confidence
    adx = row["adx_raw"]
    if adx >= 25 and (score >= 80 or score <= 15): conf = "HIGH"
    elif adx >= 20 and (score >= 55 or score <= 30): conf = "MEDIUM"
    else: conf = "LOW"

    # Component detail list
    abs_interp, abs_raw_s = _interp_abs(row["abs_mom_score"], row["abs_mom_raw"])
    cle_interp, cle_raw_s = _interp_clenow(row["clenow_score"], row["clenow_raw"], row["clenow_r2"])
    ma_interp,  ma_raw_s  = _interp_ma(row["ma_score"])
    adx_interp, adx_raw_s = _interp_adx(row["adx_score"], row["adx_raw"])
    hw_interp,  hw_raw_s  = _interp_52w(row["hw_score"], row["close"], row["high_52w"])

    comps = [
        dict(name="12m Absolute Momentum", wt="25%", score=row["abs_mom_score"],
             raw=abs_raw_s, interp=abs_interp),
        dict(name="Clenow Adj. Slope",     wt="30%", score=row["clenow_score"],
             raw=cle_raw_s, interp=cle_interp),
        dict(name="MA Regime (Triple)",     wt="20%", score=row["ma_score"],
             raw=ma_raw_s,  interp=ma_interp),
        dict(name="ADX Trend Strength",     wt="10%", score=row["adx_score"],
             raw=adx_raw_s, interp=adx_interp),
        dict(name="52-Week High Prox.",     wt="15%", score=row["hw_score"],
             raw=hw_raw_s,  interp=hw_interp),
    ]

    # Overlays as pseudo-components for display
    comps.append(dict(
        name="Market Regime (SPY)",
        wt="gate",
        score=100.0 if risk_on else 0.0,
        raw="RISK ON" if risk_on else "RISK OFF",
        interp=("SPY above 200d SMA \u2014 no suppression"
                if risk_on else "SPY below 200d SMA \u2014 score capped at 35"),
    ))
    gap_score = 0.0 if has_gap else 100.0
    gap_raw_str = f"{gap_pct:.1f}% on {gap_date}" if has_gap else "None"
    comps.append(dict(
        name="Gap Filter (90d)",
        wt="penalty",
        score=gap_score,
        raw=gap_raw_str,
        interp=("\u221220 pt penalty applied \u2014 event-driven gap detected"
                if has_gap else "No large gaps detected"),
    ))

    return dict(
        signal=sig, css=css, score=score, confidence=conf,
        components=comps, risk_on=risk_on,
        has_gap=has_gap, gap_pct=gap_pct, gap_date=gap_date,
        price=row["close"], sma100=row["sma100"], sma200=row["sma200"],
        ema50=row["ema50"], high_52w=row["high_52w"], low_52w=row["low_52w"],
        atr20=row["atr20"], adx=adx,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL RATIONALE GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_rationale(ticker: str, r: dict) -> str:
    """One-paragraph plain-English explanation."""
    parts = []

    # Market regime
    if not r["risk_on"]:
        parts.append(
            f"The S&P 500 is trading below its 200-day SMA, placing the broad "
            f"market in a risk-off regime. Under this condition the composite "
            f"score for {ticker} is capped at 35 regardless of individual strength."
        )

    # Dominant signals
    comps = {c["name"]: c for c in r["components"][:5]}

    abs_s = comps["12m Absolute Momentum"]["score"]
    cle_s = comps["Clenow Adj. Slope"]["score"]
    ma_s  = comps["MA Regime (Triple)"]["score"]
    adx_s = comps["ADX Trend Strength"]["score"]
    hw_s  = comps["52-Week High Prox."]["score"]

    # Momentum narrative
    if abs_s >= 100 and cle_s >= 60:
        parts.append(
            f"{ticker}'s 12-month return (skipping the last month) exceeds the "
            f"T-bill hurdle, confirming positive absolute momentum. The 90-day "
            f"Clenow regression slope is in the {cle_s:.0f}th percentile of its "
            f"trailing year, indicating a clean uptrend."
        )
    elif abs_s >= 100:
        parts.append(
            f"{ticker} clears the absolute momentum hurdle, but the Clenow slope "
            f"quality sits at only {cle_s:.0f}/100 \u2014 the trend is present but noisy."
        )
    elif cle_s >= 60:
        parts.append(
            f"The 90-day trend slope is solid (score {cle_s:.0f}), but the "
            f"12-month return failed to clear the T-bill hurdle, meaning absolute "
            f"momentum is negative."
        )
    else:
        parts.append(
            f"Both the 12-month absolute momentum and the 90-day trend slope are "
            f"weak, suggesting {ticker} lacks directional conviction."
        )

    # MA regime
    n_ma = int(round(ma_s / 33.33))
    if n_ma == 3:
        parts.append("All three moving-average conditions are bullish.")
    elif n_ma >= 2:
        parts.append(f"The MA regime is mixed ({n_ma}/3 conditions met).")
    else:
        parts.append("The moving-average regime is bearish with at most one condition met.")

    # ADX
    adx_val = r["adx"]
    if adx_val < 20:
        parts.append(
            f"ADX is {adx_val:.1f}, signaling a choppy, range-bound market. "
            f"All component scores have been discounted by 40% to reflect low "
            f"directional reliability."
        )
    elif adx_val >= 30:
        parts.append(f"ADX at {adx_val:.1f} confirms a strong directional trend.")

    # 52w proximity
    hw_pct = r["price"] / r["high_52w"] * 100 if r["high_52w"] > 0 else 0
    if hw_pct >= 95:
        parts.append(f"Price is within 5% of its 52-week high ({hw_pct:.1f}%).")
    elif hw_pct < 80:
        parts.append(f"Price has pulled back significantly from its peak ({hw_pct:.1f}% of 52w high).")

    # Gap
    if r["has_gap"]:
        parts.append(
            f"A {r['gap_pct']:.1f}% single-day gap was detected on "
            f"{r['gap_date']}. This suggests event-driven rather than "
            f"sustainable momentum, so the score has been penalized by 20 points."
        )

    # Final verdict
    parts.append(
        f"The composite score is {r['score']:.1f}/100, yielding a "
        f"<b>{r['signal']}</b> signal with {r['confidence']} confidence "
        f"over a 6-month horizon."
    )

    return " ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOTLY CHARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RANGES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}


def _slice(df: pd.DataFrame, days: int) -> pd.DataFrame:
    n = min(days, len(df) - 1)
    return df.iloc[-n:].copy()


def chart_price(tkr: str, df: pd.DataFrame, days: int) -> go.Figure:
    cdf = _slice(df, days)
    d = cdf.index

    fig = go.Figure()

    # Shade red/green based on price vs 200d SMA
    sma200 = cdf["sma200"]
    above = (cdf["close"] > sma200)
    cat = above.map({True: "bull", False: "bear"})
    cmap = {"bull": "rgba(13,74,46,0.30)", "bear": "rgba(74,13,13,0.30)"}
    prev, bs = None, d[0]
    blocks = []
    for i, (dt_, r_) in enumerate(zip(d, cat)):
        if r_ != prev and prev is not None:
            blocks.append((bs, d[i - 1], prev))
            bs = dt_
        prev = r_
    blocks.append((bs, d[-1], prev))
    for s, e, rt in blocks:
        fig.add_vrect(x0=s, x1=e, fillcolor=cmap.get(rt, "rgba(0,0,0,0)"), line_width=0)

    fig.add_trace(go.Scatter(x=d, y=cdf["close"], name="Price",
                             line=dict(color="#58a6ff", width=2.2),
                             hovertemplate="$%{y:,.2f}<extra>Price</extra>"))
    fig.add_trace(go.Scatter(x=d, y=cdf["sma100"], name="100d SMA",
                             line=dict(color="#f0883e", width=1.2, dash="dash"), opacity=.85,
                             hovertemplate="$%{y:,.2f}<extra>100d SMA</extra>"))
    fig.add_trace(go.Scatter(x=d, y=cdf["sma200"], name="200d SMA",
                             line=dict(color="#f778ba", width=1.2, dash="dash"), opacity=.85,
                             hovertemplate="$%{y:,.2f}<extra>200d SMA</extra>"))
    fig.add_trace(go.Scatter(x=d, y=cdf["ema50"], name="50d EMA",
                             line=dict(color="#7ee787", width=1.2, dash="dashdot"), opacity=.85,
                             hovertemplate="$%{y:,.2f}<extra>50d EMA</extra>"))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        title=dict(text=f"{tkr} \u2014 Price & Moving Averages", font=dict(size=15, color="#e6edf3")),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#a0aab5"),
                    bgcolor="rgba(0,0,0,0)"),
        height=400, margin=dict(l=50, r=20, t=50, b=25), hovermode="x unified",
        yaxis=dict(gridcolor="#161b22", tickprefix="$", tickformat=",.0f"),
        xaxis=dict(gridcolor="#161b22"),
    )
    return fig


def chart_score(tkr: str, df: pd.DataFrame, days: int) -> go.Figure:
    cdf = _slice(df, days)
    s = cdf["composite"].dropna()
    fig = go.Figure()
    if len(s):
        colors = ["#3fb950" if v >= 70 else ("#d29922" if v >= 45 else "#f85149") for v in s]
        fig.add_trace(go.Bar(x=s.index, y=s.values, marker_color=colors, showlegend=False,
                             hovertemplate="%{y:.1f}<extra>Score</extra>"))
        fig.add_hline(y=70, line_dash="dash", line_color="#3fb950", line_width=1,
                      annotation_text="BUY 70", annotation_font_color="#3fb950")
        fig.add_hline(y=45, line_dash="dash", line_color="#f85149", line_width=1,
                      annotation_text="AVOID 45", annotation_font_color="#f85149")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        title=dict(text=f"{tkr} \u2014 Composite Trend Score", font=dict(size=15, color="#e6edf3")),
        height=260, margin=dict(l=50, r=20, t=50, b=25), hovermode="x unified",
        yaxis=dict(gridcolor="#161b22", range=[-5, 105], title="Score"),
        xaxis=dict(gridcolor="#161b22"),
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bar(score: float) -> str:
    cls = "bar-g" if score >= 70 else ("bar-y" if score >= 45 else "bar-r")
    pct = max(1, min(100, score))
    return f'<div class="bar-bg"><div class="bar-fill {cls}" style="width:{pct:.0f}%"></div></div>'


def _kl(label: str, val: float, price: float, is_price: bool = True) -> str:
    if is_price:
        arr = ('<span class="kl-up"> \u25b2</span>' if price > val
               else '<span class="kl-dn"> \u25bc</span>')
        return (f'<div class="kl-row"><span class="kl-label">{label}</span>'
                f'<span class="kl-val">${val:,.2f}{arr}</span></div>')
    return (f'<div class="kl-row"><span class="kl-label">{label}</span>'
            f'<span class="kl-val">{val:,.2f}</span></div>')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Top bar: ticker input + button ──
    top1, top2, top3 = st.columns([2, 1, 4])
    with top1:
        ticker = st.text_input(
            "Ticker", value="NVDA", max_chars=10, label_visibility="collapsed",
            placeholder="Enter US ticker (e.g. NVDA, AAPL, SPY)",
        ).upper().strip()
    with top2:
        go_btn = st.button("\U0001f50d  Analyze", type="primary", use_container_width=True)

    # ── Landing ──
    if not go_btn and "res" not in st.session_state:
        st.markdown(
            '<div style="text-align:center;margin-top:80px;">'
            '<div style="font-size:3rem;">\U0001f4c8</div>'
            '<div style="color:#c9d1d9;font-size:1.3rem;font-weight:700;margin-top:12px;">'
            'Trend Following Signal System</div>'
            '<div style="color:#6e7a8a;font-size:.92rem;margin-top:8px;">'
            'Clenow \u00b7 Antonacci \u00b7 Moskowitz / AQR \u00b7 Carver<br>'
            '7 components \u00b7 Polygon.io data \u00b7 Enter a ticker above</div></div>',
            unsafe_allow_html=True)
        return

    # ── Compute ──
    if go_btn:
        with st.spinner(f"Fetching Polygon data for {ticker} + SPY\u2026"):
            try:
                raw_df = fetch_ohlcv(ticker)
                risk_on, spy_price, spy_sma = check_market_regime()
                has_gap, gap_pct, gap_date = check_gap(raw_df)
                result_df = compute_all(raw_df, risk_on, has_gap)
                latest = result_df.dropna(subset=["composite"]).iloc[-1]
                res = interpret(latest, risk_on, has_gap, gap_pct, gap_date)
                res["as_of"] = latest.name.strftime("%Y-%m-%d")
                res["spy_price"] = spy_price
                res["spy_sma"]   = spy_sma
                st.session_state["res"] = res
                st.session_state["rdf"] = result_df
                st.session_state["tkr"] = ticker
            except Exception as exc:
                st.error(str(exc))
                return

    res       = st.session_state["res"]
    result_df = st.session_state["rdf"]
    tkr       = st.session_state["tkr"]

    # ── Banners ──
    if not res["risk_on"]:
        st.markdown(
            '<div class="banner-red">'
            '\u26a0\ufe0f  MARKET REGIME: RISK OFF \u2014 SPY below 200d SMA '
            f'(${res["spy_price"]:,.2f} vs ${res["spy_sma"]:,.2f}). '
            'All signals suppressed, composite capped at 35.</div>',
            unsafe_allow_html=True)
    if res["has_gap"]:
        st.markdown(
            '<div class="banner-amber">'
            f'\u26a0\ufe0f  GAP DETECTED \u2014 {res["gap_pct"]:.1f}% single-day gap '
            f'on {res["gap_date"]}. Score penalized \u221220 pts (event-driven momentum).</div>',
            unsafe_allow_html=True)

    # ── Header cards ──
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f'<div class="card"><div class="card-label">Signal</div>'
                    f'<div class="{res["css"]}">{res["signal"]}</div></div>',
                    unsafe_allow_html=True)
    with h2:
        st.markdown(f'<div class="card"><div class="card-label">Composite Score</div>'
                    f'<div class="score-big">{res["score"]:.1f}</div>'
                    f'<div class="card-sub">out of 100</div></div>',
                    unsafe_allow_html=True)
    with h3:
        cc = {"HIGH": "#3fb950", "MEDIUM": "#d29922", "LOW": "#6e7a8a"}[res["confidence"]]
        st.markdown(f'<div class="card"><div class="card-label">6-Month Confidence</div>'
                    f'<div class="card-value" style="color:{cc};font-size:1.6rem;">'
                    f'{res["confidence"]}</div>'
                    f'<div class="card-sub">{tkr} \u00b7 {res["as_of"]}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Range toggle ──
    range_labels = list(_RANGES.keys())
    if "rng" not in st.session_state:
        st.session_state["rng"] = "1Y"
    rcols = st.columns(len(range_labels))
    for i, lbl in enumerate(range_labels):
        with rcols[i]:
            if st.button(lbl, key=f"r_{lbl}", use_container_width=True,
                         type="primary" if st.session_state["rng"] == lbl else "secondary"):
                st.session_state["rng"] = lbl
                st.rerun()
    win = _RANGES[st.session_state["rng"]]

    # ── Charts ──
    st.plotly_chart(chart_price(tkr, result_df, win), use_container_width=True)
    st.plotly_chart(chart_score(tkr, result_df, win), use_container_width=True)

    # ── Two columns: breakdown + levels ──
    cl, cr = st.columns([3, 2])

    with cl:
        st.markdown("#### Component Breakdown")
        for c in res["components"]:
            s = c["score"]
            wt_tag = f'<span class="comp-wt">{c["wt"]}</span>'
            raw_tag = f'<span class="comp-raw">{c["raw"]}</span>' if c["raw"] else ""
            st.markdown(
                f'<div class="comp-row">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span class="comp-name">{c["name"]}</span>'
                f'<span>{wt_tag} \u00b7 {s:.0f}/100 {raw_tag}</span></div>'
                f'{_bar(s)}'
                f'<div class="comp-interp">{c["interp"]}</div></div>',
                unsafe_allow_html=True)

    with cr:
        st.markdown("#### Key Levels")
        price = res["price"]
        st.markdown(
            f'<div class="card" style="margin-bottom:12px;">'
            f'<div class="card-label">Current Price</div>'
            f'<div class="card-value" style="font-size:1.8rem;">${price:,.2f}</div></div>',
            unsafe_allow_html=True)

        levels = (
            _kl("100-day SMA", res["sma100"], price)
          + _kl("200-day SMA", res["sma200"], price)
          + _kl("50-day EMA",  res["ema50"],  price)
          + _kl("52-week High", res["high_52w"], price)
          + _kl("52-week Low",  res["low_52w"],  price)
          + _kl("ATR(20)", res["atr20"], price, is_price=False)
        )
        st.markdown(f'<div class="card">{levels}</div>', unsafe_allow_html=True)

        dist = (price / res["high_52w"] - 1) * 100 if res["high_52w"] > 0 else 0
        st.markdown(
            f'<div class="card" style="margin-top:12px;">'
            f'<div class="card-label">Distance to 52w High</div>'
            f'<div class="card-value">{dist:+.1f}%</div></div>',
            unsafe_allow_html=True)

    # ── Signal Rationale ──
    st.markdown("#### Signal Rationale")
    rationale = build_rationale(tkr, res)
    st.markdown(f'<div class="rationale">{rationale}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Trend Following Signal System v5 — Streamlit + Polygon.io
==========================================================
7-component composite trend signal for US equities.
Based on: Clenow, Antonacci, Moskowitz / AQR, Carver.

v4 changes
----------
FIX 1  20-day EMA smoothing on composite score
FIX 2  Hysteresis state machine (enter BUY @ 70 for 3 days, exit @ 55, etc.)
FIX 3  Volatility-regime-scaled thresholds (stock vol / SPY vol)

v5 changes
----------
Backtest uses yfinance for 10-year history (Polygon fallback if yf fails).
Sharpe ratio for strategy and buy-and-hold.
Date range presets: Full 10Y, COVID, Rate Hike, Last 2Y.
"""

from __future__ import annotations

import os
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from polygon import RESTClient

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

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
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
.stApp{background:#0a0e14}
section[data-testid="stSidebar"]{background:#12161e}
.signal-buy{color:#3fb950;font-size:2.6rem;font-weight:900;letter-spacing:.06em}
.signal-hold{color:#d29922;font-size:2.6rem;font-weight:900;letter-spacing:.06em}
.signal-avoid{color:#f85149;font-size:2.6rem;font-weight:900;letter-spacing:.06em}
.score-big{font-size:3.8rem;font-weight:900;line-height:1;
  background:linear-gradient(135deg,#58a6ff,#7ee787);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:#12161e;border:1px solid #1e2531;border-radius:12px;
  padding:16px 20px;text-align:center}
.card-label{color:#6e7a8a;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.09em;margin-bottom:4px}
.card-value{color:#e6edf3;font-size:1.25rem;font-weight:700}
.card-sub{color:#6e7a8a;font-size:.73rem;margin-top:2px}
.comp-row{background:#12161e;border:1px solid #1e2531;border-radius:10px;
  padding:14px 18px;margin-bottom:8px}
.comp-name{color:#c9d1d9;font-weight:700;font-size:.92rem}
.comp-wt{color:#6e7a8a;font-size:.78rem}
.comp-raw{color:#58a6ff;font-size:.78rem;font-family:monospace}
.comp-interp{color:#6e7a8a;font-size:.80rem;margin-top:4px}
.bar-bg{background:#1a1f2b;border-radius:6px;height:10px;width:100%;margin-top:6px}
.bar-fill{height:10px;border-radius:6px}
.bar-g{background:linear-gradient(90deg,#238636,#3fb950)}
.bar-y{background:linear-gradient(90deg,#9e6a03,#d29922)}
.bar-r{background:linear-gradient(90deg,#b62324,#f85149)}
.kl-row{display:flex;justify-content:space-between;padding:6px 0;
  border-bottom:1px solid #1a1f2b}
.kl-label{color:#6e7a8a;font-size:.84rem}
.kl-val{color:#e6edf3;font-weight:700;font-size:.88rem}
.kl-up{color:#3fb950;font-size:.82rem}
.kl-dn{color:#f85149;font-size:.82rem}
.banner-red{background:#2d1215;border:1px solid #f8514980;border-radius:10px;
  padding:12px 18px;color:#f85149;font-weight:700;font-size:.95rem;margin-bottom:12px}
.banner-amber{background:#2d2610;border:1px solid #d2992280;border-radius:10px;
  padding:12px 18px;color:#d29922;font-weight:700;font-size:.95rem;margin-bottom:12px}
.rationale{background:#12161e;border:1px solid #1e2531;border-radius:10px;
  padding:18px 22px;color:#a0aab5;font-size:.88rem;line-height:1.65;margin-top:16px}
.rationale b{color:#c9d1d9}
.bt-stat{background:#12161e;border:1px solid #1e2531;border-radius:10px;
  padding:14px 16px;text-align:center}
.bt-label{color:#6e7a8a;font-size:.70rem;text-transform:uppercase;letter-spacing:.08em}
.bt-val{color:#e6edf3;font-size:1.15rem;font-weight:700;margin-top:2px}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POLYGON DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _polygon() -> RESTClient:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY is not set. Run: export POLYGON_API_KEY=your_key")
    return RESTClient(api_key=key)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, years: int = 2) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    client = _polygon()
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365 + 400))
    aggs = list(client.list_aggs(
        ticker=ticker, multiplier=1, timespan="day",
        from_=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"),
        adjusted=True, sort="asc", limit=50_000))
    if not aggs:
        raise ValueError(f"No data for '{ticker}'. Check the symbol.")
    rows = [{"date": pd.Timestamp(a.timestamp, unit="ms"),
             "open": a.open, "high": a.high, "low": a.low,
             "close": a.close, "volume": a.volume} for a in aggs]
    df = (pd.DataFrame(rows).set_index("date").sort_index()
          [["open", "high", "low", "close", "volume"]].dropna())
    if len(df) < 252:
        raise ValueError(f"Only {len(df)} days for {ticker} (need 252+).")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_backtest_ohlcv(ticker: str) -> tuple[pd.DataFrame, bool]:
    """Fetch 10-year daily OHLCV for backtest. Try yfinance first, Polygon fallback.
    Returns (df, used_yfinance_flag)."""
    ticker = ticker.upper().strip()
    if _YF_AVAILABLE:
        try:
            raw = yf.download(ticker, period="10y", auto_adjust=True,
                              progress=False, timeout=30)
            if raw is not None and len(raw) >= 252:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                df.index.name = "date"
                if len(df) >= 252:
                    return df, True
        except Exception:
            pass
    try:
        return fetch_ohlcv(ticker, years=2), False
    except Exception:
        raise ValueError(f"Could not fetch backtest data for {ticker} from yfinance or Polygon.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 1 — 12m ABSOLUTE MOMENTUM (25 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TBILL_ANNUAL = 0.05


def calc_abs_momentum(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    p12 = close.shift(252)
    p1 = close.shift(21)
    ret = (p1 / p12) - 1
    score = (ret > TBILL_ANNUAL).astype(float) * 100
    return score, ret


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 2 — CLENOW SLOPE (30 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _slope_r2(lp: np.ndarray) -> tuple[float, float]:
    n = len(lp)
    if n < 30:
        return np.nan, np.nan
    x = np.arange(n)
    slope, _, r_val, _, _ = stats.linregress(x, lp)
    return (np.exp(slope * 252) - 1) * r_val ** 2, r_val ** 2


def calc_clenow(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    lc = np.log(close)
    w = 90
    raw_v, r2_v = [], []
    for i in range(len(lc)):
        if i < w - 1:
            raw_v.append(np.nan); r2_v.append(np.nan)
        else:
            a, r2 = _slope_r2(lc.iloc[i - w + 1: i + 1].values)
            raw_v.append(a); r2_v.append(r2)
    raw = pd.Series(raw_v, index=close.index, dtype=float)
    r2 = pd.Series(r2_v, index=close.index, dtype=float)
    score = raw.rolling(252, min_periods=90).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1], kind="rank")
        if len(x.dropna()) > 20 else 50, raw=False).clip(0, 100)
    return score, raw, r2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 3 — MA REGIME TRIPLE (20 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_ma(close: pd.Series):
    sma100 = close.rolling(100, min_periods=80).mean()
    sma200 = close.rolling(200, min_periods=150).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    c_a = (close > sma100).astype(float)
    c_b = (close > sma200).astype(float)
    c_c = (ema50 > ema200).astype(float)
    score = ((c_a + c_b + c_c) * 33.33).clip(0, 100)
    return score, sma100, sma200, ema50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 4 — ADX (10 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _wilder_adx(h, lo, c, period=14):
    pdm = h.diff(); mdm = -lo.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    a = 1.0 / period
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def calc_adx(h, lo, c):
    adx = _wilder_adx(h, lo, c)
    score = pd.Series(25.0, index=adx.index)
    score = score.where(adx < 20, np.nan)
    score = score.fillna(adx.where((adx >= 20) & (adx < 25)).apply(
        lambda x: 50.0 if pd.notna(x) else np.nan))
    score = score.fillna(adx.where((adx >= 25) & (adx < 30)).apply(
        lambda x: 75.0 if pd.notna(x) else np.nan))
    score = score.fillna(adx.where(adx >= 30).apply(
        lambda x: 100.0 if pd.notna(x) else np.nan))
    score = score.fillna(25.0)
    discount = pd.Series(1.0, index=adx.index).where(adx >= 20, 0.6)
    return score, discount, adx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 5 — 52w HIGH (15 %)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_52w(close):
    h52 = close.rolling(252, min_periods=126).max()
    l52 = close.rolling(252, min_periods=126).min()
    return (close / h52 * 100).clip(0, 100), h52, l52


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERLAYS — SPY gate, gap filter, ATR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_market_regime():
    spy = fetch_ohlcv("SPY")
    sma = spy["close"].rolling(200, min_periods=150).mean()
    return (spy["close"].iloc[-1] > sma.iloc[-1]), spy["close"].iloc[-1], sma.iloc[-1], spy


def check_gap(df):
    rec = df.iloc[-90:]
    gap = (rec["open"] / rec["close"].shift(1) - 1).abs()
    m = gap > 0.15
    if m.any():
        wi = gap[m].idxmax()
        return True, float(gap.loc[wi] * 100), wi.strftime("%Y-%m-%d")
    return False, None, None


def calc_atr(h, lo, c, p=20):
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 3 — VOLATILITY REGIME THRESHOLDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VOL_THRESHOLDS = {
    "high":   {"enter": 70, "exit_buy": 55, "enter_avoid": 40, "exit_avoid": 50, "label": "High-Vol"},
    "medium": {"enter": 68, "exit_buy": 52, "enter_avoid": 38, "exit_avoid": 48, "label": "Med-Vol"},
    "low":    {"enter": 65, "exit_buy": 48, "enter_avoid": 35, "exit_avoid": 45, "label": "Low-Vol"},
}


def calc_vol_regime(stock_close: pd.Series, spy_close: pd.Series) -> tuple[float, float, float, str, dict]:
    """Return (stock_vol, spy_vol, vol_ratio, regime_label, thresholds_dict)."""
    s_ret = stock_close.pct_change().dropna().iloc[-90:]
    spy_ret = spy_close.pct_change().dropna().iloc[-90:]
    s_vol = float(s_ret.std() * np.sqrt(252))
    spy_vol = float(spy_ret.std() * np.sqrt(252))
    ratio = s_vol / spy_vol if spy_vol > 0 else 1.0
    if ratio >= 1.5:
        key = "high"
    elif ratio >= 0.8:
        key = "medium"
    else:
        key = "low"
    return s_vol, spy_vol, ratio, key, _VOL_THRESHOLDS[key]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX 2 — HYSTERESIS STATE MACHINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_state_machine(smoothed: pd.Series, th: dict) -> pd.Series:
    """
    Hysteresis with 3-day confirmation for BUY entry.
    States: HOLD (default), BUY, AVOID
    """
    enter = th["enter"]
    exit_buy = th["exit_buy"]
    enter_avoid = th["enter_avoid"]
    exit_avoid = th["exit_avoid"]

    states = pd.Series("HOLD", index=smoothed.index)
    state = "HOLD"
    above_enter_count = 0     # consecutive days score >= enter

    for i in range(len(smoothed)):
        v = smoothed.iloc[i]
        if pd.isna(v):
            states.iloc[i] = state
            above_enter_count = 0
            continue

        if state == "HOLD":
            if v >= enter:
                above_enter_count += 1
                if above_enter_count >= 3:
                    state = "BUY"
                    above_enter_count = 0
            else:
                above_enter_count = 0
            if v < enter_avoid:
                state = "AVOID"
                above_enter_count = 0

        elif state == "BUY":
            above_enter_count = 0
            if v < exit_buy:
                state = "HOLD"
            # Could fall straight to AVOID if drop is severe
            if v < enter_avoid:
                state = "AVOID"

        elif state == "AVOID":
            above_enter_count = 0
            if v > exit_avoid:
                state = "HOLD"
            # Could jump to BUY-pending from AVOID via HOLD first
            # (stay strict: must pass through HOLD → 3-day confirm → BUY)

        states.iloc[i] = state

    return states


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPOSITE ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_all(df: pd.DataFrame, risk_on: bool, has_gap: bool,
                spy_close: pd.Series) -> tuple[pd.DataFrame, float, float, float, str, dict]:
    c, h, lo = df["close"], df["high"], df["low"]

    abs_s, abs_r = calc_abs_momentum(c)
    cle_s, cle_r, cle_r2 = calc_clenow(c)
    ma_s, sma100, sma200, ema50 = calc_ma(c)
    adx_s, adx_d, adx_r = calc_adx(h, lo, c)
    hw_s, h52, l52 = calc_52w(c)
    atr20 = calc_atr(h, lo, c)

    # Raw weighted composite (with ADX discount)
    raw_comp = (
        0.25 * abs_s * adx_d
      + 0.30 * cle_s * adx_d
      + 0.20 * ma_s * adx_d
      + 0.10 * adx_s
      + 0.15 * hw_s * adx_d
    )

    # Gap penalty
    if has_gap:
        raw_comp = (raw_comp - 20).clip(0, 100)
    # Market regime cap
    if not risk_on:
        raw_comp = raw_comp.clip(upper=35)
    raw_comp = raw_comp.clip(0, 100)

    # FIX 1 — 20-day EMA smoothing
    smoothed = raw_comp.ewm(span=20, adjust=False).mean().clip(0, 100)

    # FIX 3 — vol regime
    s_vol, spy_vol, vol_ratio, vol_key, thresholds = calc_vol_regime(c, spy_close)

    # FIX 2 — hysteresis state machine
    states = run_state_machine(smoothed, thresholds)

    r = df.copy()
    r["abs_mom_score"] = abs_s;  r["abs_mom_raw"] = abs_r
    r["clenow_score"] = cle_s;   r["clenow_raw"] = cle_r;  r["clenow_r2"] = cle_r2
    r["ma_score"] = ma_s
    r["adx_score"] = adx_s;      r["adx_raw"] = adx_r;      r["adx_discount"] = adx_d
    r["hw_score"] = hw_s
    r["raw_composite"] = raw_comp
    r["composite"] = smoothed
    r["state"] = states
    r["sma100"] = sma100;  r["sma200"] = sma200;  r["ema50"] = ema50
    r["high_52w"] = h52;   r["low_52w"] = l52;     r["atr20"] = atr20

    return r, s_vol, spy_vol, vol_ratio, vol_key, thresholds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_backtest(df: pd.DataFrame) -> dict:
    """
    Simple long-only backtest using the hysteresis states.
    Enter at next day's open when state flips to BUY.
    Exit  at next day's open when state flips to HOLD or AVOID.
    """
    valid = df.dropna(subset=["composite"]).copy()
    if len(valid) < 20:
        return None

    dates = valid.index
    opens = valid["open"]
    closes = valid["close"]
    states = valid["state"]

    # Track trades
    in_trade = False
    entry_price = 0.0
    entry_date = None
    trades: list[dict] = []

    # Equity series (start at $100)
    equity = pd.Series(100.0, index=dates, dtype=float)
    bnh_equity = pd.Series(100.0, index=dates, dtype=float)

    # Buy & hold: invest at first open
    bnh_start = opens.iloc[0]
    for i in range(len(valid)):
        bnh_equity.iloc[i] = 100.0 * closes.iloc[i] / bnh_start

    # Strategy
    prev_state = "HOLD"
    shares = 0.0
    cash = 100.0

    for i in range(1, len(valid)):
        curr_state = states.iloc[i]
        prev = states.iloc[i - 1]
        open_px = opens.iloc[i]

        # Entry: state just became BUY (wasn't BUY before)
        if curr_state == "BUY" and prev != "BUY" and not in_trade:
            shares = cash / open_px
            entry_price = open_px
            entry_date = dates[i]
            cash = 0.0
            in_trade = True

        # Exit: was BUY, now not BUY
        elif prev == "BUY" and curr_state != "BUY" and in_trade:
            cash = shares * open_px
            exit_price = open_px
            trades.append({
                "entry_date": entry_date,
                "exit_date": dates[i],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (exit_price / entry_price - 1) * 100,
                "days": (dates[i] - entry_date).days,
            })
            shares = 0.0
            in_trade = False

        # Mark equity
        if in_trade:
            equity.iloc[i] = shares * closes.iloc[i]
        else:
            equity.iloc[i] = cash

    # Close any open trade at last close
    if in_trade:
        last_close = closes.iloc[-1]
        cash = shares * last_close
        trades.append({
            "entry_date": entry_date,
            "exit_date": dates[-1],
            "entry_price": entry_price,
            "exit_price": last_close,
            "return_pct": (last_close / entry_price - 1) * 100,
            "days": (dates[-1] - entry_date).days,
        })
        equity.iloc[-1] = cash

    return {
        "equity": equity,
        "bnh_equity": bnh_equity,
        "trades": trades,
    }


def _sharpe(equity_series: pd.Series) -> float:
    """Annualized Sharpe ratio (Rf=0): mean(daily_ret)/std(daily_ret) * sqrt(252)."""
    daily = equity_series.pct_change().dropna()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return float((daily.mean() / daily.std()) * np.sqrt(252))


def _max_dd(eq: pd.Series) -> float:
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    return float(dd.min())


def calc_window_stats(bt: dict, start: str | None = None, end: str | None = None) -> dict:
    """Compute all backtest stats for a date window. None = full range."""
    equity = bt["equity"]
    bnh = bt["bnh_equity"]

    if start:
        equity = equity.loc[start:]
        bnh = bnh.loc[start:]
    if end:
        equity = equity.loc[:end]
        bnh = bnh.loc[:end]

    if len(equity) < 2:
        return None

    # Rebase both to 100 at window start
    eq = equity / equity.iloc[0] * 100
    bh = bnh / bnh.iloc[0] * 100

    strat_ret = (eq.iloc[-1] / 100 - 1) * 100
    bnh_ret = (bh.iloc[-1] / 100 - 1) * 100
    strat_dd = _max_dd(eq)
    bnh_dd = _max_dd(bh)
    strat_sharpe = _sharpe(eq)
    bnh_sharpe = _sharpe(bh)

    # Filter trades to window
    all_trades = bt["trades"]
    t_start = equity.index[0]
    t_end = equity.index[-1]
    window_trades = [t for t in all_trades
                     if t["entry_date"] >= t_start and t["entry_date"] <= t_end]
    n_trades = len(window_trades)
    wins = sum(1 for t in window_trades if t["return_pct"] > 0)
    win_rate = (wins / n_trades * 100) if n_trades > 0 else 0
    avg_dur = np.mean([t["days"] for t in window_trades]) if window_trades else 0

    return {
        "equity": eq,
        "bnh_equity": bh,
        "strat_return": strat_ret,
        "bnh_return": bnh_ret,
        "strat_dd": strat_dd,
        "bnh_dd": bnh_dd,
        "strat_sharpe": strat_sharpe,
        "bnh_sharpe": bnh_sharpe,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_duration": avg_dur,
        "trades": window_trades,
        "start_date": equity.index[0],
        "end_date": equity.index[-1],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERPRETATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _i_abs(s, r):
    if pd.isna(r): return "Insufficient history", ""
    rs = f"{r*100:+.1f}%"
    if s >= 100: return f"12m-1m return ({rs}) beats T-bill ({TBILL_ANNUAL*100:.0f}%)", rs
    return f"12m-1m return ({rs}) below T-bill hurdle", rs

def _i_cle(s, r, r2):
    if pd.isna(r): return "Insufficient history", ""
    rs = f"slope\u00d7R\u00b2={r:.3f}"
    if s >= 70: return f"Clean, strong trend ({rs}, R\u00b2={r2:.2f})", rs
    if s >= 40: return f"Moderate trend quality ({rs})", rs
    return f"Weak or deteriorating trend ({rs})", rs

def _i_ma(s):
    n = int(round(s / 33.33))
    return (f"{n}/3 MA conditions bullish" if n > 0 else "No MA conditions met"), f"{n}/3"

def _i_adx(s, r):
    rs = f"ADX={r:.1f}"
    if r >= 30: return f"Strong directional trend ({rs})", rs
    if r >= 25: return f"Moderate trend ({rs})", rs
    if r >= 20: return f"Marginal trend ({rs})", rs
    return f"Choppy / no trend ({rs}) \u2014 40% discount applied", rs

def _i_52w(s, p, h52):
    pct = p / h52 * 100 if h52 > 0 else 0
    if pct >= 95: return f"Near 52w high ({pct:.1f}%)", f"{pct:.1f}%"
    if pct >= 80: return f"Healthy pullback ({pct:.1f}%)", f"{pct:.1f}%"
    return f"Deep drawdown ({pct:.1f}%)", f"{pct:.1f}%"


def interpret(row, risk_on, has_gap, gap_pct, gap_date, thresholds, vol_key, vol_ratio):
    score = row["composite"]
    state = row["state"]
    adx = row["adx_raw"]

    css_map = {"BUY": "signal-buy", "HOLD": "signal-hold", "AVOID": "signal-avoid"}
    css = css_map.get(state, "signal-hold")

    # Confidence
    if adx >= 25 and (score >= 80 or score <= 15): conf = "HIGH"
    elif adx >= 20 and (score >= 55 or score <= 30): conf = "MEDIUM"
    else: conf = "LOW"

    ai, ar = _i_abs(row["abs_mom_score"], row["abs_mom_raw"])
    ci, cr_ = _i_cle(row["clenow_score"], row["clenow_raw"], row["clenow_r2"])
    mi, mr = _i_ma(row["ma_score"])
    di, dr = _i_adx(row["adx_score"], row["adx_raw"])
    hi, hr = _i_52w(row["hw_score"], row["close"], row["high_52w"])

    comps = [
        dict(name="12m Absolute Momentum", wt="25%", score=row["abs_mom_score"], raw=ar, interp=ai),
        dict(name="Clenow Adj. Slope", wt="30%", score=row["clenow_score"], raw=cr_, interp=ci),
        dict(name="MA Regime (Triple)", wt="20%", score=row["ma_score"], raw=mr, interp=mi),
        dict(name="ADX Trend Strength", wt="10%", score=row["adx_score"], raw=dr, interp=di),
        dict(name="52-Week High Prox.", wt="15%", score=row["hw_score"], raw=hr, interp=hi),
    ]

    # Overlays
    comps.append(dict(name="Market Regime (SPY)", wt="gate",
        score=100.0 if risk_on else 0.0,
        raw="RISK ON" if risk_on else "RISK OFF",
        interp="SPY above 200d SMA" if risk_on else "SPY below 200d SMA \u2014 capped at 35"))
    gap_sc = 0.0 if has_gap else 100.0
    comps.append(dict(name="Gap Filter (90d)", wt="penalty",
        score=gap_sc, raw=f"{gap_pct:.1f}% on {gap_date}" if has_gap else "None",
        interp="\u221220 pt penalty" if has_gap else "No large gaps"))

    # Vol regime row
    th = thresholds
    comps.append(dict(name="Vol Regime", wt="scaling",
        score=vol_ratio * 40,  # visual only
        raw=f"ratio={vol_ratio:.2f}",
        interp=(f"{th['label']} band: enter {th['enter']}, "
                f"exit BUY {th['exit_buy']}, avoid {th['enter_avoid']} "
                f"(stock vol/SPY vol = {vol_ratio:.2f})")))

    return dict(
        signal=state, css=css, score=score, confidence=conf,
        components=comps, risk_on=risk_on,
        has_gap=has_gap, gap_pct=gap_pct, gap_date=gap_date,
        price=row["close"], sma100=row["sma100"], sma200=row["sma200"],
        ema50=row["ema50"], high_52w=row["high_52w"], low_52w=row["low_52w"],
        atr20=row["atr20"], adx=adx, thresholds=thresholds,
        vol_key=vol_key, vol_ratio=vol_ratio, raw_score=row["raw_composite"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RATIONALE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_rationale(tkr, r):
    parts = []
    if not r["risk_on"]:
        parts.append(f"The S&P 500 is below its 200-day SMA, placing the market "
                     f"in risk-off mode. {tkr}'s composite is capped at 35.")

    cmap = {c["name"]: c for c in r["components"][:5]}
    a_s = cmap["12m Absolute Momentum"]["score"]
    c_s = cmap["Clenow Adj. Slope"]["score"]
    m_s = cmap["MA Regime (Triple)"]["score"]

    if a_s >= 100 and c_s >= 60:
        parts.append(f"{tkr}'s 12-month return clears the T-bill hurdle and the "
                     f"Clenow slope is in the {c_s:.0f}th percentile \u2014 strong, clean trend.")
    elif a_s >= 100:
        parts.append(f"{tkr} has positive absolute momentum but the Clenow slope "
                     f"at {c_s:.0f}/100 suggests the trend is noisy.")
    elif c_s >= 60:
        parts.append(f"The 90-day slope is solid ({c_s:.0f}) but 12-month return "
                     f"failed the T-bill hurdle.")
    else:
        parts.append(f"Both momentum and trend slope are weak for {tkr}.")

    n_ma = int(round(m_s / 33.33))
    parts.append(f"MA regime: {n_ma}/3 conditions bullish.")

    adx_v = r["adx"]
    if adx_v < 20:
        parts.append(f"ADX at {adx_v:.1f} signals a choppy market; scores discounted 40%.")
    elif adx_v >= 30:
        parts.append(f"ADX at {adx_v:.1f} confirms a strong directional trend.")

    th = r["thresholds"]
    parts.append(f"Vol regime: {th['label']} (stock/SPY vol = {r['vol_ratio']:.2f}). "
                 f"BUY entry threshold = {th['enter']}, exit = {th['exit_buy']}.")

    # Hysteresis state
    parts.append(f"The 20-day EMA smoothed score is {r['score']:.1f}. "
                 f"Hysteresis state: <b>{r['signal']}</b> "
                 f"(3-day confirmation required for BUY entry). "
                 f"Confidence: {r['confidence']}.")

    if r["has_gap"]:
        parts.append(f"A {r['gap_pct']:.1f}% gap on {r['gap_date']} triggered a "
                     f"\u221220 pt penalty.")
    return " ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RANGES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
_PLOT_BG = "#0a0e14"
_GRID = "#161b22"

def _sl(df, days):
    return df.iloc[-min(days, len(df)-1):].copy()


def chart_price(tkr, df, days):
    cd = _sl(df, days); d = cd.index
    fig = go.Figure()
    # regime shading
    above = cd["close"] > cd["sma200"]
    cat = above.map({True: "b", False: "r"})
    cm = {"b": "rgba(13,74,46,.30)", "r": "rgba(74,13,13,.30)"}
    prev = bs = d[0]; blocks = []
    for i, (dt_, r_) in enumerate(zip(d, cat)):
        if r_ != prev and i > 0: blocks.append((bs, d[i-1], prev)); bs = dt_
        prev = r_
    blocks.append((bs, d[-1], prev))
    for s, e, rt in blocks:
        fig.add_vrect(x0=s, x1=e, fillcolor=cm.get(rt, "rgba(0,0,0,0)"), line_width=0)
    fig.add_trace(go.Scatter(x=d, y=cd["close"], name="Price",
        line=dict(color="#58a6ff", width=2.2), hovertemplate="$%{y:,.2f}<extra>Price</extra>"))
    fig.add_trace(go.Scatter(x=d, y=cd["sma100"], name="100d SMA",
        line=dict(color="#f0883e", width=1.2, dash="dash"), opacity=.85))
    fig.add_trace(go.Scatter(x=d, y=cd["sma200"], name="200d SMA",
        line=dict(color="#f778ba", width=1.2, dash="dash"), opacity=.85))
    fig.add_trace(go.Scatter(x=d, y=cd["ema50"], name="50d EMA",
        line=dict(color="#7ee787", width=1.2, dash="dashdot"), opacity=.85))
    fig.update_layout(template="plotly_dark", paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        title=dict(text=f"{tkr} \u2014 Price & Moving Averages", font=dict(size=15, color="#e6edf3")),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#a0aab5"), bgcolor="rgba(0,0,0,0)"),
        height=400, margin=dict(l=50, r=20, t=50, b=25), hovermode="x unified",
        yaxis=dict(gridcolor=_GRID, tickprefix="$", tickformat=",.0f"), xaxis=dict(gridcolor=_GRID))
    return fig


def chart_score(tkr, df, days, thresholds):
    cd = _sl(df, days)
    raw = cd["raw_composite"].dropna()
    sm = cd["composite"].dropna()
    st_s = cd["state"]
    fig = go.Figure()

    # State background shading
    state_colors = {"BUY": "rgba(13,74,46,.25)", "HOLD": "rgba(40,40,50,.20)", "AVOID": "rgba(74,13,13,.25)"}
    if len(st_s) > 0:
        prev_st = st_s.iloc[0]; bs = st_s.index[0]; blocks = []
        for i in range(1, len(st_s)):
            if st_s.iloc[i] != prev_st:
                blocks.append((bs, st_s.index[i-1], prev_st))
                bs = st_s.index[i]
            prev_st = st_s.iloc[i]
        blocks.append((bs, st_s.index[-1], prev_st))
        for s, e, state in blocks:
            fig.add_vrect(x0=s, x1=e, fillcolor=state_colors.get(state, "rgba(0,0,0,0)"), line_width=0)

    # Raw as faint gray bars
    if len(raw) > 0:
        fig.add_trace(go.Bar(x=raw.index, y=raw.values, name="Raw Score",
            marker_color="rgba(100,110,125,0.25)", showlegend=True,
            hovertemplate="%{y:.1f}<extra>Raw</extra>"))

    # Smoothed as solid line colored by state
    if len(sm) > 0:
        colors = []
        for idx in sm.index:
            s = st_s.get(idx, "HOLD")
            if s == "BUY": colors.append("#3fb950")
            elif s == "AVOID": colors.append("#f85149")
            else: colors.append("#d29922")
        # Plot as line segments
        fig.add_trace(go.Scatter(x=sm.index, y=sm.values, name="Smoothed (20d EMA)",
            mode="lines", line=dict(color="#58a6ff", width=2.5),
            hovertemplate="%{y:.1f}<extra>Smoothed</extra>"))

    th = thresholds
    fig.add_hline(y=th["enter"], line_dash="dash", line_color="#3fb950", line_width=1,
                  annotation_text=f"BUY {th['enter']}", annotation_font_color="#3fb950")
    fig.add_hline(y=th["exit_buy"], line_dash="dot", line_color="rgba(63,185,80,0.5)", line_width=1,
                  annotation_text=f"Exit BUY {th['exit_buy']}", annotation_font_color="rgba(63,185,80,0.5)")
    fig.add_hline(y=th["enter_avoid"], line_dash="dash", line_color="#f85149", line_width=1,
                  annotation_text=f"AVOID {th['enter_avoid']}", annotation_font_color="#f85149")

    fig.update_layout(template="plotly_dark", paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        title=dict(text=f"{tkr} \u2014 Composite Score (smoothed + raw)", font=dict(size=15, color="#e6edf3")),
        height=300, margin=dict(l=50, r=20, t=50, b=25), hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#a0aab5"), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=_GRID, range=[-5, 105], title="Score"), xaxis=dict(gridcolor=_GRID),
        barmode="overlay")
    return fig


def chart_equity(bt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["equity"].index, y=bt["equity"].values,
        name="Strategy", line=dict(color="#58a6ff", width=2.2),
        hovertemplate="$%{y:,.2f}<extra>Strategy</extra>"))
    fig.add_trace(go.Scatter(x=bt["bnh_equity"].index, y=bt["bnh_equity"].values,
        name="Buy & Hold", line=dict(color="#6e7a8a", width=1.5, dash="dash"),
        hovertemplate="$%{y:,.2f}<extra>Buy & Hold</extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        title=dict(text="Equity Curve ($100 start)", font=dict(size=15, color="#e6edf3")),
        height=380, margin=dict(l=50, r=20, t=50, b=25), hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#a0aab5"), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=_GRID, tickprefix="$", tickformat=",.0f"), xaxis=dict(gridcolor=_GRID))
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bar(s):
    cls = "bar-g" if s >= 70 else ("bar-y" if s >= 45 else "bar-r")
    return f'<div class="bar-bg"><div class="bar-fill {cls}" style="width:{max(1,min(100,s)):.0f}%"></div></div>'

def _kl(label, val, price, is_price=True):
    if is_price:
        arr = '<span class="kl-up"> \u25b2</span>' if price > val else '<span class="kl-dn"> \u25bc</span>'
        return f'<div class="kl-row"><span class="kl-label">{label}</span><span class="kl-val">${val:,.2f}{arr}</span></div>'
    return f'<div class="kl-row"><span class="kl-label">{label}</span><span class="kl-val">{val:,.2f}</span></div>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # Top bar
    t1, t2, t3 = st.columns([2, 1, 4])
    with t1:
        ticker = st.text_input("Ticker", value="NVDA", max_chars=10,
            label_visibility="collapsed",
            placeholder="Enter US ticker (e.g. NVDA, AAPL, SPY)").upper().strip()
    with t2:
        go_btn = st.button("\U0001f50d  Analyze", type="primary", use_container_width=True)

    # Landing
    if not go_btn and "res" not in st.session_state:
        st.markdown(
            '<div style="text-align:center;margin-top:80px;">'
            '<div style="font-size:3rem;">\U0001f4c8</div>'
            '<div style="color:#c9d1d9;font-size:1.3rem;font-weight:700;margin-top:12px;">'
            'Trend Following Signal System v5</div>'
            '<div style="color:#6e7a8a;font-size:.92rem;margin-top:8px;">'
            'EMA-smoothed scoring \u00b7 Hysteresis state machine \u00b7 Vol-regime thresholds \u00b7 10-year backtest<br>'
            'Polygon.io + yfinance \u00b7 Enter a ticker above</div></div>',
            unsafe_allow_html=True)
        return

    # Compute
    if go_btn:
        with st.spinner(f"Fetching data for {ticker} + SPY\u2026"):
            try:
                raw_df = fetch_ohlcv(ticker)
                risk_on, spy_price, spy_sma, spy_df = check_market_regime()
                has_gap, gap_pct, gap_date = check_gap(raw_df)
                result_df, s_vol, spy_vol, vol_ratio, vol_key, thresholds = \
                    compute_all(raw_df, risk_on, has_gap, spy_df["close"])
                latest = result_df.dropna(subset=["composite"]).iloc[-1]
                res = interpret(latest, risk_on, has_gap, gap_pct, gap_date,
                                thresholds, vol_key, vol_ratio)
                res["as_of"] = latest.name.strftime("%Y-%m-%d")
                res["spy_price"] = spy_price
                res["spy_sma"] = spy_sma
                res["s_vol"] = s_vol
                res["spy_vol"] = spy_vol
                # Backtest: fetch 10y via yfinance, fallback to Polygon
                bt_df, bt_used_yf = fetch_backtest_ohlcv(ticker)
                _, spy_bt_used_yf = fetch_backtest_ohlcv("SPY")
                spy_bt_df = fetch_backtest_ohlcv("SPY")[0]
                spy_bt_sma = spy_bt_df["close"].rolling(200, min_periods=150).mean()
                bt_risk_on = (spy_bt_df["close"].iloc[-1] > spy_bt_sma.iloc[-1])
                bt_has_gap, _, _ = check_gap(bt_df)
                bt_result_df, *_ = compute_all(bt_df, bt_risk_on, bt_has_gap,
                                               spy_bt_df["close"])
                bt = run_backtest(bt_result_df)
                st.session_state["res"] = res
                st.session_state["rdf"] = result_df
                st.session_state["tkr"] = ticker
                st.session_state["bt"] = bt
                st.session_state["bt_used_yf"] = bt_used_yf
                st.session_state["bt_days"] = len(bt_df)
            except Exception as exc:
                st.error(str(exc))
                return

    res = st.session_state["res"]
    rdf = st.session_state["rdf"]
    tkr = st.session_state["tkr"]
    bt  = st.session_state.get("bt")
    bt_used_yf = st.session_state.get("bt_used_yf", True)
    bt_days = st.session_state.get("bt_days", 0)

    # Tabs
    tab_signal, tab_bt = st.tabs(["\U0001f4ca  Signal", "\U0001f4c8  Backtest"])

    # ════════════════════════════════════════════════════════════
    # TAB: SIGNAL
    # ════════════════════════════════════════════════════════════
    with tab_signal:
        # Banners
        if not res["risk_on"]:
            st.markdown(
                f'<div class="banner-red">\u26a0\ufe0f  MARKET REGIME: RISK OFF \u2014 '
                f'SPY below 200d SMA (${res["spy_price"]:,.2f} vs ${res["spy_sma"]:,.2f}). '
                f'Composite capped at 35.</div>', unsafe_allow_html=True)
        if res["has_gap"]:
            st.markdown(
                f'<div class="banner-amber">\u26a0\ufe0f  GAP DETECTED \u2014 '
                f'{res["gap_pct"]:.1f}% gap on {res["gap_date"]}. '
                f'Score penalized \u221220 pts.</div>', unsafe_allow_html=True)

        # Header cards
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(f'<div class="card"><div class="card-label">Signal (Hysteresis)</div>'
                        f'<div class="{res["css"]}">{res["signal"]}</div></div>', unsafe_allow_html=True)
        with h2:
            st.markdown(f'<div class="card"><div class="card-label">Smoothed Score</div>'
                        f'<div class="score-big">{res["score"]:.1f}</div>'
                        f'<div class="card-sub">raw: {res["raw_score"]:.1f}</div></div>', unsafe_allow_html=True)
        with h3:
            cc = {"HIGH": "#3fb950", "MEDIUM": "#d29922", "LOW": "#6e7a8a"}[res["confidence"]]
            st.markdown(f'<div class="card"><div class="card-label">Confidence</div>'
                        f'<div class="card-value" style="color:{cc};font-size:1.6rem;">'
                        f'{res["confidence"]}</div>'
                        f'<div class="card-sub">{tkr} \u00b7 {res["as_of"]}</div></div>', unsafe_allow_html=True)
        with h4:
            th = res["thresholds"]
            st.markdown(f'<div class="card"><div class="card-label">Vol Regime</div>'
                        f'<div class="card-value" style="font-size:1.2rem;">{th["label"]}</div>'
                        f'<div class="card-sub">ratio={res["vol_ratio"]:.2f} \u00b7 '
                        f'enter={th["enter"]} exit={th["exit_buy"]}</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Range toggle
        rng_labels = list(_RANGES.keys())
        if "rng" not in st.session_state: st.session_state["rng"] = "1Y"
        rcols = st.columns(len(rng_labels))
        for i, lbl in enumerate(rng_labels):
            with rcols[i]:
                if st.button(lbl, key=f"r_{lbl}", use_container_width=True,
                             type="primary" if st.session_state["rng"] == lbl else "secondary"):
                    st.session_state["rng"] = lbl; st.rerun()
        win = _RANGES[st.session_state["rng"]]

        # Charts
        st.plotly_chart(chart_price(tkr, rdf, win), use_container_width=True)
        st.plotly_chart(chart_score(tkr, rdf, win, res["thresholds"]), use_container_width=True)

        # Breakdown + levels
        cl, cr = st.columns([3, 2])
        with cl:
            st.markdown("#### Component Breakdown")
            for c in res["components"]:
                s = c["score"]
                wt = f'<span class="comp-wt">{c["wt"]}</span>'
                raw = f'<span class="comp-raw">{c["raw"]}</span>' if c["raw"] else ""
                st.markdown(
                    f'<div class="comp-row">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span class="comp-name">{c["name"]}</span>'
                    f'<span>{wt} \u00b7 {s:.0f}/100 {raw}</span></div>'
                    f'{_bar(s)}'
                    f'<div class="comp-interp">{c["interp"]}</div></div>',
                    unsafe_allow_html=True)

        with cr:
            st.markdown("#### Key Levels")
            p = res["price"]
            st.markdown(f'<div class="card" style="margin-bottom:12px;">'
                        f'<div class="card-label">Current Price</div>'
                        f'<div class="card-value" style="font-size:1.8rem;">${p:,.2f}</div></div>',
                        unsafe_allow_html=True)
            lvl = (_kl("100-day SMA", res["sma100"], p) + _kl("200-day SMA", res["sma200"], p)
                 + _kl("50-day EMA", res["ema50"], p) + _kl("52-week High", res["high_52w"], p)
                 + _kl("52-week Low", res["low_52w"], p) + _kl("ATR(20)", res["atr20"], p, False))
            st.markdown(f'<div class="card">{lvl}</div>', unsafe_allow_html=True)
            dist = (p / res["high_52w"] - 1) * 100 if res["high_52w"] > 0 else 0
            st.markdown(f'<div class="card" style="margin-top:12px;">'
                        f'<div class="card-label">Distance to 52w High</div>'
                        f'<div class="card-value">{dist:+.1f}%</div></div>', unsafe_allow_html=True)

        # Rationale
        st.markdown("#### Signal Rationale")
        st.markdown(f'<div class="rationale">{build_rationale(tkr, res)}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # TAB: BACKTEST
    # ════════════════════════════════════════════════════════════
    _BT_PRESETS = {
        "Full 10 years": (None, None),
        "COVID (2019\u20132021)": ("2019-01-01", "2021-12-31"),
        "Rate hike cycle (2022\u20132023)": ("2022-01-01", "2023-12-31"),
        "Last 2 years": (None, None),   # computed dynamically
    }

    with tab_bt:
        if bt is None:
            st.warning("Insufficient data to run a backtest.")
        else:
            st.markdown(f"#### {tkr} \u2014 Trend-Following Backtest")

            # Fallback warning
            if not bt_used_yf:
                st.markdown(
                    '<div class="banner-amber">\u26a0\ufe0f  yfinance unavailable \u2014 '
                    'backtest uses Polygon data (~2 years). '
                    'Install yfinance for full 10-year history.</div>',
                    unsafe_allow_html=True)

            src = "yfinance (10y)" if bt_used_yf else "Polygon (~2y fallback)"
            st.markdown(f'<div class="card-sub" style="margin-bottom:16px;">'
                        f'Data: {src} \u00b7 {bt_days:,} trading days. '
                        f'Enter at next open on BUY, exit at next open when signal drops. '
                        f'No transaction costs.</div>',
                        unsafe_allow_html=True)

            # Date range presets
            if "bt_preset" not in st.session_state:
                st.session_state["bt_preset"] = "Full 10 years"
            preset_labels = list(_BT_PRESETS.keys())
            pcols = st.columns(len(preset_labels))
            for i, lbl in enumerate(preset_labels):
                with pcols[i]:
                    if st.button(lbl, key=f"bp_{lbl}", use_container_width=True,
                                 type="primary" if st.session_state["bt_preset"] == lbl else "secondary"):
                        st.session_state["bt_preset"] = lbl
                        st.rerun()

            sel = st.session_state["bt_preset"]
            if sel == "Last 2 years":
                two_y_ago = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
                w_start, w_end = two_y_ago, None
            else:
                w_start, w_end = _BT_PRESETS[sel]

            ws = calc_window_stats(bt, start=w_start, end=w_end)

            if ws is None:
                st.warning(f"No backtest data available for the \u201c{sel}\u201d window.")
            else:
                # Row 1: Return, Max DD, Sharpe (strategy vs B&H)
                c1, c2, c3, c4 = st.columns(4)
                strat_c = "#3fb950" if ws["strat_return"] > 0 else "#f85149"
                bnh_c = "#3fb950" if ws["bnh_return"] > 0 else "#f85149"
                with c1:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">Strategy Return</div>'
                                f'<div class="bt-val" style="color:{strat_c}">'
                                f'{ws["strat_return"]:+.1f}%</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">Buy & Hold Return</div>'
                                f'<div class="bt-val" style="color:{bnh_c}">'
                                f'{ws["bnh_return"]:+.1f}%</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">Strategy Max DD</div>'
                                f'<div class="bt-val" style="color:#f85149">'
                                f'{ws["strat_dd"]:.1f}%</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">B&H Max DD</div>'
                                f'<div class="bt-val" style="color:#f85149">'
                                f'{ws["bnh_dd"]:.1f}%</div></div>', unsafe_allow_html=True)

                # Row 2: Sharpe, Trades, Win Rate, Avg Duration
                c5, c6, c7, c8 = st.columns(4)
                sh_strat = ws["strat_sharpe"]
                sh_bnh = ws["bnh_sharpe"]
                sh_strat_c = "#3fb950" if sh_strat > sh_bnh else ("#f85149" if sh_strat < sh_bnh else "#e6edf3")
                sh_bnh_c = "#3fb950" if sh_bnh > sh_strat else ("#f85149" if sh_bnh < sh_strat else "#e6edf3")
                with c5:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">Strategy Sharpe</div>'
                                f'<div class="bt-val" style="color:{sh_strat_c}">'
                                f'{sh_strat:.2f}</div></div>', unsafe_allow_html=True)
                with c6:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">B&H Sharpe</div>'
                                f'<div class="bt-val" style="color:{sh_bnh_c}">'
                                f'{sh_bnh:.2f}</div></div>', unsafe_allow_html=True)
                with c7:
                    wr_c = "#3fb950" if ws["win_rate"] >= 50 else "#d29922"
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">Win Rate</div>'
                                f'<div class="bt-val" style="color:{wr_c}">'
                                f'{ws["win_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
                with c8:
                    st.markdown(f'<div class="bt-stat"><div class="bt-label">Trades / Avg Days</div>'
                                f'<div class="bt-val">{ws["n_trades"]} / {ws["avg_duration"]:.0f}d</div></div>',
                                unsafe_allow_html=True)

                st.markdown("")
                st.plotly_chart(chart_equity(ws), use_container_width=True)

                # Trade log
                if ws["trades"]:
                    st.markdown("#### Trade Log")
                    tlog = pd.DataFrame(ws["trades"])
                    tlog["entry_date"] = pd.to_datetime(tlog["entry_date"]).dt.strftime("%Y-%m-%d")
                    tlog["exit_date"] = pd.to_datetime(tlog["exit_date"]).dt.strftime("%Y-%m-%d")
                    tlog["entry_price"] = tlog["entry_price"].map("${:,.2f}".format)
                    tlog["exit_price"] = tlog["exit_price"].map("${:,.2f}".format)
                    tlog["return_pct"] = tlog["return_pct"].map("{:+.1f}%".format)
                    tlog["days"] = tlog["days"].astype(int)
                    tlog.columns = ["Entry", "Exit", "Entry $", "Exit $", "Return", "Days"]
                    st.dataframe(tlog, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

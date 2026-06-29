"""
🔬 Backtest — AI Options Copilot
================================
Standalone Streamlit page. Drop this file into a `pages/` folder NEXT TO your
main app script:

    your_project/
      ai_options_copilot_v4_fixed.py   ← your live dashboard (the entrypoint)
      pages/
        2_Backtest.py                  ← this file

Streamlit auto-builds the left-hand page nav. No change to the main file needed.
(Rename to "2_🔬_Backtest.py" if you want an emoji in the nav label.)

─────────────────────────────────────────────────────────────────────────────
WHAT THIS DOES — read this before trusting a single number:
  • Backtests the CORE directional signal (EMA-trend + RSI + volume) — the same
    scoring the live dashboard uses — on the UNDERLYING instrument.
  • Lookahead-safe: indicators at bar i use only bars 0..i; entries fill at the
    NEXT bar's open; exits are checked against later bars' highs/lows.
  • Models round-trip cost and slippage. Sizes positions by fixed-fractional
    risk off current equity (compounding).

WHAT THIS DOES NOT DO — equally important:
  • It does NOT simulate option P&L. Buying a CALL/PUT adds implied-volatility
    and theta decay this test ignores. Read results as the edge of the
    DIRECTION (≈ trading the index future), not of an options position.
  • It tests the BASE signal only — not the MTF / VIX / PCR / expiry overlays.
    Those can't be reconstructed historically from the intraday endpoint.
  • Tuning the entry threshold until the curve looks pretty is curve-fitting.
    Judge robustness across symbols and sub-periods, not the prettiest line.
─────────────────────────────────────────────────────────────────────────────
"""

import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# ===================== PAGE CONFIG =====================
st.set_page_config(layout="wide", page_title="Backtest · AI Options Copilot", page_icon="🔬")

# ===================== LIGHT THEME (matches the live dashboard) =====================
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at 20% 50%, #ffffff 0%, #f4f5f7 100%); color:#16191d; }
    #MainMenu, footer { visibility: hidden; }
    .glass-card { background:#ffffff; border:1px solid rgba(0,0,0,0.08); border-radius:16px;
                  padding:1.1rem; box-shadow:0 4px 18px rgba(0,0,0,0.08); margin-bottom:0.6rem; }
    .metric-label { font-size:0.74rem; text-transform:uppercase; letter-spacing:0.5px; color:#5b6470; }
    .stButton > button { background:linear-gradient(135deg,#ea580c,#f97316); color:#fff; border:none;
                         border-radius:8px; font-weight:600; padding:0.5rem 1rem; width:100%; }
    .stButton > button:hover { box-shadow:0 0 20px rgba(234,88,12,0.35); }
    [data-testid="stMetricValue"] { color:#16191d !important; font-weight:700; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:#5b6470 !important; }
    section[data-testid="stSidebar"] { background:#f4f5f7; border-right:1px solid rgba(0,0,0,0.06); }
    section[data-testid="stSidebar"] * { color:#16191d; }
    h1,h2,h3,h4 { color:#16191d; }
    .pos { color:#16a34a; font-weight:700; } .neg { color:#dc2626; font-weight:700; }
    .disc { background:rgba(234,88,12,0.07); border:1px solid rgba(234,88,12,0.25);
            border-radius:10px; padding:0.7rem 0.9rem; font-size:0.83rem; color:#9a3412; }
</style>
""", unsafe_allow_html=True)

# ===================== SECRETS =====================
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID    = st.secrets.get("DHAN_CLIENT_ID", "")

DHAN_EPOCH_OFFSET    = 315_532_800
DHAN_EPOCH_THRESHOLD = 1_500_000_000

PRESET_ASSETS = {
    "NIFTY 50":   {"id": "13",    "seg": "IDX_I",  "inst": "INDEX"},
    "BANK NIFTY": {"id": "25",    "seg": "IDX_I",  "inst": "INDEX"},
    "FINNIFTY":   {"id": "27",    "seg": "IDX_I",  "inst": "INDEX"},
    "RELIANCE":   {"id": "2885",  "seg": "NSE_EQ", "inst": "EQUITY"},
    "HDFCBANK":   {"id": "3456",  "seg": "NSE_EQ", "inst": "EQUITY"},
    "ICICIBANK":  {"id": "4963",  "seg": "NSE_EQ", "inst": "EQUITY"},
    "TCS":        {"id": "11536", "seg": "NSE_EQ", "inst": "EQUITY"},
    "INFY":       {"id": "1594",  "seg": "NSE_EQ", "inst": "EQUITY"},
    "SBIN":       {"id": "3045",  "seg": "NSE_EQ", "inst": "EQUITY"},
}


def DHAN_HEADERS():
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    DHAN_CLIENT_ID,
        "Content-Type": "application/json",
        "Accept":       "application/json",
    }


# ===================== DATA =====================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_history(security_id: str, exchange_segment: str, instrument: str, days_back: int) -> pd.DataFrame:
    """Pull a long daily-candle history from Dhan's /charts/historical endpoint."""
    url = "https://api.dhan.co/v2/charts/historical"
    payload = {
        "securityId":      str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument":      instrument,
        "expiryCode":      0,
        "oi":              False,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "toDate":          pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    try:
        res = requests.post(url, json=payload, headers=DHAN_HEADERS(), timeout=30)
        if res.status_code in (401, 403):
            st.error("🔑 Dhan token expired — regenerate at https://dhan.co and update your secrets.")
            return pd.DataFrame()
        if res.status_code != 200:
            st.error(f"Dhan API error {res.status_code}")
            return pd.DataFrame()
        data = res.json()
        ts = data.get("timestamp", data.get("start_Time", []))
        if not ts:
            return pd.DataFrame()
        rows = []
        for i, t in enumerate(ts):
            t = int(t)
            if t < DHAN_EPOCH_THRESHOLD:
                t += DHAN_EPOCH_OFFSET
            if not (1_300_000_000 <= t <= 2_000_000_000):
                continue
            rows.append({
                "timestamp": pd.to_datetime(t, unit="s", utc=True).tz_convert("Asia/Kolkata"),
                "open": float(data["open"][i]), "high": float(data["high"][i]),
                "low": float(data["low"][i]),   "close": float(data["close"][i]),
                "volume": float(data["volume"][i]),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Network error: {e}")
        return pd.DataFrame()


# ===================== INDICATORS (causal) =====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["EMA_20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = c.ewm(span=50, adjust=False).mean()

    # RSI(14) — causal Wilder-ish via simple rolling means
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    # ATR(14)
    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift()).abs()
    lc = (df["low"]  - c.shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["VMA_20"] = df["volume"].rolling(20).mean()
    return df


def score_bar(row) -> float:
    """Replicates the live core signal score (trend 40 / momentum 40 / volume 20)."""
    # Trend
    if row["close"] > row["EMA_20"] > row["EMA_50"]:
        trend = 100
    elif row["close"] < row["EMA_20"] < row["EMA_50"]:
        trend = 0
    else:
        trend = 50
    # Momentum (RSI)
    rsi = row["RSI"]
    if 50 < rsi < 70:       mom = 90
    elif rsi >= 70:         mom = 20
    elif 45 <= rsi <= 50:   mom = 50
    elif 30 < rsi < 45:     mom = 25
    else:                   mom = 0
    # Volume direction
    vol = 50
    if pd.notna(row["VMA_20"]) and row["volume"] > row["VMA_20"] * 1.3:
        vol = 100 if row["close"] > row["open"] else 0
    return 0.40 * trend + 0.40 * mom + 0.20 * vol


# ===================== BACKTEST ENGINE =====================
def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    """
    One-position-at-a-time, long/short directional backtest.
    Entry: next bar open after a qualifying signal.
    Exit:  first of stop / target / time-stop, checked on each subsequent bar.
    """
    buy_th   = params["buy_th"]
    sell_th  = params["sell_th"]
    side     = params["side"]              # "Both" | "Long only" | "Short only"
    target_R = params["target_R"]          # 1.0 (T1) or 2.0 (T2)
    max_bars = params["max_bars"]
    rt_cost  = params["rt_cost"] / 100.0   # round-trip cost as fraction of entry notional
    slip_pts = params["slip_pts"]
    risk_pct = params["risk_pct"] / 100.0
    equity   = params["capital"]

    n = len(df)
    warmup = 60
    if n <= warmup + 5:
        return {"error": "Not enough bars after warmup to backtest."}

    position = None       # dict
    pending  = None       # dict {dir, atr, score, sig_idx}
    trades   = []
    eq_curve = []         # (timestamp, equity)

    for i in range(warmup, n):
        bar = df.iloc[i]

        # 1) fill a pending entry at THIS bar's open
        if position is None and pending is not None:
            d        = pending["dir"]          # +1 long, -1 short
            atr      = pending["atr"]
            stop_dist = atr * pending["atr_mult"]
            if stop_dist > 0 and pd.notna(stop_dist):
                fill = bar["open"] + d * slip_pts          # adverse slippage on entry
                sl   = fill - d * stop_dist
                tgt  = fill + d * stop_dist * target_R
                risk_amt = equity * risk_pct
                units = risk_amt / stop_dist
                position = {
                    "dir": d, "entry": fill, "sl": sl, "tgt": tgt,
                    "units": units, "entry_idx": i,
                    "entry_time": bar["timestamp"], "score": pending["score"],
                }
            pending = None

        # 2) manage an open position on THIS bar's range
        if position is not None:
            d = position["dir"]
            exit_price = None
            reason = None
            hi, lo = bar["high"], bar["low"]
            # conservative: if both stop & target inside the bar, assume stop first
            if d == 1:
                if lo <= position["sl"]:
                    exit_price, reason = position["sl"], "stop"
                elif hi >= position["tgt"]:
                    exit_price, reason = position["tgt"], "target"
            else:
                if hi >= position["sl"]:
                    exit_price, reason = position["sl"], "stop"
                elif lo <= position["tgt"]:
                    exit_price, reason = position["tgt"], "target"
            if exit_price is None and (i - position["entry_idx"]) >= max_bars:
                exit_price, reason = bar["close"], "time"

            if exit_price is not None:
                exit_fill = exit_price - d * slip_pts       # adverse slippage on exit
                gross = (exit_fill - position["entry"]) * d * position["units"]
                cost  = (abs(position["entry"]) + abs(exit_fill)) * position["units"] * rt_cost / 2
                pnl   = gross - cost
                equity += pnl
                stop_dist = abs(position["entry"] - position["sl"])
                R = ((exit_fill - position["entry"]) * d) / stop_dist if stop_dist else 0.0
                trades.append({
                    "entry_time": position["entry_time"], "exit_time": bar["timestamp"],
                    "dir": "LONG" if d == 1 else "SHORT",
                    "entry": round(position["entry"], 2), "exit": round(exit_fill, 2),
                    "exit_reason": reason, "R": round(R, 2),
                    "pnl": round(pnl, 0), "bars_held": i - position["entry_idx"],
                    "equity": round(equity, 0), "score": round(position["score"], 1),
                })
                position = None

        # 3) generate a new signal at THIS bar's close (only if flat & nothing pending)
        if position is None and pending is None and pd.notna(bar["ATR"]) and pd.notna(bar["EMA_50"]):
            s = score_bar(bar)
            direction = 0
            if s >= buy_th and side in ("Both", "Long only"):
                direction = 1
            elif s <= sell_th and side in ("Both", "Short only"):
                direction = -1
            if direction != 0:
                atr_mult = 2.0 if abs(s - 50) > 28 else 1.5     # same logic as live levels
                pending = {"dir": direction, "atr": bar["ATR"], "atr_mult": atr_mult,
                           "score": s, "sig_idx": i}

        eq_curve.append((bar["timestamp"], equity))

    eq = pd.DataFrame(eq_curve, columns=["timestamp", "equity"])
    tdf = pd.DataFrame(trades)
    return {"equity_curve": eq, "trades": tdf, "final_equity": equity,
            "start_capital": params["capital"], "df": df}


def compute_metrics(res: dict) -> dict:
    tdf = res["trades"]
    eq  = res["equity_curve"]
    cap = res["start_capital"]
    if tdf.empty:
        return {"n_trades": 0}

    wins   = tdf[tdf["pnl"] > 0]["pnl"]
    losses = tdf[tdf["pnl"] < 0]["pnl"]
    gross_w = wins.sum()
    gross_l = abs(losses.sum())
    n = len(tdf)

    # max drawdown on equity curve
    curve = eq["equity"].values
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    max_dd = dd.min() * 100 if len(dd) else 0.0

    days = max((eq["timestamp"].iloc[-1] - eq["timestamp"].iloc[0]).days, 1)
    total_ret = (res["final_equity"] / cap - 1) * 100
    cagr = ((res["final_equity"] / cap) ** (365.0 / days) - 1) * 100

    return {
        "n_trades":   n,
        "win_rate":   len(wins) / n * 100,
        "profit_factor": (gross_w / gross_l) if gross_l > 0 else float("inf"),
        "expectancy_R": tdf["R"].mean(),
        "avg_win":    wins.mean() if len(wins) else 0,
        "avg_loss":   losses.mean() if len(losses) else 0,
        "total_ret":  total_ret,
        "cagr":       cagr,
        "max_dd":     max_dd,
        "final_equity": res["final_equity"],
        "longs":      (tdf["dir"] == "LONG").sum(),
        "shorts":     (tdf["dir"] == "SHORT").sum(),
        "avg_bars":   tdf["bars_held"].mean(),
    }


# ===================== UI =====================
st.markdown("# 🔬 Strategy Backtest")
st.markdown(
    "<div class='disc'>This tests the <b>core directional signal on the underlying</b> "
    "(≈ trading the index/stock itself), <b>not</b> option P&amp;L — buying CALLs/PUTs adds IV "
    "and theta this model can't see. It's lookahead-safe and includes costs. Use it to ask "
    "<i>“does the direction have an edge?”</i>, then judge robustness across symbols and periods "
    "rather than chasing the prettiest equity curve.</div>",
    unsafe_allow_html=True,
)
st.write("")

if not DHAN_ACCESS_TOKEN:
    st.error("🔐 Missing DHAN_ACCESS_TOKEN in secrets.")
    st.stop()

with st.sidebar:
    st.markdown("## 🔬 Backtest Settings")
    sym_name = st.selectbox("Instrument", list(PRESET_ASSETS.keys()))
    a = PRESET_ASSETS[sym_name]
    lookback_label = st.selectbox("History", ["1 year", "2 years", "3 years", "5 years"], index=2)
    lookback_days = {"1 year": 365, "2 years": 730, "3 years": 1095, "5 years": 1825}[lookback_label]

    st.divider()
    st.markdown("### Signal")
    buy_th  = st.slider("BUY threshold (score ≥)",  55, 90, 60)
    sell_th = st.slider("SELL threshold (score ≤)", 10, 45, 42)
    side    = st.radio("Direction", ["Both", "Long only", "Short only"], index=0)

    st.divider()
    st.markdown("### Trade management")
    target_R = st.radio("Target", ["T1 (1R)", "T2 (2R)"], index=0)
    target_R = 1.0 if target_R.startswith("T1") else 2.0
    max_bars = st.slider("Time stop (max bars held)", 3, 40, 15)

    st.divider()
    st.markdown("### Money & costs")
    capital  = st.number_input("Starting capital (₹)", value=500_000, step=50_000, min_value=10_000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
    rt_cost  = st.slider("Round-trip cost (%)", 0.0, 0.50, 0.10, 0.01)
    slip_pts = st.number_input("Slippage (points/side)", value=1.0, step=0.5, min_value=0.0)

    run = st.button("▶️ Run Backtest", use_container_width=True)

if not run:
    st.info("Set your parameters in the sidebar and click **Run Backtest**. Daily timeframe — "
            "the only one with enough history for a meaningful sample.")
    st.stop()

with st.spinner(f"Fetching {lookback_label} of daily data for {sym_name}…"):
    raw = fetch_daily_history(a["id"], a["seg"], a["inst"], lookback_days)

if raw.empty or len(raw) < 80:
    st.error("Not enough data returned to backtest. Try a longer history or another instrument.")
    st.stop()

df = add_indicators(raw)
params = {
    "buy_th": buy_th, "sell_th": sell_th, "side": side, "target_R": target_R,
    "max_bars": max_bars, "rt_cost": rt_cost, "slip_pts": slip_pts,
    "risk_pct": risk_pct, "capital": capital,
}
res = run_backtest(df, params)
if "error" in res:
    st.error(res["error"])
    st.stop()

m = compute_metrics(res)
st.caption(f"**{sym_name}** · Daily · {raw['timestamp'].iloc[0].date()} → {raw['timestamp'].iloc[-1].date()} "
           f"· {len(raw)} bars · {m.get('n_trades', 0)} trades")

if m.get("n_trades", 0) == 0:
    st.warning("No trades were generated with these thresholds. Loosen the BUY/SELL thresholds and re-run.")
    st.stop()

# ── Headline metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net return", f"{m['total_ret']:.1f}%")
c2.metric("CAGR", f"{m['cagr']:.1f}%")
c3.metric("Win rate", f"{m['win_rate']:.1f}%")
pf = m["profit_factor"]
c4.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
c5.metric("Max drawdown", f"{m['max_dd']:.1f}%")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Trades", m["n_trades"])
c7.metric("Expectancy", f"{m['expectancy_R']:.2f} R")
c8.metric("Final equity", f"₹{m['final_equity']:,.0f}")
c9.metric("Long / Short", f"{m['longs']} / {m['shorts']}")
c10.metric("Avg hold", f"{m['avg_bars']:.1f} bars")

# Quick read on quality
verdict = []
if m["profit_factor"] != float("inf") and m["profit_factor"] < 1.1:
    verdict.append("Profit factor below ~1.1 — no meaningful edge after costs.")
if m["n_trades"] < 30:
    verdict.append(f"Only {m['n_trades']} trades — too few to conclude anything; treat as anecdote.")
if m["max_dd"] < -35:
    verdict.append("Drawdown beyond ~35% — likely untradeable psychologically.")
if verdict:
    st.markdown("<div class='disc'>⚠️ " + "<br>⚠️ ".join(verdict) + "</div>", unsafe_allow_html=True)

# ── Equity curve + drawdown ───────────────────────────────────────────────────
eq = res["equity_curve"]
curve = eq["equity"].values
peak = np.maximum.accumulate(curve)
dd = (curve - peak) / peak * 100

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    row_heights=[0.7, 0.3], subplot_titles=("Equity curve", "Drawdown %"))
fig.add_trace(go.Scatter(x=eq["timestamp"], y=curve, name="Equity",
                         line=dict(color="#ea580c", width=2), fill="tozeroy",
                         fillcolor="rgba(234,88,12,0.08)"), row=1, col=1)
fig.add_trace(go.Scatter(x=eq["timestamp"], y=dd, name="Drawdown",
                         line=dict(color="#dc2626", width=1), fill="tozeroy",
                         fillcolor="rgba(220,38,38,0.10)"), row=2, col=1)
fig.update_layout(template="plotly_white", height=460, margin=dict(l=0, r=0, t=30, b=0),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  showlegend=False, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Price with trade markers ──────────────────────────────────────────────────
tdf = res["trades"]
pfig = go.Figure()
pfig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close",
                          line=dict(color="#9aa3af", width=1)))
longs  = tdf[tdf["dir"] == "LONG"]
shorts = tdf[tdf["dir"] == "SHORT"]
pfig.add_trace(go.Scatter(x=longs["entry_time"], y=longs["entry"], mode="markers", name="Long entry",
                          marker=dict(color="#16a34a", size=8, symbol="triangle-up")))
pfig.add_trace(go.Scatter(x=shorts["entry_time"], y=shorts["entry"], mode="markers", name="Short entry",
                          marker=dict(color="#dc2626", size=8, symbol="triangle-down")))
pfig.update_layout(template="plotly_white", height=360, margin=dict(l=0, r=0, t=30, b=0),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   title="Price & entries", hovermode="x unified",
                   legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
st.plotly_chart(pfig, use_container_width=True, config={"displayModeBar": False})

# ── R-multiple distribution + trade log ───────────────────────────────────────
colA, colB = st.columns([1, 1.4])
with colA:
    st.markdown("#### R-multiple distribution")
    hfig = go.Figure(go.Histogram(x=tdf["R"], nbinsx=20, marker_color="#ea580c"))
    hfig.add_vline(x=0, line_color="#16191d", line_width=1)
    hfig.update_layout(template="plotly_white", height=300, margin=dict(l=0, r=0, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="R", yaxis_title="trades")
    st.plotly_chart(hfig, use_container_width=True, config={"displayModeBar": False})
    st.metric("Avg win", f"₹{m['avg_win']:,.0f}")
    st.metric("Avg loss", f"₹{m['avg_loss']:,.0f}")

with colB:
    st.markdown("#### Trade log")
    show = tdf.copy()
    show["entry_time"] = show["entry_time"].dt.strftime("%Y-%m-%d")
    show["exit_time"]  = show["exit_time"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        show[["entry_time", "exit_time", "dir", "entry", "exit", "exit_reason",
              "R", "pnl", "bars_held", "score"]],
        use_container_width=True, hide_index=True, height=300,
    )
    st.download_button("⬇️ Download trades (CSV)", tdf.to_csv(index=False).encode(),
                       file_name=f"backtest_{sym_name.replace(' ', '_')}.csv", mime="text/csv")

st.caption("Reminder: results are in-sample on the underlying. Re-run across several instruments "
           "and time windows before believing the edge, and remember the live system also buys "
           "options — IV and theta will change the real outcome.")

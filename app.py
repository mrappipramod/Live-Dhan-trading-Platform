"""
AI Options Copilot v4 — Institutional Grade
============================================
New in v4:
  • India VIX filter          — suppress signals in high-volatility regimes
  • Multi-timeframe (MTF)     — 5m + 1H + Daily confluence; trade only when aligned
  • OI / PCR analysis         — Put-Call Ratio from Dhan option chain (Index only)
  • VWAP + ±1σ/2σ bands       — institutional intraday level
  • Support & Resistance       — swing high/low detection, entry proximity check
  • Expiry calendar awareness  — flag/suppress signals near weekly expiry
  • Signal journal             — every signal logged to Supabase with outcome tracking
  • Telegram alerts            — push when high-confidence signal fires
  • Position sizing engine     — Kelly / fixed-fractional lot calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from supabase import create_client, Client

# ===================== PAGE CONFIG =====================
st.set_page_config(
    layout="wide",
    page_title="AI Options Copilot v4",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ===================== CSS =====================
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at 20% 50%, #0d1117 0%, #080B0E 100%); color: #e6edf3; }
    #MainMenu, footer, header { visibility: hidden; }
    .glass-card {
        background: rgba(22,27,34,0.6); backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
        padding: 1.2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, border-color 0.2s; margin-bottom: 0.6rem;
    }
    .glass-card:hover { border-color: rgba(88,166,255,0.3); transform: translateY(-2px); }
    .metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; }
    .badge-call   { background:rgba(46,160,67,0.25);  color:#3fb950; padding:0.3rem 1rem; border-radius:30px; font-weight:700; font-size:1.2rem; border:1px solid rgba(46,160,67,0.4);   display:inline-block; }
    .badge-put    { background:rgba(248,81,73,0.25);  color:#f85149; padding:0.3rem 1rem; border-radius:30px; font-weight:700; font-size:1.2rem; border:1px solid rgba(248,81,73,0.4);  display:inline-block; }
    .badge-neutral{ background:rgba(139,148,158,0.2); color:#8b949e; padding:0.3rem 1rem; border-radius:30px; font-weight:700; font-size:1.2rem; border:1px solid rgba(139,148,158,0.3);display:inline-block; }
    .badge-buy    { background:rgba(46,160,67,0.2);   color:#3fb950; padding:0.25rem 0.9rem; border-radius:20px; font-weight:700; font-size:1.1rem; border:1px solid rgba(46,160,67,0.3);   display:inline-block; }
    .badge-sell   { background:rgba(248,81,73,0.2);   color:#f85149; padding:0.25rem 0.9rem; border-radius:20px; font-weight:700; font-size:1.1rem; border:1px solid rgba(248,81,73,0.3);  display:inline-block; }
    .badge-hold   { background:rgba(139,148,158,0.2); color:#8b949e; padding:0.25rem 0.9rem; border-radius:20px; font-weight:700; font-size:1.1rem; border:1px solid rgba(139,148,158,0.3);display:inline-block; }
    .reason-item  { padding:0.35rem 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.85rem; }
    .reason-item:last-child { border-bottom: none; }
    .stButton > button { background:linear-gradient(135deg,#238636,#2ea043); color:white; border:none; border-radius:8px; font-weight:600; padding:0.5rem 1rem; transition:all 0.2s; width:100%; }
    .stButton > button:hover { transform:scale(1.02); box-shadow:0 0 20px rgba(46,160,67,0.3); }
    .confidence-bar  { height:6px; border-radius:3px; background:#21262d; margin-top:5px; }
    .confidence-fill { height:6px; border-radius:3px; background:linear-gradient(90deg,#3fb950,#58a6ff); }
    .info-note  { background:rgba(88,166,255,0.08); border:1px solid rgba(88,166,255,0.2); border-radius:8px; padding:0.4rem 0.7rem; font-size:0.78rem; color:#58a6ff; margin-top:0.4rem; }
    .warn-note  { background:rgba(248,81,73,0.08);  border:1px solid rgba(248,81,73,0.2);  border-radius:8px; padding:0.4rem 0.7rem; font-size:0.78rem; color:#f85149; margin-top:0.4rem; }
    .ok-note    { background:rgba(46,160,67,0.08);  border:1px solid rgba(46,160,67,0.2);  border-radius:8px; padding:0.4rem 0.7rem; font-size:0.78rem; color:#3fb950; margin-top:0.4rem; }
    .mtf-row    { display:flex; gap:0.4rem; margin-bottom:0.5rem; }
    .mtf-pill   { flex:1; text-align:center; padding:0.4rem 0.2rem; border-radius:8px; font-size:0.75rem; font-weight:600; }
    .mtf-bull   { background:rgba(46,160,67,0.15);  color:#3fb950; border:1px solid rgba(46,160,67,0.3); }
    .mtf-bear   { background:rgba(248,81,73,0.15);  color:#f85149; border:1px solid rgba(248,81,73,0.3); }
    .mtf-neut   { background:rgba(139,148,158,0.15);color:#8b949e; border:1px solid rgba(139,148,158,0.3); }
    .vix-ok     { color:#3fb950; font-weight:600; }
    .vix-warn   { color:#ffa657; font-weight:600; }
    .vix-danger { color:#f85149; font-weight:600; }
    .sr-tag     { display:inline-block; padding:0.1rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:600; margin:0.15rem; }
    .sr-res     { background:rgba(248,81,73,0.15);  color:#f85149; }
    .sr-sup     { background:rgba(46,160,67,0.15);  color:#3fb950; }
    .pcr-bull   { color:#3fb950; font-weight:700; }
    .pcr-bear   { color:#f85149; font-weight:700; }
    .pcr-neut   { color:#8b949e; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ==================== SECRETS ====================
SUPABASE_URL       = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY       = st.secrets.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_KEY", ""))
DHAN_ACCESS_TOKEN  = st.secrets.get("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID     = st.secrets.get("DHAN_CLIENT_ID", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = st.secrets.get("TELEGRAM_CHAT_ID", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("🔐 Missing Supabase credentials.")
    st.stop()
if not DHAN_ACCESS_TOKEN:
    st.error("🔐 Missing DHAN_ACCESS_TOKEN. Dhan tokens expire daily — regenerate at https://dhan.co")
    st.stop()
if not DHAN_CLIENT_ID:
    st.warning("⚠️ DHAN_CLIENT_ID not set — option chain / PCR will be unavailable. Find it at web.dhan.co → Profile → Client ID")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== CONSTANTS ====================
OPTIONS_SEGMENTS      = {"IDX_I", "NSE_FNO"}
DHAN_EPOCH_OFFSET     = 315_532_800
DHAN_EPOCH_THRESHOLD  = 1_500_000_000

# India VIX thresholds
VIX_SAFE    = 15.0   # below → normal signals
VIX_CAUTION = 20.0   # 15–20 → narrow ATR, flag
VIX_DANGER  = 25.0   # above → suppress options signals

# Nifty/BankNifty weekly expiry days (0=Mon … 6=Sun)
EXPIRY_WEEKDAY = {
    "13": 3,   # Nifty → Thursday
    "25": 2,   # Bank Nifty → Wednesday
    "27": 3,   # FinNifty → Thursday
}


# ==================== SUPABASE HELPERS ====================
def ensure_journal_table():
    """Create signal_journal table if it doesn't exist (idempotent via upsert)."""
    pass  # Supabase DDL must be run in the dashboard; see README comment below


# ==================== DHAN API ====================
DHAN_HEADERS = lambda: {
    "access-token": DHAN_ACCESS_TOKEN,
    "client-id":    DHAN_CLIENT_ID,
    "Content-Type": "application/json",
    "Accept":       "application/json",
}

def pull_historical_dhan(security_id, exchange_segment, instrument_type, days_back, interval=None):
    url = "https://api.dhan.co/v2/charts/historical"
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument_type,
        "expiryCode": 0,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "toDate":   pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    if interval:
        payload["interval"] = interval
    try:
        res = requests.post(url, json=payload, headers=DHAN_HEADERS(), timeout=15)
        if res.status_code in (401, 403):
            st.error("🔑 Dhan token expired. Regenerate at https://dhan.co")
            return []
        if res.status_code != 200:
            st.error(f"Dhan API error {res.status_code}")
            return []
        data = res.json()
        timestamps = data.get("timestamp", data.get("start_Time", []))
        if not timestamps:
            return []
        candles = []
        for i, ts in enumerate(timestamps):
            ts = int(ts)
            if ts < DHAN_EPOCH_THRESHOLD:
                ts += DHAN_EPOCH_OFFSET
            if not (1_420_000_000 <= ts <= 2_000_000_000):
                continue
            dt_str = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata").isoformat()
            candles.append({
                "timestamp": dt_str,
                "open":   data["open"][i],  "high": data["high"][i],
                "low":    data["low"][i],   "close": data["close"][i],
                "volume": data["volume"][i],
            })
        return candles
    except Exception as e:
        st.error(f"Network error: {e}")
    return []


@st.cache_data(ttl=300)
def fetch_india_vix() -> float | None:
    """
    Fetch India VIX from NSE's JSON endpoint.
    Returns float or None on failure.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com",
            "Accept": "application/json",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        res = session.get(
            "https://www.nseindia.com/api/allIndices",
            headers=headers, timeout=10
        )
        if res.status_code == 200:
            data = res.json().get("data", [])
            for item in data:
                if "INDIA VIX" in item.get("indexSymbol", "").upper():
                    return float(item.get("last", 0))
    except Exception:
        pass
    return None


# PCR cache keyed by security_id only — mode (intraday/daily) is irrelevant
@st.cache_data(ttl=180, show_spinner=False)
def fetch_option_chain_pcr(security_id: str) -> dict:
    """
    Fetch PCR, max pain, and OI walls from Dhan option chain API v2.

    Two-step flow per Dhan docs:
      1. POST /v2/optionchain/expirylist  → get nearest expiry date string
      2. POST /v2/optionchain             → get full chain for that expiry

    Required headers: access-token + client-id (both mandatory per Dhan docs).
    UnderlyingScrip must be int, Expiry must be a valid date string "YYYY-MM-DD".

    Response structure: data.oc is a dict keyed by strike price strings,
    each value has "ce" and "pe" sub-dicts with "oi", "last_price", greeks etc.
    """
    if not DHAN_CLIENT_ID:
        return {}

    expiry_url = "https://api.dhan.co/v2/optionchain/expirylist"
    chain_url  = "https://api.dhan.co/v2/optionchain"

    expiry_payload = {
        "UnderlyingScrip": int(security_id),
        "UnderlyingSeg":   "IDX_I",
    }

    try:
        # Step 1: get expiry list and pick nearest
        exp_res = requests.post(
            expiry_url, json=expiry_payload,
            headers=DHAN_HEADERS(), timeout=15
        )
        if exp_res.status_code != 200:
            return {}

        exp_data   = exp_res.json()
        expiry_list = exp_data.get("data", [])
        if not expiry_list:
            return {}

        # Dhan returns dates as "YYYY-MM-DD" strings, sorted nearest first
        nearest_expiry = expiry_list[0]

        # Step 2: fetch full option chain for nearest expiry
        chain_payload = {
            "UnderlyingScrip": int(security_id),
            "UnderlyingSeg":   "IDX_I",
            "Expiry":          nearest_expiry,
        }
        chain_res = requests.post(
            chain_url, json=chain_payload,
            headers=DHAN_HEADERS(), timeout=15
        )
        if chain_res.status_code != 200:
            return {}

        data = chain_res.json()
        if data.get("status") != "success":
            return {}

        # data["data"]["oc"] is {"25650.000000": {"ce": {...}, "pe": {...}}, ...}
        oc = data.get("data", {}).get("oc", {})
        if not oc:
            return {}

        total_call_oi = 0
        total_put_oi  = 0
        parsed_strikes = []

        for strike_str, sides in oc.items():
            sp       = float(strike_str)
            call_oi  = int((sides.get("ce") or {}).get("oi", 0) or 0)
            put_oi   = int((sides.get("pe") or {}).get("oi", 0) or 0)
            total_call_oi += call_oi
            total_put_oi  += put_oi
            parsed_strikes.append({"strike": sp, "call_oi": call_oi, "put_oi": put_oi})

        if total_call_oi == 0:
            return {}

        pcr = round(total_put_oi / total_call_oi, 2)

        # Max pain: strike causing maximum aggregate loss for option buyers
        max_pain_strike = None
        min_pain = float("inf")
        for row in parsed_strikes:
            sp   = row["strike"]
            pain = sum(
                max(0, sp - r["strike"]) * r["call_oi"] +
                max(0, r["strike"] - sp) * r["put_oi"]
                for r in parsed_strikes
            )
            if pain < min_pain:
                min_pain        = pain
                max_pain_strike = sp

        # Top 3 OI strikes — call walls = resistance, put walls = support
        top_call = sorted(parsed_strikes, key=lambda x: x["call_oi"], reverse=True)[:3]
        top_put  = sorted(parsed_strikes, key=lambda x: x["put_oi"],  reverse=True)[:3]

        interpretation = (
            "Bullish" if pcr > 1.3
            else "Bearish" if pcr < 0.7
            else "Neutral"
        )

        return {
            "pcr":                 pcr,
            "max_pain":            max_pain_strike,
            "total_call_oi":       total_call_oi,
            "total_put_oi":        total_put_oi,
            "interpretation":      interpretation,
            "expiry":              nearest_expiry,
            "top_call_resistance": [r["strike"] for r in top_call],
            "top_put_support":     [r["strike"] for r in top_put],
        }

    except Exception:
        return {}


def days_to_expiry(security_id: str) -> int | None:
    """Return calendar days to next weekly expiry for known index IDs."""
    weekday = EXPIRY_WEEKDAY.get(str(security_id))
    if weekday is None:
        return None
    today = date.today()
    days_ahead = (weekday - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return days_ahead


def send_telegram_alert(message: str):
    """Fire-and-forget Telegram message."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


def log_signal_to_journal(signal: dict, display_name: str, mode_label: str):
    """
    Upsert signal into signal_journal table.
    Required Supabase table DDL (run once in Supabase SQL editor):
    ─────────────────────────────────────────────────────────────
    create table if not exists signal_journal (
        id            bigint generated always as identity primary key,
        created_at    timestamptz default now(),
        symbol        text,
        instrument    text,
        mode          text,
        score         int,
        confidence    int,
        direction     text,
        option_action text,
        regime        text,
        entry         text,
        stop_loss     text,
        target_1      text,
        target_2      text,
        vix           float,
        pcr           float,
        mtf_aligned   bool,
        outcome       text default 'pending'
    );
    ─────────────────────────────────────────────────────────────
    """
    try:
        row = {
            "symbol":       display_name,
            "instrument":   mode_label,
            "mode":         mode_label,
            "score":        signal.get("score"),
            "confidence":   signal.get("confidence"),
            "direction":    signal.get("direction"),
            "option_action":signal.get("option_action") or signal.get("direction"),
            "regime":       signal.get("regime"),
            "entry":        signal.get("entry"),
            "stop_loss":    signal.get("stop_loss"),
            "target_1":     signal.get("target_1"),
            "target_2":     signal.get("target_2"),
            "vix":          signal.get("vix"),
            "pcr":          signal.get("pcr"),
            "mtf_aligned":  signal.get("mtf_aligned"),
            "outcome":      "pending",
        }
        supabase.table("signal_journal").insert(row).execute()
    except Exception:
        pass  # Journal failure must never crash the dashboard


# ==================== QUANT ENGINE ====================
class AdvancedQuantEngine:

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 5:
            return df

        close = df["close"]
        df["EMA_9"]  = close.ewm(span=9,  adjust=False).mean()
        df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

        # RSI (exclude last candle to avoid lookahead on live incomplete bar)
        delta = close.iloc[:-1].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / (loss + 1e-10)
        rsi   = (100 - 100 / (1 + rs)).reindex(df.index).ffill().fillna(50)
        df["RSI"] = rsi

        # ATR
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - close.shift()).abs()
        lc  = (df["low"]  - close.shift()).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().fillna(close * 0.005)

        # Volume MA
        df["VMA_20"] = df["volume"].rolling(20).mean().fillna(df["volume"])

        # VWAP + bands (session-level: reset each day for intraday, full series for daily)
        typical = (df["high"] + df["low"] + close) / 3
        cum_vol = df["volume"].cumsum()
        cum_tvol = (typical * df["volume"]).cumsum()
        df["VWAP"] = cum_tvol / cum_vol.replace(0, np.nan)

        # VWAP standard deviation bands
        squared_diff = (typical - df["VWAP"]) ** 2
        rolling_var  = (squared_diff * df["volume"]).cumsum() / cum_vol.replace(0, np.nan)
        vwap_std     = np.sqrt(rolling_var.fillna(0))
        df["VWAP_U1"] = df["VWAP"] + vwap_std
        df["VWAP_L1"] = df["VWAP"] - vwap_std
        df["VWAP_U2"] = df["VWAP"] + 2 * vwap_std
        df["VWAP_L2"] = df["VWAP"] - 2 * vwap_std

        # Support & Resistance via swing high/low (20-bar rolling)
        window = 20
        df["swing_high"] = df["high"].rolling(window, center=True).max()
        df["swing_low"]  = df["low"].rolling(window, center=True).min()

        return df

    @staticmethod
    def get_sr_levels(df: pd.DataFrame, n: int = 3) -> tuple[list, list]:
        """Return top-N resistance and support levels from recent price history."""
        if df.empty or "swing_high" not in df.columns:
            return [], []
        recent = df.tail(60)
        # Cluster close values to find meaningful levels
        resistances = sorted(
            recent["swing_high"].dropna().unique(), reverse=True
        )[:n]
        supports = sorted(
            recent["swing_low"].dropna().unique()
        )[:n]
        return list(resistances), list(supports)

    @staticmethod
    def compute_signal_single(df: pd.DataFrame) -> dict:
        """Core signal computation for one timeframe. Returns scored dict."""
        if df.empty or len(df) < 5 or "RSI" not in df.columns:
            return {"direction": "HOLD", "score": 50, "trend": 50, "momentum": 50, "volume": 50}

        latest = df.iloc[-1]

        # Trend (40%)
        trend_score = 50
        if latest["close"] > latest["EMA_20"] > latest["EMA_50"]:
            trend_score = 100
        elif latest["close"] < latest["EMA_20"] < latest["EMA_50"]:
            trend_score = 0

        # Momentum / RSI (40%)
        rsi = latest["RSI"]
        if 50 < rsi < 70:   momentum_score = 90
        elif rsi >= 70:      momentum_score = 20
        elif 45 <= rsi <= 50:momentum_score = 50
        elif 30 < rsi < 45: momentum_score = 25
        else:                momentum_score = 0

        # Volume directional (20%)
        vol_surge = latest["volume"] > latest["VMA_20"] * 1.3
        volume_score = 50
        if vol_surge:
            volume_score = 100 if latest["close"] > latest["open"] else 0

        score = (trend_score * 0.40) + (momentum_score * 0.40) + (volume_score * 0.20)

        if score >= 78:   direction = "BUY"
        elif score >= 60: direction = "BUY"
        elif score <= 25: direction = "SELL"
        elif score <= 42: direction = "SELL"
        else:             direction = "HOLD"

        return {
            "direction": direction, "score": round(score, 1),
            "trend": trend_score, "momentum": momentum_score, "volume": volume_score,
            "rsi": round(rsi, 1), "close": latest["close"],
        }

    @staticmethod
    def process_ai_copilot(
        df_5m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_1d: pd.DataFrame,
        is_options_eligible: bool,
        vix: float | None,
        pcr_data: dict,
        security_id: str,
    ) -> dict:

        empty = {
            "regime": "Awaiting Data", "score": 0, "confidence": 0,
            "direction": "HOLD", "option_action": "NEUTRAL",
            "entry": "-", "stop_loss": "-", "target_1": "-", "target_2": "-",
            "reasons": ["Insufficient data."], "vix": vix, "pcr": None,
            "mtf_aligned": False, "mtf": {}, "sr": {}, "vwap_position": None,
            "expiry_days": None, "suppressed": False, "suppress_reason": "",
        }

        if df_5m.empty or "RSI" not in df_5m.columns:
            return empty

        # ── Per-timeframe signals ────────────────────────────────────────────
        sig_5m = AdvancedQuantEngine.compute_signal_single(df_5m)
        sig_1h = AdvancedQuantEngine.compute_signal_single(df_1h) if not df_1h.empty else {"direction": "HOLD", "score": 50}
        sig_1d = AdvancedQuantEngine.compute_signal_single(df_1d) if not df_1d.empty else {"direction": "HOLD", "score": 50}

        mtf = {"5m": sig_5m, "1H": sig_1h, "1D": sig_1d}

        # MTF alignment: all three must agree (or at least 5m + one higher TF)
        directions = [sig_5m["direction"], sig_1h["direction"], sig_1d["direction"]]
        all_agree  = len(set(directions)) == 1 and directions[0] != "HOLD"
        two_agree  = (
            sig_5m["direction"] == sig_1h["direction"] and sig_5m["direction"] != "HOLD"
        ) or (
            sig_5m["direction"] == sig_1d["direction"] and sig_5m["direction"] != "HOLD"
        )
        mtf_aligned = all_agree or two_agree

        # ── Base score from 5m (primary timeframe) ───────────────────────────
        base_score = sig_5m["score"]

        # MTF bonus / penalty
        if all_agree:
            base_score = min(100, base_score + 8)
        elif two_agree:
            base_score = min(100, base_score + 4)
        else:
            base_score = max(0, base_score - 10)  # penalise misalignment

        # ── VIX adjustment ───────────────────────────────────────────────────
        vix_regime = "normal"
        if vix is not None:
            if vix >= VIX_DANGER:
                vix_regime = "danger"
                base_score = max(0, base_score - 15)
            elif vix >= VIX_CAUTION:
                vix_regime = "caution"
                base_score = max(0, base_score - 7)

        # ── PCR adjustment ───────────────────────────────────────────────────
        pcr = pcr_data.get("pcr")
        if pcr is not None:
            pcr_interp = pcr_data.get("interpretation", "Neutral")
            if pcr_interp == "Bullish" and base_score > 50:
                base_score = min(100, base_score + 5)
            elif pcr_interp == "Bearish" and base_score < 50:
                base_score = max(0, base_score - 5)
            elif pcr_interp == "Bullish" and base_score < 50:
                base_score = max(0, base_score - 5)   # contrarian signal

        # ── VWAP position ────────────────────────────────────────────────────
        latest = df_5m.iloc[-1]
        vwap_position = None
        if "VWAP" in df_5m.columns and not pd.isna(latest.get("VWAP")):
            if latest["close"] > latest["VWAP"]:
                vwap_position = "above"
                if base_score > 50:
                    base_score = min(100, base_score + 3)
            else:
                vwap_position = "below"
                if base_score < 50:
                    base_score = max(0, base_score - 3)

        # ── S/R proximity penalty ─────────────────────────────────────────────
        resistances, supports = AdvancedQuantEngine.get_sr_levels(df_5m)
        atr = latest["ATR"]
        near_resistance = any(abs(latest["close"] - r) < atr * 0.3 for r in resistances)
        near_support    = any(abs(latest["close"] - s) < atr * 0.3 for s in supports)

        if near_resistance and base_score > 50:
            base_score = max(50, base_score - 8)   # don't buy into resistance
        if near_support and base_score < 50:
            base_score = min(50, base_score + 8)   # don't sell into support

        # ── Expiry calendar ──────────────────────────────────────────────────
        expiry_days = days_to_expiry(security_id)
        near_expiry = expiry_days is not None and expiry_days <= 1

        # ── Final direction ──────────────────────────────────────────────────
        if base_score >= 78:   regime, direction = "Strong Bullish", "BUY"
        elif base_score >= 60: regime, direction = "Bullish",        "BUY"
        elif base_score <= 25: regime, direction = "Strong Bearish", "SELL"
        elif base_score <= 42: regime, direction = "Bearish",        "SELL"
        else:                  regime, direction = "Sideways",       "HOLD"

        # ── Suppression logic ────────────────────────────────────────────────
        suppressed = False
        suppress_reason = ""
        if vix_regime == "danger" and is_options_eligible:
            suppressed = True
            suppress_reason = f"⚠️ India VIX at {vix:.1f} — options signals suppressed above {VIX_DANGER}. Options premium is too expensive."
        if near_expiry and direction != "HOLD":
            suppressed = True
            suppress_reason = "⚠️ Expiry day — options carry extreme gamma/theta risk. Avoid buying options today."

        if suppressed:
            option_action = "NEUTRAL"
        elif is_options_eligible:
            option_action = "CALL" if direction == "BUY" else ("PUT" if direction == "SELL" else "NEUTRAL")
        else:
            option_action = None

        # ── Confidence ───────────────────────────────────────────────────────
        confidence = min(100, int(abs(base_score - 50) / 50 * 100))
        if not mtf_aligned:
            confidence = max(0, confidence - 20)

        # ── ATR levels with regime-aware multiplier ──────────────────────────
        atr_mult = 2.0 if abs(base_score - 50) > 28 else 1.5
        if vix_regime == "caution":
            atr_mult *= 1.2   # widen SL in high VIX
        if direction == "BUY":
            sl = round(latest["close"] - atr * atr_mult, 2)
            t1 = round(latest["close"] + atr * atr_mult, 2)
            t2 = round(latest["close"] + atr * atr_mult * 2, 2)
        elif direction == "SELL":
            sl = round(latest["close"] + atr * atr_mult, 2)
            t1 = round(latest["close"] - atr * atr_mult, 2)
            t2 = round(latest["close"] - atr * atr_mult * 2, 2)
        else:
            sl = t1 = t2 = "-"

        # ── Reasoning ────────────────────────────────────────────────────────
        reasons = []
        rsi_val = sig_5m.get("rsi", latest.get("RSI", 50))

        if sig_5m["trend"] == 100:
            reasons.append("5m: Price above EMA 20 & 50 — uptrend confirmed.")
        elif sig_5m["trend"] == 0:
            reasons.append("5m: Price below EMA 20 & 50 — downtrend confirmed.")
        else:
            reasons.append("5m: Mixed EMA alignment — no clear trend.")

        if rsi_val >= 70:
            reasons.append(f"RSI {rsi_val} — overbought. Momentum may stall soon.")
        elif rsi_val > 60:
            reasons.append(f"RSI {rsi_val} — healthy momentum expansion.")
        elif rsi_val <= 30:
            reasons.append(f"RSI {rsi_val} — oversold. Watch for bounce.")
        elif rsi_val < 40:
            reasons.append(f"RSI {rsi_val} — weakening momentum.")

        if all_agree:
            reasons.append(f"✅ All 3 timeframes aligned: {direction}. High conviction.")
        elif two_agree:
            reasons.append(f"📊 2/3 timeframes aligned: {direction}.")
        else:
            reasons.append("⚠️ Timeframes conflicted — lower confidence.")

        if vwap_position == "above":
            reasons.append("Price above VWAP — institutional bias is bullish.")
        elif vwap_position == "below":
            reasons.append("Price below VWAP — institutional bias is bearish.")

        if near_resistance:
            reasons.append("⚠️ Price near resistance — longs face overhead supply.")
        if near_support:
            reasons.append("Price near support — shorts face demand zone.")

        if pcr is not None:
            reasons.append(f"PCR {pcr} — options flow signals {pcr_data.get('interpretation','Neutral')} sentiment.")

        if vix is not None:
            if vix_regime == "danger":
                reasons.append(f"🔴 VIX {vix:.1f} — extreme volatility, signals unreliable.")
            elif vix_regime == "caution":
                reasons.append(f"🟡 VIX {vix:.1f} — elevated volatility, widen stops.")
            else:
                reasons.append(f"🟢 VIX {vix:.1f} — normal volatility regime.")

        if near_expiry:
            reasons.append(f"⚠️ {expiry_days}d to expiry — theta/gamma risk very high.")

        if not reasons:
            reasons.append("No clear directional bias — awaiting catalyst.")

        return {
            "regime": regime, "score": int(base_score), "confidence": confidence,
            "direction": direction, "option_action": option_action,
            "entry": str(round(latest["close"], 2)),
            "stop_loss": str(sl), "target_1": str(t1), "target_2": str(t2),
            "reasons": reasons, "vix": vix, "pcr": pcr, "pcr_data": pcr_data,
            "mtf_aligned": mtf_aligned, "mtf": mtf,
            "sr": {"resistances": resistances, "supports": supports},
            "vwap_position": vwap_position,
            "expiry_days": expiry_days, "near_expiry": near_expiry,
            "suppressed": suppressed, "suppress_reason": suppress_reason,
            "is_options_eligible": is_options_eligible,
            "atr": atr,
        }


# ==================== POSITION SIZING ====================
def calculate_position_size(capital: float, risk_pct: float, entry: float, stop_loss: float, lot_size: int = 50) -> dict:
    """
    Fixed-fractional position sizing.
    Returns number of lots, shares/units, and risk amount.
    """
    if entry <= 0 or stop_loss <= 0 or entry == stop_loss:
        return {}
    risk_amount  = capital * (risk_pct / 100)
    risk_per_unit = abs(entry - stop_loss)
    units        = risk_amount / risk_per_unit
    lots         = max(1, int(units / lot_size))
    actual_risk  = lots * lot_size * risk_per_unit
    reward_1     = lots * lot_size * risk_per_unit   # 1:1 default; adjust to T1/T2 externally
    return {
        "lots": lots,
        "units": lots * lot_size,
        "risk_amount": round(actual_risk, 2),
        "risk_per_unit": round(risk_per_unit, 2),
        "capital_at_risk_pct": round(actual_risk / capital * 100, 2),
    }


# ==================== SYNC ====================
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


def sync_data(symbol, seg, inst, name, days_back, interval, cleanup_days, table="live_candles", show_status=True):
    candles = pull_historical_dhan(symbol, seg, inst, days_back, interval)
    if not candles:
        if show_status:
            st.warning("No data returned.")
        return False
    try:
        payload_list = [{
            "symbol": str(symbol), "timestamp": c["timestamp"],
            "open": float(c["open"]), "high": float(c["high"]),
            "low":  float(c["low"]), "close": float(c["close"]),
            "volume": int(c["volume"]),
        } for c in candles]
        supabase.table(table).upsert(payload_list, on_conflict="symbol,timestamp").execute()
        cutoff = (pd.Timestamp.now() - pd.Timedelta(days=cleanup_days)).isoformat()
        supabase.table(table).delete().eq("symbol", str(symbol)).lt("timestamp", cutoff).execute()
        if show_status:
            st.success(f"✅ Synced {len(payload_list)} candles")
        return True
    except Exception as e:
        if show_status:
            st.error(f"❌ Supabase: {e.args[0] if e.args else e}")
        return False


def fetch_df(symbol, days_back_rows=300, table="live_candles") -> pd.DataFrame:
    res = supabase.table(table).select("*").eq("symbol", str(symbol)) \
        .order("timestamp", desc=True).limit(days_back_rows).execute()
    if not res.data:
        return pd.DataFrame()
    df = pd.DataFrame(res.data).iloc[::-1].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def auto_sync_if_stale(active_sec_id, active_seg, active_inst, display_name,
                        days_back, interval, cleanup_days, auto_sync_mins, mode_label, data_mode):
    # Check the correct table for this mode
    target_table = "candles_1d" if data_mode == "Daily (Swing)" else "live_candles"
    sync_key = f"last_sync_{active_sec_id}_{target_table}"
    last_sync = st.session_state.get(sync_key)
    if last_sync and (datetime.now() - last_sync).total_seconds() < auto_sync_mins * 60:
        return
    try:
        res = supabase.table(target_table).select("timestamp") \
            .eq("symbol", str(active_sec_id)).order("timestamp", desc=True).limit(1).execute()
    except Exception:
        return
    should_sync = not res.data
    if not should_sync:
        latest_ts = pd.to_datetime(res.data[0]["timestamp"])
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("Asia/Kolkata")
        age = datetime.now().astimezone() - latest_ts
        if age > timedelta(minutes=auto_sync_mins):
            should_sync = True
            st.info(f"⏰ Data {int(age.total_seconds()//60)}m old — auto-syncing…")
    if should_sync:
        ok = sync_data(str(active_sec_id), active_seg, active_inst, display_name,
                       days_back, interval, cleanup_days,
                       table=target_table, show_status=True)
        if ok:
            st.session_state[sync_key] = datetime.now()
    else:
        st.session_state[sync_key] = datetime.now()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚡ Asset Selection")
    search_mode = st.radio("Method", ["Pre‑defined", "Custom"], index=0, horizontal=True)
    if search_mode == "Pre‑defined":
        selected_name = st.selectbox("Instrument", list(PRESET_ASSETS.keys()))
        active_sec_id = PRESET_ASSETS[selected_name]["id"]
        active_seg    = PRESET_ASSETS[selected_name]["seg"]
        active_inst   = PRESET_ASSETS[selected_name]["inst"]
        display_name  = selected_name
    else:
        active_sec_id = st.text_input("Security ID", value="13")
        active_seg    = st.selectbox("Segment", ["IDX_I", "NSE_EQ", "NSE_FNO", "BSE_EQ"])
        active_inst   = st.selectbox("Type", ["INDEX", "EQUITY", "OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"])
        display_name  = f"ID: {active_sec_id}"

    st.divider()
    data_mode = st.radio("📊 Chart Mode", ["Daily (Swing)", "Intraday (Live)"], index=1)
    if data_mode == "Daily (Swing)":
        days_back, cleanup_days, interval, auto_sync_mins, mode_label = 100, 100, None, 60, "Daily"
    else:
        days_back, cleanup_days, interval, auto_sync_mins, mode_label = 7, 7, "5", 5, "Intraday (5m)"

    is_options_eligible = active_seg in OPTIONS_SEGMENTS

    st.divider()
    st.markdown("### 💰 Position Sizing")
    capital    = st.number_input("Capital (₹)", value=500_000, step=10_000, min_value=10_000)
    risk_pct   = st.slider("Risk per trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.25)
    lot_size   = st.number_input("Lot size", value=50, step=1, min_value=1)

    st.divider()
    st.markdown("### 🔔 Alerts")
    alert_threshold = st.slider("Telegram alert threshold (confidence %)", 50, 95, 70)
    enable_alerts   = st.checkbox("Enable Telegram alerts", value=bool(TELEGRAM_BOT_TOKEN))
    enable_journal  = st.checkbox("Log signals to journal", value=True)

    st.divider()
    sync_clicked = st.button("🔄 Sync Now", use_container_width=True)
    if st.button("🔄 Sync All Timeframes", use_container_width=True):
        with st.spinner("Syncing 5m, 1H, 1D…"):
            # 5m — always 7 days intraday
            ok5  = sync_data(str(active_sec_id), active_seg, active_inst, display_name,
                             7, "5", 7, table="live_candles")
            # 1H — always 30 days of 60-min candles regardless of current mode
            ok1h = sync_data(str(active_sec_id), active_seg, active_inst, display_name,
                             30, "60", 30, table="candles_1h")
            # 1D — always 200 days of daily candles; needs 150+ for reliable EMA 50
            ok1d = sync_data(str(active_sec_id), active_seg, active_inst, display_name,
                             200, None, 200, table="candles_1d")
            # Clear session sync stamp so auto-sync doesn't skip on next rerun
            for key in list(st.session_state.keys()):
                if key.startswith("last_sync_"):
                    del st.session_state[key]
            if ok5 and ok1h and ok1d:
                st.success("✅ All 3 timeframes synced successfully")

    if is_options_eligible:
        st.caption("✅ Options signals enabled")
    else:
        st.caption("ℹ️ Equity — directional only")


# ==================== MANUAL SYNC ====================
if sync_clicked:
    if data_mode == "Daily (Swing)":
        # Daily mode: refresh the 1D candles table
        sync_data(str(active_sec_id), active_seg, active_inst, display_name,
                  200, None, 200, table="candles_1d")
    else:
        # Intraday mode: refresh the 5m candles table
        sync_data(str(active_sec_id), active_seg, active_inst, display_name,
                  7, "5", 7, table="live_candles")

# Auto-sync targets the mode-appropriate primary table
if data_mode == "Daily (Swing)":
    auto_sync_if_stale(active_sec_id, active_seg, active_inst, display_name,
                       200, None, 200, auto_sync_mins, mode_label, data_mode)
else:
    auto_sync_if_stale(active_sec_id, active_seg, active_inst, display_name,
                       7, "5", 7, auto_sync_mins, mode_label, data_mode)

# ==================== LOAD DATA ====================
# Table layout (fixed, mode-independent):
#   live_candles  → 5m intraday candles  (7 days)
#   candles_1h    → 1H candles           (30 days)
#   candles_1d    → daily candles        (200 days)
#
# Daily mode uses candles_1d for its chart — NOT live_candles.
# This prevents the two modes from overwriting each other in the same table.

with st.spinner("Loading data…"):
    df_5m = fetch_df(active_sec_id, 300, "live_candles")   # 5m always
    df_1h = fetch_df(active_sec_id, 200, "candles_1h")     # 1H always
    df_1d = fetch_df(active_sec_id, 150, "candles_1d")     # daily always

    # Pick the primary display frame based on mode
    # Daily mode → show daily candles on chart; Intraday → show 5m candles
    df_primary = df_1d if data_mode == "Daily (Swing)" else df_5m

    mtf_warning = None
    missing = []
    if df_5m.empty:  missing.append("5m (click Sync Now)")
    if df_1h.empty:  missing.append("1H")
    if df_1d.empty:  missing.append("1D")

    if missing:
        mtf_warning = (
            f"⚠️ Missing data: {', '.join(missing)}. "
            f"Click **Sync All Timeframes** to populate all tables. "
            f"In Daily mode only 1D data is needed for the chart; MTF needs all three."
        )

    for _df in [df_5m, df_1h, df_1d]:
        if not _df.empty:
            AdvancedQuantEngine.compute_indicators(_df)

    vix      = fetch_india_vix()
    pcr_data = fetch_option_chain_pcr(active_sec_id) if is_options_eligible else {}

# NSE option chain used for PCR — no debug expander needed

copilot = AdvancedQuantEngine.process_ai_copilot(
    df_primary, df_1h, df_1d,
    is_options_eligible=is_options_eligible,
    vix=vix, pcr_data=pcr_data,
    security_id=str(active_sec_id),
)

# ── Journal & alert (fire once per session per signal change) ─────────────────
sig_key = f"last_signal_{active_sec_id}"
last_sig = st.session_state.get(sig_key, {})
signal_changed = last_sig.get("direction") != copilot["direction"] or \
                 abs(last_sig.get("score", 0) - copilot["score"]) > 5

if signal_changed:
    st.session_state[sig_key] = copilot
    if enable_journal:
        log_signal_to_journal(copilot, display_name, mode_label)
    if enable_alerts and copilot["confidence"] >= alert_threshold and not copilot["suppressed"]:
        msg = (
            f"*⚡ AI Options Copilot — {display_name}*\n"
            f"Signal: *{copilot.get('option_action') or copilot['direction']}*\n"
            f"Regime: {copilot['regime']} | Score: {copilot['score']}% | Conf: {copilot['confidence']}%\n"
            f"Entry: {copilot['entry']} | SL: {copilot['stop_loss']} | T1: {copilot['target_1']}\n"
            f"VIX: {vix:.1f if vix else 'N/A'} | PCR: {copilot['pcr'] or 'N/A'}\n"
            f"MTF aligned: {'✅' if copilot['mtf_aligned'] else '❌'}"
        )
        send_telegram_alert(msg)

# ==================== DASHBOARD ====================
if mtf_warning:
    st.warning(mtf_warning)

st.markdown("<h1 style='margin-bottom:0;'>⚡ AI Options Copilot <sup style='font-size:0.5em;color:#58a6ff'>v4</sup></h1>", unsafe_allow_html=True)
st.caption(
    f"**{display_name}** · {mode_label} · "
    f"VIX: **{f'{vix:.1f}' if vix else 'N/A'}** · "
    f"PCR: **{copilot['pcr'] or 'N/A'}** · "
    f"MTF: **{'✅ Aligned' if copilot['mtf_aligned'] else '❌ Conflicted'}**"
)

if df_5m.empty:
    st.info("📭 No data. Use sidebar to sync.")
    st.stop()

# Suppression banner
if copilot["suppressed"]:
    st.error(copilot["suppress_reason"])

col_left, col_right = st.columns([3, 1.2], gap="large")

# ── LEFT: Chart ───────────────────────────────────────────────────────────────
with col_left:
    tab_chart, tab_journal, tab_sizing = st.tabs(["📈 Chart", "📓 Signal Journal", "💰 Position Sizing"])

    with tab_chart:
        chart_df    = df_primary if not df_primary.empty else (df_1d if not df_1d.empty else pd.DataFrame())
        chart_label = "Daily" if data_mode == "Daily (Swing)" else "5m"
        if chart_df.empty:
            st.warning("No chart data. Click Sync All Timeframes.")
            st.stop()
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=(f"Price Action + VWAP ({chart_label})", "Volume", "RSI"),
        )
        # Candles
        fig.add_trace(go.Candlestick(
            x=chart_df["timestamp"],
            open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=chart_df["close"],
            name="Candles", increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        ), row=1, col=1)

        # EMAs
        for col_name, color, label in [("EMA_9","#d2a8ff","EMA 9"),("EMA_20","#ffa657","EMA 20"),("EMA_50","#58a6ff","EMA 50")]:
            if col_name in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df[col_name], name=label,
                                         line=dict(color=color, width=1.5)), row=1, col=1)

        # VWAP + bands
        if "VWAP" in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["VWAP"], name="VWAP",
                                     line=dict(color="#e3b341", width=2, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["VWAP_U1"], name="VWAP +1σ",
                                     line=dict(color="rgba(227,179,65,0.35)", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["VWAP_L1"], name="VWAP -1σ",
                                     line=dict(color="rgba(227,179,65,0.35)", width=1, dash="dash"),
                                     fill="tonexty", fillcolor="rgba(227,179,65,0.04)"), row=1, col=1)

        # S/R horizontal lines
        for r in copilot["sr"].get("resistances", [])[:2]:
            fig.add_hline(y=r, line_dash="dot", line_color="rgba(248,81,73,0.4)", line_width=1, row=1, col=1)
        for s in copilot["sr"].get("supports", [])[:2]:
            fig.add_hline(y=s, line_dash="dot", line_color="rgba(46,160,67,0.4)", line_width=1, row=1, col=1)

        # Volume
        colors = ["#3fb950" if chart_df["close"].iloc[i] >= chart_df["open"].iloc[i] else "#f85149" for i in range(len(df_5m))]
        fig.add_trace(go.Bar(x=chart_df["timestamp"], y=chart_df["volume"], name="Volume", marker_color=colors), row=2, col=1)
        if "VMA_20" in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["VMA_20"], name="Vol MA",
                                     line=dict(color="#8b949e", width=1)), row=2, col=1)

        # RSI
        if "RSI" in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["RSI"], name="RSI",
                                     line=dict(color="#58a6ff", width=1.5)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,81,73,0.5)",  line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="rgba(46,160,67,0.5)",  line_width=1, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(139,148,158,0.3)", line_width=1, row=3, col=1)

        fig.update_layout(
            template="plotly_dark", height=620,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_journal:
        st.markdown("### 📓 Signal Journal")
        st.caption("Signals are logged automatically when direction or score changes significantly.")
        try:
            journal_res = supabase.table("signal_journal") \
                .select("*").order("created_at", desc=True).limit(50).execute()
            if journal_res.data:
                jdf = pd.DataFrame(journal_res.data)
                cols_show = ["created_at", "symbol", "direction", "option_action",
                             "score", "confidence", "regime", "entry", "stop_loss",
                             "target_1", "vix", "pcr", "mtf_aligned", "outcome"]
                jdf = jdf[[c for c in cols_show if c in jdf.columns]]
                st.dataframe(jdf, use_container_width=True, hide_index=True)

                # Outcome updater
                st.markdown("#### Update signal outcome")
                if "id" in pd.DataFrame(journal_res.data).columns:
                    ids = [str(r["id"]) for r in journal_res.data if r.get("outcome") == "pending"]
                    if ids:
                        sel_id = st.selectbox("Signal ID to update", ids)
                        outcome = st.selectbox("Outcome", ["win", "loss", "breakeven", "pending"])
                        if st.button("Update outcome"):
                            supabase.table("signal_journal").update({"outcome": outcome}) \
                                .eq("id", int(sel_id)).execute()
                            st.success("Updated!")
                            st.rerun()

                # Win rate
                if "outcome" in jdf.columns:
                    resolved = jdf[jdf["outcome"].isin(["win","loss","breakeven"])]
                    if not resolved.empty:
                        wins = (resolved["outcome"] == "win").sum()
                        total = len(resolved)
                        st.metric("Win rate", f"{wins/total*100:.1f}%", f"{wins}/{total} resolved signals")
            else:
                st.info("No signals logged yet. The journal fills automatically as signals generate.")
        except Exception as e:
            st.warning(f"Journal table not found. Create it using the DDL in the source code comments. ({e})")

    with tab_sizing:
        st.markdown("### 💰 Position Size Calculator")
        if copilot["stop_loss"] != "-" and copilot["entry"] != "-":
            try:
                entry_f = float(copilot["entry"])
                sl_f    = float(copilot["stop_loss"])
                sizing  = calculate_position_size(capital, risk_pct, entry_f, sl_f, lot_size)
                if sizing:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Lots", sizing["lots"])
                    c2.metric("Units", f"{sizing['units']:,}")
                    c3.metric("Risk (₹)", f"₹{sizing['risk_amount']:,.0f}")
                    c4.metric("Capital at risk", f"{sizing['capital_at_risk_pct']}%")

                    reward_t1 = round(abs(float(copilot["target_1"]) - entry_f) * sizing["units"], 2) if copilot["target_1"] != "-" else "-"
                    reward_t2 = round(abs(float(copilot["target_2"]) - entry_f) * sizing["units"], 2) if copilot["target_2"] != "-" else "-"
                    rr1 = round(reward_t1 / sizing["risk_amount"], 2) if sizing["risk_amount"] else "-"
                    rr2 = round(reward_t2 / sizing["risk_amount"], 2) if sizing["risk_amount"] else "-"

                    st.markdown(f"""
                    <div class='glass-card'>
                        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;'>
                            <div><span class='metric-label'>Risk/unit</span><div style='font-weight:600;'>₹{sizing['risk_per_unit']}</div></div>
                            <div><span class='metric-label'>Reward T1</span><div style='font-weight:600;color:#3fb950;'>₹{reward_t1:,}</div></div>
                            <div><span class='metric-label'>Reward T2</span><div style='font-weight:600;color:#58a6ff;'>₹{reward_t2:,}</div></div>
                            <div><span class='metric-label'>R:R T1</span><div style='font-weight:600;'>1 : {rr1}</div></div>
                            <div><span class='metric-label'>R:R T2</span><div style='font-weight:600;'>1 : {rr2}</div></div>
                            <div><span class='metric-label'>Lot size</span><div style='font-weight:600;'>{lot_size}</div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if sizing["capital_at_risk_pct"] > 2:
                        st.warning("⚠️ Risk exceeds 2% of capital. Consider reducing lot size.")
            except Exception:
                st.info("Set Entry & SL first via a sync.")
        else:
            st.info("Sync data first to populate entry and stop-loss.")

# ── RIGHT: Signal panel ───────────────────────────────────────────────────────
with col_right:

    # ── MTF alignment ─────────────────────────────────────────────────────────
    st.markdown("### 🕐 Timeframe Alignment")
    mtf = copilot.get("mtf", {})
    def mtf_class(d):
        return "mtf-bull" if d=="BUY" else ("mtf-bear" if d=="SELL" else "mtf-neut")
    def mtf_label(d, s):
        return f"{d}<br><small>{s}%</small>"

    st.markdown(f"""
    <div class='glass-card' style='padding:0.8rem;'>
        <div class='mtf-row'>
            <div class='mtf-pill {mtf_class(mtf.get("5m",{}).get("direction","HOLD"))}'>
                5m<br><b>{mtf.get("5m",{}).get("direction","—")}</b><br>
                <small>{mtf.get("5m",{}).get("score","-")}%</small>
            </div>
            <div class='mtf-pill {mtf_class(mtf.get("1H",{}).get("direction","HOLD"))}'>
                1H<br><b>{mtf.get("1H",{}).get("direction","—")}</b><br>
                <small>{mtf.get("1H",{}).get("score","-")}%</small>
            </div>
            <div class='mtf-pill {mtf_class(mtf.get("1D",{}).get("direction","HOLD"))}'>
                1D<br><b>{mtf.get("1D",{}).get("direction","—")}</b><br>
                <small>{mtf.get("1D",{}).get("score","-")}%</small>
            </div>
        </div>
        <div style='font-size:0.78rem;color:{"#3fb950" if copilot["mtf_aligned"] else "#f85149"};text-align:center;'>
            {"✅ Timeframes aligned — higher conviction" if copilot["mtf_aligned"] else "❌ Timeframes conflicted — lower conviction"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main signal card ──────────────────────────────────────────────────────
    if is_options_eligible:
        st.markdown("### 🎯 Option Strategy")
        oa = copilot["option_action"]
        if oa == "CALL":     badge_html = "<span class='badge-call'>BUY CALL</span>"
        elif oa == "PUT":    badge_html = "<span class='badge-put'>BUY PUT</span>"
        else:                badge_html = "<span class='badge-neutral'>NO TRADE</span>"
    else:
        st.markdown("### 📡 Directional Signal")
        d = copilot["direction"]
        if d == "BUY":       badge_html = "<span class='badge-buy'>BUY</span>"
        elif d == "SELL":    badge_html = "<span class='badge-sell'>SELL</span>"
        else:                badge_html = "<span class='badge-hold'>HOLD</span>"

    st.markdown(f"""
    <div class="glass-card" style="text-align:center;padding:1.5rem 1rem;">
        <div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.4rem;">
            {"OPTION ACTION" if is_options_eligible else "DIRECTIONAL BIAS"}
        </div>
        <div style="margin:0.4rem 0;">{badge_html}</div>
        <div style="display:flex;justify-content:center;gap:1.5rem;margin-top:0.8rem;">
            <div><span class="metric-label">Score</span>
                 <div style="font-size:1.6rem;font-weight:700;">{copilot['score']}%</div></div>
            <div><span class="metric-label">Conf.</span>
                 <div style="font-size:1.6rem;font-weight:700;">{copilot['confidence']}%</div></div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{copilot['confidence']}%;"></div>
        </div>
        <div style="margin-top:0.6rem;font-size:0.78rem;color:#8b949e;">{copilot['regime']}</div>
    </div>
    """, unsafe_allow_html=True)

    if not is_options_eligible:
        st.markdown("<div class='info-note'>ℹ️ Equity — use Index/FNO for options signals.</div>", unsafe_allow_html=True)

    # ── VIX & PCR ──────────────────────────────────────────────────────────────
    vix_class  = "vix-ok" if (vix or 0) < VIX_SAFE else ("vix-warn" if (vix or 0) < VIX_CAUTION else "vix-danger")
    vix_label  = f"{vix:.1f}" if vix else "N/A"
    vix_regime_label = "🔴 Danger" if vix and vix >= VIX_DANGER else ("🟡 Caution" if vix and vix >= VIX_CAUTION else "🟢 Normal")
    pcr_val    = copilot.get("pcr")
    pcr_interp = copilot.get("pcr_data", {}).get("interpretation", "N/A")
    pcr_class  = "pcr-bull" if pcr_interp == "Bullish" else ("pcr-bear" if pcr_interp == "Bearish" else "pcr-neut")
    max_pain   = copilot.get("pcr_data", {}).get("max_pain")
    expiry_days = copilot.get("expiry_days")
    near_expiry = copilot.get("near_expiry", False)

    # Build optional rows cleanly — no inline ternary string concat inside f-strings
    max_pain_row = (
        f"<div><span class='metric-label'>Max Pain</span>"
        f"<div style='font-weight:600;'>₹{max_pain:,.0f}</div></div>"
        if max_pain else ""
    )
    expiry_color = "color:#f85149;" if near_expiry else ""
    expiry_label = copilot.get("pcr_data", {}).get("expiry", "")
    expiry_row = (
        f"<div><span class='metric-label'>Expiry ({expiry_label})</span>"
        f"<div style='font-weight:600;{expiry_color}'>{expiry_days}d away</div></div>"
        if expiry_days else ""
    )

    # Top OI levels from NSE option chain
    top_call_res = copilot.get("pcr_data", {}).get("top_call_resistance", [])
    top_put_sup  = copilot.get("pcr_data", {}).get("top_put_support", [])
    call_res_str = " · ".join([f"₹{int(s):,}" for s in top_call_res]) if top_call_res else "N/A"
    put_sup_str  = " · ".join([f"₹{int(s):,}" for s in top_put_sup])  if top_put_sup  else "N/A"
    oi_levels_row = (
        f"<div style='grid-column:span 2; margin-top:0.3rem; border-top:1px solid rgba(255,255,255,0.06); padding-top:0.4rem;'>"
        f"<span class='metric-label'>Call OI walls (resistance)</span>"
        f"<div style='font-size:0.8rem;color:#f85149;font-weight:600;'>{call_res_str}</div>"
        f"<span class='metric-label' style='margin-top:0.3rem;display:block;'>Put OI walls (support)</span>"
        f"<div style='font-size:0.8rem;color:#3fb950;font-weight:600;'>{put_sup_str}</div>"
        f"</div>"
        if top_call_res or top_put_sup else ""
    )

    st.markdown(f"""
    <div class="glass-card">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
            <div><span class="metric-label">India VIX</span>
                 <div class="{vix_class}">{vix_label}</div></div>
            <div><span class="metric-label">VIX regime</span>
                 <div class="{vix_class}">{vix_regime_label}</div></div>
            <div><span class="metric-label">PCR</span>
                 <div class="{pcr_class}">{pcr_val or "N/A"}</div></div>
            <div><span class="metric-label">OI sentiment</span>
                 <div class="{pcr_class}">{pcr_interp}</div></div>
            {max_pain_row}
            {expiry_row}
            {oi_levels_row}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Trade levels ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Trade Levels")
    vwap_pos = copilot.get("vwap_position")
    st.markdown(f"""
    <div class="glass-card">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
            <div><span class="metric-label">Entry</span>
                 <div style="font-weight:600;">{copilot['entry']}</div></div>
            <div><span class="metric-label">Stop Loss</span>
                 <div style="font-weight:600;color:#f85149;">{copilot['stop_loss']}</div></div>
            <div><span class="metric-label">Target 1</span>
                 <div style="font-weight:600;color:#3fb950;">{copilot['target_1']}</div></div>
            <div><span class="metric-label">Target 2</span>
                 <div style="font-weight:600;color:#58a6ff;">{copilot['target_2']}</div></div>
        </div>
        <div style="margin-top:0.6rem;font-size:0.78rem;color:#8b949e;">
            VWAP: <span style="color:{'#3fb950' if vwap_pos=='above' else '#f85149' if vwap_pos else '#8b949e'};">
            {"Above ↑" if vwap_pos=="above" else "Below ↓" if vwap_pos=="below" else "N/A"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── S/R levels ────────────────────────────────────────────────────────────
    res_list = copilot["sr"].get("resistances", [])
    sup_list = copilot["sr"].get("supports", [])
    if res_list or sup_list:
        res_tags = "".join([f"<span class='sr-tag sr-res'>R: {round(r,1)}</span>" for r in res_list[:3]])
        sup_tags = "".join([f"<span class='sr-tag sr-sup'>S: {round(s,1)}</span>" for s in sup_list[:3]])
        st.markdown(f"""
        <div class="glass-card">
            <span class="metric-label">Key Levels</span><br>
            <div style="margin-top:0.4rem;">{res_tags}{sup_tags}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Reasoning ────────────────────────────────────────────────────────────
    st.markdown("### 🧠 Reasoning")
    reason_html = "".join([f"<div class='reason-item'>{r}</div>" for r in copilot["reasons"]])
    st.markdown(f"<div class='glass-card'>{reason_html}</div>", unsafe_allow_html=True)

    # ── Key stats ──────────────────────────────────────────────────────────────
    st.markdown("### 📈 Key Stats")
    last = chart_df.iloc[-1] if not chart_df.empty else df_1d.iloc[-1] if not df_1d.empty else pd.Series()
    stats = {
        "High":   round(last["high"], 2),
        "Low":    round(last["low"], 2),
        "Volume": f"{int(last['volume']):,}",
        "RSI":    round(last["RSI"], 1) if "RSI" in chart_df.columns else "-",
        "ATR":    round(last["ATR"], 2) if "ATR" in chart_df.columns else "-",
        "VWAP":   round(last["VWAP"], 2) if "VWAP" in chart_df.columns and not pd.isna(last.get("VWAP")) else "-",
    }
    stat_cols = st.columns(2)
    for i, (k, v) in enumerate(stats.items()):
        with stat_cols[i % 2]:
            st.metric(label=k, value=v)

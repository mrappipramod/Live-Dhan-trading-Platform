import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from supabase import create_client, Client

# ===================== PAGE CONFIG =====================
st.set_page_config(
    layout="wide",
    page_title="AI Options Copilot",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #0d1117 0%, #080B0E 100%);
        color: #e6edf3;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, border-color 0.2s;
    }
    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.3);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
    }
    /* Options badges */
    .badge-call {
        background: rgba(46, 160, 67, 0.25);
        color: #3fb950;
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        border: 1px solid rgba(46, 160, 67, 0.4);
        display: inline-block;
    }
    .badge-put {
        background: rgba(248, 81, 73, 0.25);
        color: #f85149;
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        border: 1px solid rgba(248, 81, 73, 0.4);
        display: inline-block;
    }
    .badge-neutral {
        background: rgba(139, 148, 158, 0.2);
        color: #8b949e;
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        border: 1px solid rgba(139, 148, 158, 0.3);
        display: inline-block;
    }
    /* Equity direction badges */
    .badge-buy {
        background: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        border: 1px solid rgba(46, 160, 67, 0.3);
        display: inline-block;
    }
    .badge-sell {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        border: 1px solid rgba(248, 81, 73, 0.3);
        display: inline-block;
    }
    .badge-hold {
        background: rgba(139, 148, 158, 0.2);
        color: #8b949e;
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        border: 1px solid rgba(139, 148, 158, 0.3);
        display: inline-block;
    }
    .reason-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.9rem;
    }
    .reason-item:last-child { border-bottom: none; }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(46, 160, 67, 0.3);
    }
    .stRadio > div {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.5rem;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #21262d;
        margin-top: 6px;
    }
    .confidence-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #3fb950, #58a6ff);
    }
    .info-note {
        background: rgba(88, 166, 255, 0.08);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
        color: #58a6ff;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SECRETS ====================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_KEY", ""))
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("🔐 Missing Supabase credentials. Please set secrets.")
    st.stop()

if not DHAN_ACCESS_TOKEN:
    st.error("🔐 Missing DHAN_ACCESS_TOKEN. Please set it in secrets. Dhan tokens expire daily — regenerate at https://dhan.co")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== CONSTANTS ====================
# Segments that support options trading
OPTIONS_SEGMENTS = {"IDX_I", "NSE_FNO"}

# Dhan epoch offset correction (documented):
# Dhan API returns timestamps relative to an epoch that pre-dates Unix epoch
# by 315532800 seconds (approx 10 years). Applied only when ts < 2017-07-14.
DHAN_EPOCH_OFFSET = 315532800
DHAN_EPOCH_THRESHOLD = 1_500_000_000  # 2017-07-14 Unix timestamp


# ==================== QUANT ENGINE ====================
class AdvancedQuantEngine:

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators on OHLCV data.
        NOTE: Uses df.iloc[:-1] for rolling calculations to avoid lookahead
        bias from an incomplete live candle, then fills the last row forward.
        """
        if len(df) < 5:
            return df

        # --- FIX: exclude last (potentially incomplete) candle from rolling calc ---
        close = df['close'].copy()
        close_for_calc = close.iloc[:-1]

        # EMAs — computed on full series (EWM handles edge naturally), then assigned
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        # EMA_9 added to chart as fast signal
        df['EMA_9'] = close.ewm(span=9, adjust=False).mean()

        # RSI — exclude last candle then forward-fill
        delta = close_for_calc.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_series = rsi_series.reindex(df.index).ffill().fillna(50)
        df['RSI'] = rsi_series

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean().fillna(df['close'] * 0.005)

        # Volume MA
        df['VMA_20'] = df['volume'].rolling(window=20).mean().fillna(df['volume'])

        return df

    @staticmethod
    def process_ai_copilot(df: pd.DataFrame, is_options_eligible: bool = False) -> dict:
        """
        Generate signal from indicators.
        is_options_eligible: True for IDX_I / NSE_FNO segments only.
        """
        empty = {
            "regime": "Awaiting Data",
            "score": 0,
            "direction": "HOLD",
            "option_action": "NEUTRAL",
            "entry": "-",
            "stop_loss": "-",
            "target_1": "-",
            "target_2": "-",
            "reasons": ["Insufficient data."],
            "confidence": 0,
            "is_options_eligible": is_options_eligible,
        }
        if df.empty or 'RSI' not in df.columns:
            return empty

        latest = df.iloc[-1]

        # ── 1. Trend (40%) ──────────────────────────────────────────────────
        trend_score = 50
        if latest['close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
            trend_score = 100
        elif latest['close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
            trend_score = 0

        # ── 2. Momentum / RSI (40%) ─────────────────────────────────────────
        # Correctly penalises overbought (≥70) and rewards healthy momentum (50–70)
        rsi = latest['RSI']
        if 50 < rsi < 70:
            momentum_score = 90   # healthy expansion
        elif rsi >= 70:
            momentum_score = 20   # overbought — caution
        elif 45 <= rsi <= 50:
            momentum_score = 50   # neutral
        elif 30 < rsi < 45:
            momentum_score = 25   # weakening
        elif rsi <= 30:
            momentum_score = 0    # oversold / bearish momentum

        # ── 3. Volume — directional only (20%) ─────────────────────────────
        # Volume alone is not bullish; only a surge on a green candle is.
        volume_score = 50
        vol_surge = latest['volume'] > latest['VMA_20'] * 1.3
        if vol_surge:
            if latest['close'] > latest['open']:
                volume_score = 100   # bullish surge
            else:
                volume_score = 0     # bearish surge

        final_score = (trend_score * 0.40) + (momentum_score * 0.40) + (volume_score * 0.20)

        # ── Direction ───────────────────────────────────────────────────────
        if final_score >= 78:
            regime, direction = "Strong Bullish", "BUY"
        elif final_score >= 60:
            regime, direction = "Bullish", "BUY"
        elif final_score <= 25:
            regime, direction = "Strong Bearish", "SELL"
        elif final_score <= 42:
            regime, direction = "Bearish", "SELL"
        else:
            regime, direction = "Sideways", "HOLD"

        # ── Options action — only for eligible segments ──────────────────────
        if is_options_eligible:
            if direction == "BUY":
                option_action = "CALL"
            elif direction == "SELL":
                option_action = "PUT"
            else:
                option_action = "NEUTRAL"
        else:
            option_action = None  # not applicable

        # ── Confidence ───────────────────────────────────────────────────────
        confidence = min(100, int(abs(final_score - 50) / 50 * 100))

        # ── ATR-scaled levels (multiplier varies by regime strength) ─────────
        atr = latest['ATR']
        # Use wider multiplier in strong regimes, tighter in moderate
        atr_mult = 2.0 if abs(final_score - 50) > 28 else 1.5
        if direction == "BUY":
            sl = round(latest['close'] - (atr * atr_mult), 2)
            t1 = round(latest['close'] + (atr * atr_mult), 2)
            t2 = round(latest['close'] + (atr * atr_mult * 2), 2)
        elif direction == "SELL":
            sl = round(latest['close'] + (atr * atr_mult), 2)
            t1 = round(latest['close'] - (atr * atr_mult), 2)
            t2 = round(latest['close'] - (atr * atr_mult * 2), 2)
        else:
            sl = t1 = t2 = "-"

        # ── Reasoning ────────────────────────────────────────────────────────
        reasons = []
        if trend_score == 100:
            reasons.append("Price above EMA 20 & 50 – confirmed uptrend structure.")
        elif trend_score == 0:
            reasons.append("Price below EMA 20 & 50 – confirmed downtrend structure.")
        else:
            reasons.append("Price mixed vs EMAs – no clear trend alignment.")

        if rsi >= 70:
            reasons.append(f"RSI {round(rsi,1)} – overbought zone, momentum may stall.")
        elif rsi > 60:
            reasons.append(f"RSI {round(rsi,1)} – momentum expanding, healthy.")
        elif rsi <= 30:
            reasons.append(f"RSI {round(rsi,1)} – oversold zone, watch for reversal.")
        elif rsi < 40:
            reasons.append(f"RSI {round(rsi,1)} – momentum weakening.")

        if vol_surge:
            if volume_score == 100:
                reasons.append("Volume surge on green candle – buying pressure confirmed.")
            else:
                reasons.append("⚠️ Volume surge on red candle – selling pressure detected.")

        if not reasons:
            reasons.append("No clear directional bias – awaiting catalyst.")

        return {
            "regime": regime,
            "score": int(final_score),
            "direction": direction,
            "option_action": option_action,
            "entry": f"{round(latest['close'], 2)}",
            "stop_loss": str(sl),
            "target_1": str(t1),
            "target_2": str(t2),
            "reasons": reasons,
            "confidence": confidence,
            "is_options_eligible": is_options_eligible,
        }


# ==================== DHAN API ====================
def pull_historical_dhan(
    security_id: str,
    exchange_segment: str,
    instrument_type: str,
    days_back: int,
    interval: str = None
):
    url = "https://api.dhan.co/v2/charts/historical"
    headers = {
        "access-token": DHAN_ACCESS_TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument_type,
        "expiryCode": 0,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    if interval:
        payload["interval"] = interval

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)

        # Surface auth errors clearly
        if res.status_code in (401, 403):
            st.error(
                "🔑 Dhan API auth failed (401/403). Your access token may have expired. "
                "Regenerate it at https://dhan.co and update DHAN_ACCESS_TOKEN in secrets."
            )
            return []
        if res.status_code != 200:
            st.error(f"Dhan API error {res.status_code}: {res.text[:200]}")
            return []

        data = res.json()
        timestamps = data.get("timestamp", data.get("start_Time", []))
        if not timestamps:
            return []

        opens   = data.get("open",   [])
        highs   = data.get("high",   [])
        lows    = data.get("low",    [])
        closes  = data.get("close",  [])
        volumes = data.get("volume", [])

        candles = []
        MIN_VALID_TS = pd.Timestamp("2015-01-01").timestamp()
        MAX_VALID_TS = pd.Timestamp("2035-01-01").timestamp()

        for i in range(len(timestamps)):
            ts = int(timestamps[i])
            # Documented Dhan epoch correction (pre-2017 raw values use a shifted epoch)
            if ts < DHAN_EPOCH_THRESHOLD:
                ts += DHAN_EPOCH_OFFSET

            # Sanity check — warn if timestamp still looks wrong
            if not (MIN_VALID_TS <= ts <= MAX_VALID_TS):
                st.warning(f"Suspicious timestamp after correction: {ts}. Skipping row {i}.")
                continue

            dt_str = (
                pd.to_datetime(ts, unit="s", utc=True)
                .tz_convert("Asia/Kolkata")
                .isoformat()
            )
            candles.append({
                "timestamp": dt_str,
                "open":   opens[i],
                "high":   highs[i],
                "low":    lows[i],
                "close":  closes[i],
                "volume": volumes[i],
            })
        return candles

    except requests.exceptions.Timeout:
        st.error("Dhan API timed out. Check your connection.")
    except Exception as e:
        st.error(f"Network error: {e}")
    return []


# ==================== PRESETS ====================
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
        active_sec_id = st.text_input("Security ID", value="3456")
        active_seg    = st.selectbox("Segment", ["NSE_EQ", "IDX_I", "NSE_FNO", "BSE_EQ"])
        active_inst   = st.selectbox("Type", ["EQUITY", "INDEX", "OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"])
        display_name  = f"ID: {active_sec_id}"

    st.divider()

    data_mode = st.radio(
        "📊 Chart Mode",
        ["Daily (Swing)", "Intraday (Live)"],
        index=0,
        help="Daily: 100 days, EMA 50 reliable. Intraday: 5-min candles, 7 days.",
    )

    if data_mode == "Daily (Swing)":
        days_back        = 100
        cleanup_days     = 100
        interval         = None
        auto_sync_mins   = 60
        mode_label       = "Daily"
    else:
        days_back        = 7
        cleanup_days     = 7
        interval         = "5"
        auto_sync_mins   = 5
        mode_label       = "Intraday (5m)"

    # Whether this instrument supports options signals
    is_options_eligible = active_seg in OPTIONS_SEGMENTS

    st.caption(f"Mode: **{mode_label}** · Last **{cleanup_days} days**")
    if is_options_eligible:
        st.caption("✅ Options signals enabled (Index/FNO)")
    else:
        st.caption("ℹ️ Equity — directional signals only")

    st.divider()
    sync_clicked = st.button("🔄 Sync Now", use_container_width=True)


# ==================== SYNC FUNCTION ====================
def sync_data(symbol, seg, inst, name, days_back, interval, cleanup_days, show_status=True):
    with st.spinner(f"Fetching {name} ({mode_label})..."):
        candles = pull_historical_dhan(symbol, seg, inst, days_back, interval)
        if not candles:
            if show_status:
                st.warning("No data returned. Check your token or market hours.")
            return False
        try:
            payload_list = [
                {
                    "symbol":    str(symbol),
                    "timestamp": c["timestamp"],
                    "open":      float(c["open"]),
                    "high":      float(c["high"]),
                    "low":       float(c["low"]),
                    "close":     float(c["close"]),
                    "volume":    int(c["volume"]),
                }
                for c in candles
            ]
            supabase.table("live_candles").upsert(
                payload_list, on_conflict="symbol,timestamp"
            ).execute()

            cutoff = (pd.Timestamp.now() - pd.Timedelta(days=cleanup_days)).isoformat()
            supabase.table("live_candles") \
                .delete() \
                .eq("symbol", str(symbol)) \
                .lt("timestamp", cutoff) \
                .execute()

            if show_status:
                st.success(f"✅ Synced {len(payload_list)} candles (cleaned > {cleanup_days}d)")
            return True
        except Exception as e:
            err = e.args[0] if e.args else str(e)
            if show_status:
                st.error(f"❌ Supabase error: {err}")
            return False


# ==================== MANUAL SYNC ====================
if sync_clicked:
    sync_data(
        str(active_sec_id), active_seg, active_inst,
        display_name, days_back, interval, cleanup_days
    )


# ==================== AUTO-SYNC (session-state gated) ====================
def auto_sync_if_stale():
    """
    FIX: Uses st.session_state to prevent re-firing on every Streamlit rerender.
    Streamlit reruns on every widget interaction; without this gate, the Dhan
    API would be called on every dropdown change, risking rate limit bans.
    """
    sync_key = f"last_sync_{active_sec_id}_{data_mode}"
    last_sync: datetime | None = st.session_state.get(sync_key)

    # Check if we need to hit the API at all this session
    if last_sync is not None:
        secs_since = (datetime.now() - last_sync).total_seconds()
        if secs_since < auto_sync_mins * 60:
            return  # Already synced recently in this session — skip

    # Check Supabase for the freshest candle timestamp
    try:
        latest_check = (
            supabase.table("live_candles")
            .select("timestamp")
            .eq("symbol", str(active_sec_id))
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
    except Exception:
        return  # Supabase unreachable — fail silently, user can manual sync

    should_sync = False
    if not latest_check.data:
        should_sync = True
    else:
        latest_ts = pd.to_datetime(latest_check.data[0]["timestamp"])
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("Asia/Kolkata")
        age = datetime.now().astimezone() - latest_ts
        if age > timedelta(minutes=auto_sync_mins):
            should_sync = True
            st.info(f"⏰ Data is {int(age.total_seconds()//60)} min old. Auto-syncing...")

    if should_sync:
        ok = sync_data(
            str(active_sec_id), active_seg, active_inst,
            display_name, days_back, interval, cleanup_days,
            show_status=True,
        )
        if ok:
            st.session_state[sync_key] = datetime.now()
    else:
        # Data is fresh — still stamp the session so we don't keep querying Supabase
        st.session_state[sync_key] = datetime.now()


auto_sync_if_stale()


# ==================== FETCH DATA ====================
response = (
    supabase.table("live_candles")
    .select("*")
    .eq("symbol", str(active_sec_id))
    .order("timestamp", desc=True)
    .limit(300)
    .execute()
)
raw_candles = response.data


# ==================== MAIN DASHBOARD ====================
st.markdown("<h1 style='margin-bottom:0;'>⚡ AI Options Copilot</h1>", unsafe_allow_html=True)
st.caption(
    f"Analyzing **{display_name}** · Mode: **{mode_label}** · "
    f"{'Options-eligible ✅' if is_options_eligible else 'Equity (directional only)'}"
)

if not raw_candles:
    st.info("📭 No data yet. Use the sidebar to sync.")
    st.stop()

# ── Build DataFrame ────────────────────────────────────────────────────────────
df = pd.DataFrame(raw_candles).iloc[::-1].reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# FIX: deduplicate before computing indicators to avoid corrupted rolling windows
df = (
    df.drop_duplicates(subset="timestamp")
    .sort_values("timestamp")
    .reset_index(drop=True)
)

df = AdvancedQuantEngine.compute_indicators(df)
copilot = AdvancedQuantEngine.process_ai_copilot(df, is_options_eligible=is_options_eligible)


# ==================== LAYOUT ====================
col_left, col_right = st.columns([3, 1.2], gap="large")

# ── LEFT: Chart ────────────────────────────────────────────────────────────────
with col_left:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Action", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Candles",
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
        ),
        row=1, col=1,
    )
    if "EMA_9" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["EMA_9"],  name="EMA 9",
                       line=dict(color="#d2a8ff", width=1)),
            row=1, col=1,
        )
    if "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["EMA_20"], name="EMA 20",
                       line=dict(color="#ffa657", width=1.5)),
            row=1, col=1,
        )
    if "EMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["EMA_50"], name="EMA 50",
                       line=dict(color="#58a6ff", width=1.5)),
            row=1, col=1,
        )

    colors = [
        "#3fb950" if df["close"].iloc[i] >= df["open"].iloc[i] else "#f85149"
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color=colors),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── RIGHT: Signal panel ────────────────────────────────────────────────────────
with col_right:

    # ── Signal card: options or equity depending on segment ───────────────────
    if is_options_eligible:
        st.markdown("### 🎯 Option Strategy")
        option_action = copilot["option_action"]
        if option_action == "CALL":
            badge_html = "<span class='badge-call'>BUY CALL</span>"
        elif option_action == "PUT":
            badge_html = "<span class='badge-put'>BUY PUT</span>"
        else:
            badge_html = "<span class='badge-neutral'>NO TRADE</span>"

        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:2rem 1rem;">
            <div style="font-size:0.85rem; color:#8b949e; margin-bottom:0.5rem;">RECOMMENDED ACTION</div>
            <div style="margin:0.5rem 0;">{badge_html}</div>
            <div style="display:flex; justify-content:center; gap:2rem; margin-top:1rem;">
                <div>
                    <span class="metric-label">Score</span>
                    <div style="font-size:1.8rem; font-weight:700;">{copilot['score']}%</div>
                </div>
                <div>
                    <span class="metric-label">Confidence</span>
                    <div style="font-size:1.8rem; font-weight:700;">{copilot['confidence']}%</div>
                </div>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{copilot['confidence']}%;"></div>
            </div>
            <div style="margin-top:0.8rem; font-size:0.85rem; color:#8b949e;">
                Regime: {copilot['regime']}
            </div>
        </div>
        <div class="info-note">
            ℹ️ Strike & expiry selection requires live option chain data.
            This signal is directional only.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Equity: show directional BUY / SELL / HOLD badge only
        st.markdown("### 📡 Directional Signal")
        direction = copilot["direction"]
        if direction == "BUY":
            badge_html = "<span class='badge-buy'>BUY</span>"
        elif direction == "SELL":
            badge_html = "<span class='badge-sell'>SELL</span>"
        else:
            badge_html = "<span class='badge-hold'>HOLD</span>"

        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:2rem 1rem;">
            <div style="font-size:0.85rem; color:#8b949e; margin-bottom:0.5rem;">DIRECTIONAL BIAS</div>
            <div style="margin:0.5rem 0;">{badge_html}</div>
            <div style="display:flex; justify-content:center; gap:2rem; margin-top:1rem;">
                <div>
                    <span class="metric-label">Score</span>
                    <div style="font-size:1.8rem; font-weight:700;">{copilot['score']}%</div>
                </div>
                <div>
                    <span class="metric-label">Confidence</span>
                    <div style="font-size:1.8rem; font-weight:700;">{copilot['confidence']}%</div>
                </div>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{copilot['confidence']}%;"></div>
            </div>
            <div style="margin-top:0.8rem; font-size:0.85rem; color:#8b949e;">
                Regime: {copilot['regime']}
            </div>
        </div>
        <div class="info-note">
            ℹ️ Equity instrument — options signals require Index / FNO segment.
        </div>
        """, unsafe_allow_html=True)

    # ── Trade Levels ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Trade Levels")
    sl_val  = copilot["stop_loss"]
    t1_val  = copilot["target_1"]
    t2_val  = copilot["target_2"]
    st.markdown(f"""
    <div class="glass-card">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
            <div><span class="metric-label">Entry</span>
                 <div style="font-weight:600;">{copilot['entry']}</div></div>
            <div><span class="metric-label">Stop Loss</span>
                 <div style="font-weight:600; color:#f85149;">{sl_val}</div></div>
            <div><span class="metric-label">Target 1</span>
                 <div style="font-weight:600; color:#3fb950;">{t1_val}</div></div>
            <div><span class="metric-label">Target 2</span>
                 <div style="font-weight:600; color:#58a6ff;">{t2_val}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    st.markdown("### 🧠 Reasoning")
    reason_html = "".join(
        [f"<div class='reason-item'>• {r}</div>" for r in copilot["reasons"]]
    )
    st.markdown(f"<div class='glass-card'>{reason_html}</div>", unsafe_allow_html=True)

    # ── Key Stats ─────────────────────────────────────────────────────────────
    st.markdown("### 📈 Key Stats")
    last = df.iloc[-1]
    stats = {
        "High":   last["high"],
        "Low":    last["low"],
        "Volume": f"{int(last['volume']):,}",
        "RSI":    round(last["RSI"], 1) if "RSI" in last else "-",
    }
    stat_cols = st.columns(2)
    for i, (k, v) in enumerate(stats.items()):
        with stat_cols[i % 2]:
            st.metric(label=k, value=v)

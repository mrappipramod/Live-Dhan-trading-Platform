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
    page_title="AI Trading Copilot",
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
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
    }
    .badge-buy {
        background: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }
    .badge-sell {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .badge-hold {
        background: rgba(139, 148, 158, 0.2);
        color: #8b949e;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid rgba(139, 148, 158, 0.3);
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
    .chart-container {
        background: rgba(13, 17, 23, 0.7);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 0.5rem;
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

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== QUANT ENGINE ====================
class AdvancedQuantEngine:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 5:
            return df
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean().fillna(df['close'] * 0.005)
        df['VMA_20'] = df['volume'].rolling(window=20).mean().fillna(df['volume'])
        return df

    @staticmethod
    def process_ai_copilot(df: pd.DataFrame) -> dict:
        if df.empty or 'RSI' not in df.columns:
            return {"regime": "Awaiting Data", "score": 0, "direction": "HOLD", 
                    "entry": "-", "stop_loss": "-", "target_1": "-", "target_2": "-",
                    "reasons": ["Insufficient data."]}
        latest = df.iloc[-1]
        trend_score = 50
        if latest['close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
            trend_score = 100
        elif latest['close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
            trend_score = 0
        momentum_score = 50
        if 50 < latest['RSI'] < 70:
            momentum_score = 90
        elif latest['RSI'] >= 70:
            momentum_score = 40
        elif 30 < latest['RSI'] <= 50:
            momentum_score = 20
        volume_score = 50
        if latest['volume'] > latest['VMA_20'] * 1.3:
            volume_score = 100
        final_score = (trend_score * 0.30) + (momentum_score * 0.30) + (volume_score * 0.40)
        if final_score >= 80:
            regime, direction = "Strong Bullish", "BUY"
        elif final_score >= 60:
            regime, direction = "Bullish", "BUY"
        elif final_score <= 25:
            regime, direction = "Strong Bearish", "SELL"
        elif final_score <= 40:
            regime, direction = "Bearish", "SELL"
        else:
            regime, direction = "Sideways", "HOLD"
        reasons = []
        if trend_score == 100: reasons.append("Price above EMAs – strong uptrend.")
        if latest['RSI'] > 60: reasons.append(f"RSI at {round(latest['RSI'],1)} – momentum expanding.")
        if volume_score == 100: reasons.append("Volume surge above 20‑period average.")
        if not reasons: reasons.append("No clear directional bias – awaiting catalyst.")
        atr = latest['ATR']
        return {
            "regime": regime,
            "score": int(final_score),
            "direction": direction,
            "entry": f"{round(latest['close'], 2)}",
            "stop_loss": f"{round(latest['close'] - (atr * 1.5), 2)}" if direction == "BUY" else f"{round(latest['close'] + (atr * 1.5), 2)}",
            "target_1": f"{round(latest['close'] + (atr * 1.5), 2)}" if direction == "BUY" else f"{round(latest['close'] - (atr * 1.5), 2)}",
            "target_2": f"{round(latest['close'] + (atr * 3.0), 2)}" if direction == "BUY" else f"{round(latest['close'] - (atr * 3.0), 2)}",
            "reasons": reasons
        }

# ==================== DHAN API ====================
def pull_historical_dhan(security_id: str, exchange_segment: str, instrument_type: str, days_back: int, interval: str = None):
    url = "https://api.dhan.co/v2/charts/historical"
    headers = {"access-token": DHAN_ACCESS_TOKEN, "Content-Type": "application/json", "Accept": "application/json"}
    
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument_type,
        "expiryCode": 0,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    # Add interval only for intraday (Dhan defaults to daily if omitted)
    if interval:
        payload["interval"] = interval
    
    try:
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 200:
            data = res.json()
            timestamps = data.get("timestamp", data.get("start_Time", []))
            if not timestamps:
                return []
            opens, highs, lows, closes, volumes = data.get("open", []), data.get("high", []), data.get("low", []), data.get("close", []), data.get("volume", [])
            candles = []
            for i in range(len(timestamps)):
                ts = int(timestamps[i])
                if ts < 1500000000:
                    ts += 315532800
                dt_str = pd.to_datetime(ts, unit='s', utc=True).tz_convert('Asia/Kolkata').isoformat()
                candles.append({
                    "timestamp": dt_str,
                    "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": volumes[i]
                })
            return candles
        else:
            st.error(f"Dhan API error {res.status_code}")
    except Exception as e:
        st.error(f"Network error: {e}")
    return []

# ==================== PRESETS ====================
PRESET_ASSETS = {
    "NIFTY 50": {"id": "13", "seg": "IDX_I", "inst": "INDEX"},
    "BANK NIFTY": {"id": "25", "seg": "IDX_I", "inst": "INDEX"},
    "FINNIFTY": {"id": "27", "seg": "IDX_I", "inst": "INDEX"},
    "RELIANCE": {"id": "2885", "seg": "NSE_EQ", "inst": "EQUITY"},
    "HDFCBANK": {"id": "3456", "seg": "NSE_EQ", "inst": "EQUITY"},
    "ICICIBANK": {"id": "4963", "seg": "NSE_EQ", "inst": "EQUITY"},
    "TCS": {"id": "11536", "seg": "NSE_EQ", "inst": "EQUITY"},
    "INFY": {"id": "1594", "seg": "NSE_EQ", "inst": "EQUITY"},
    "SBIN": {"id": "3045", "seg": "NSE_EQ", "inst": "EQUITY"}
}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚡ Asset Selection")
    search_mode = st.radio("Method", ["Pre‑defined", "Custom"], index=0, horizontal=True)
    
    if search_mode == "Pre‑defined":
        selected_name = st.selectbox("Instrument", list(PRESET_ASSETS.keys()))
        active_sec_id = PRESET_ASSETS[selected_name]["id"]
        active_seg = PRESET_ASSETS[selected_name]["seg"]
        active_inst = PRESET_ASSETS[selected_name]["inst"]
        display_name = selected_name
    else:
        active_sec_id = st.text_input("Security ID", value="3456")
        active_seg = st.selectbox("Segment", ["NSE_EQ", "IDX_I", "NSE_FNO", "BSE_EQ"])
        active_inst = st.selectbox("Type", ["EQUITY", "INDEX", "OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"])
        display_name = f"ID: {active_sec_id}"
    
    st.divider()
    
    # --- MODE SELECTION (Daily vs Intraday) ---
    data_mode = st.radio(
        "📊 Chart Mode",
        ["Daily (Swing)", "Intraday (Live)"],
        index=0,
        help="Daily: 100 days, accurate EMA 50. Intraday: 5-min candles, 7 days, auto-sync every 5 min."
    )
    
    # Configure parameters based on mode
    if data_mode == "Daily (Swing)":
        days_back = 100
        cleanup_days = 100
        interval = None          # Omit = daily candles
        auto_sync_minutes = 60   # Sync every hour (market doesn't change much after close)
        mode_label = "Daily"
    else:  # Intraday
        days_back = 7
        cleanup_days = 7
        interval = "5"           # 5-minute candles
        auto_sync_minutes = 5
        mode_label = "Intraday (5m)"
    
    st.caption(f"Mode: **{mode_label}** · Keeping last **{cleanup_days} days**")
    st.divider()
    
    sync_clicked = st.button("🔄 Sync Now", use_container_width=True)

# ==================== SYNC FUNCTION (reusable) ====================
def sync_data(symbol, seg, inst, name, days_back, interval, cleanup_days, show_status=True):
    """Fetch from Dhan, upsert to Supabase, and clean old data."""
    with st.spinner(f"Fetching {name} ({mode_label})..."):
        candles = pull_historical_dhan(symbol, seg, inst, days_back, interval)
        if not candles:
            if show_status: st.warning("No data. Check token or market hours.")
            return False
        
        try:
            payload_list = [{
                "symbol": str(symbol),
                "timestamp": c["timestamp"],
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": int(c["volume"])
            } for c in candles]
            
            # Upsert
            supabase.table("live_candles").upsert(payload_list, on_conflict="symbol,timestamp").execute()
            
            # Cleanup old data
            cutoff = (pd.Timestamp.now() - pd.Timedelta(days=cleanup_days)).isoformat()
            supabase.table("live_candles") \
                .delete() \
                .eq("symbol", str(symbol)) \
                .lt("timestamp", cutoff) \
                .execute()
            
            if show_status:
                st.success(f"✅ Synced {len(payload_list)} candles (cleaned > {cleanup_days} days)")
            return True
        except Exception as e:
            err = e.args[0] if e.args else str(e)
            if show_status: st.error(f"❌ Supabase error: {err}")
            return False

# ==================== MANUAL SYNC ====================
if sync_clicked:
    sync_data(str(active_sec_id), active_seg, active_inst, display_name, days_back, interval, cleanup_days)

# ==================== AUTO-SYNC ON PAGE LOAD ====================
def auto_sync_if_stale():
    """Check latest timestamp; if older than auto_sync_minutes, pull fresh data."""
    latest_check = supabase.table("live_candles") \
        .select("timestamp") \
        .eq("symbol", str(active_sec_id)) \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()
    
    should_sync = False
    if not latest_check.data:
        should_sync = True
    else:
        latest_ts = pd.to_datetime(latest_check.data[0]['timestamp'])
        age = datetime.now().astimezone() - latest_ts
        if age > timedelta(minutes=auto_sync_minutes):
            should_sync = True
            st.info(f"⏰ Data is {age.seconds//60} min old. Auto-syncing...")
    
    if should_sync:
        sync_data(str(active_sec_id), active_seg, active_inst, display_name, days_back, interval, cleanup_days, show_status=True)

# Run auto-sync (will show status messages if it triggers)
auto_sync_if_stale()

# ==================== FETCH DATA FOR DASHBOARD ====================
response = supabase.table("live_candles") \
    .select("*") \
    .eq("symbol", str(active_sec_id)) \
    .order("timestamp", desc=True) \
    .limit(300) \
    .execute()
raw_candles = response.data

# ==================== MAIN DASHBOARD ====================
st.markdown("<h1 style='margin-bottom: 0;'>⚡ Institutional AI Trading Copilot</h1>", unsafe_allow_html=True)
st.caption(f"Analyzing **{display_name}** · Mode: **{mode_label}** · Data from Dhan & Supabase")

if not raw_candles:
    st.info("📭 No data yet. Use the sidebar to sync.")
    st.stop()

# Process data
df = pd.DataFrame(raw_candles).iloc[::-1].reset_index(drop=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = AdvancedQuantEngine.compute_indicators(df)
copilot = AdvancedQuantEngine.process_ai_copilot(df)

# ==================== LAYOUT ====================
col_left, col_right = st.columns([3, 1.2], gap="large")

with col_left:
    # Candlestick + Volume subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Action", "Volume")
    )
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Candles",
        increasing_line_color='#3fb950',
        decreasing_line_color='#f85149'
    ), row=1, col=1)
    
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], name='EMA 20', line=dict(color='#ffa657', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], name='EMA 50', line=dict(color='#58a6ff', width=1.5)), row=1, col=1)
    
    # Volume bars colored by direction
    colors = ['#3fb950' if df['close'].iloc[i] >= df['open'].iloc[i] else '#f85149' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col_right:
    # ---- AI Copilot Card ----
    st.markdown("### 🤖 AI Signal")
    badge_class = f"badge-{copilot['direction'].lower()}"
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="metric-label">Regime</span>
            <span class="{badge_class}">{copilot['direction']}</span>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; margin: 0.4rem 0;">{copilot['regime']}</div>
        <div style="display: flex; gap: 1rem; margin-top: 0.8rem;">
            <div><span class="metric-label">Score</span><div style="font-size: 2rem; font-weight: 700;">{copilot['score']}%</div></div>
            <div><span class="metric-label">Entry</span><div style="font-size: 1.4rem; font-weight: 600;">{copilot['entry']}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Trade Levels ----
    st.markdown("### 📊 Trade Levels")
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div><span class="metric-label">Stop Loss</span><div style="font-weight: 600;">{copilot['stop_loss']}</div></div>
            <div><span class="metric-label">Target 1</span><div style="font-weight: 600; color: #3fb950;">{copilot['target_1']}</div></div>
            <div><span class="metric-label">Target 2</span><div style="font-weight: 600; color: #58a6ff;">{copilot['target_2']}</div></div>
            <div><span class="metric-label">Current</span><div style="font-weight: 600;">{df['close'].iloc[-1]:.2f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Reasons ----
    st.markdown("### 🧠 Reasoning")
    reason_html = "".join([f"<div class='reason-item'>• {r}</div>" for r in copilot['reasons']])
    st.markdown(f"<div class='glass-card'>{reason_html}</div>", unsafe_allow_html=True)

    # ---- Quick Stats ----
    st.markdown("### 📈 Key Stats")
    last = df.iloc[-1]
    stats = {
        "High": last['high'],
        "Low": last['low'],
        "Volume": f"{last['volume']:,}",
        "RSI": round(last['RSI'], 1) if 'RSI' in last else '-'
    }
    cols = st.columns(2)
    for i, (k, v) in enumerate(stats.items()):
        with cols[i % 2]:
            st.metric(label=k, value=v, delta=None)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client, Client

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Trading Copilot", page_icon="⚡")

st.markdown("""
    <style>
    .stApp { background-color: #080B0E; color: #D1D4DC; }
    div[data-testid="stMetricValue"] { font-family: monospace; font-size: 32px; font-weight: bold; }
    .card { background-color: #12161A; padding: 20px; border-radius: 8px; border: 1px solid #23282E; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- SECRETS & CLOUD INITIALIZATION ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
# FIX: Prefer the SERVICE_ROLE key to bypass RLS, fallback to anon key if not set
SUPABASE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_KEY", ""))
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Configuration vectors missing. Please bind secrets in your Streamlit Cloud Dashboard.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- ADVANCED QUANTITATIVE ENGINE ---
class AdvancedQuantEngine:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 5:
            return df
        
        # Exponential Moving Averages
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean().fillna(df['close'] * 0.005)
        
        # Volume Moving Average
        df['VMA_20'] = df['volume'].rolling(window=20).mean().fillna(df['volume'])
        return df

    @staticmethod
    def process_ai_copilot(df: pd.DataFrame) -> dict:
        if df.empty or 'RSI' not in df.columns:
            return {"regime": "Awaiting Data", "confidence": 0, "score": 0, "direction": "HOLD", "reasons": ["Insufficient data points."]}
        
        latest = df.iloc[-1]
        
        # 1. Trend Matrix (30%)
        trend_score = 50
        if latest['close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
            trend_score = 100
        elif latest['close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
            trend_score = 0
            
        # 2. Momentum Matrix (30%)
        momentum_score = 50
        if 50 < latest['RSI'] < 70:
            momentum_score = 90
        elif latest['RSI'] >= 70:
            momentum_score = 40  
        elif 30 < latest['RSI'] <= 50:
            momentum_score = 20
            
        # 3. Volume Core (40%)
        volume_score = 50
        if latest['volume'] > latest['VMA_20'] * 1.3:
            volume_score = 100
            
        final_score = (trend_score * 0.30) + (momentum_score * 0.30) + (volume_score * 0.40)
        
        if final_score >= 80:
            regime, direction = "Strong Bullish Trend", "BUY"
        elif final_score >= 60:
            regime, direction = "Bullish Trend", "BUY"
        elif final_score <= 25:
            regime, direction = "Strong Bearish Trend", "SELL"
        elif final_score <= 40:
            regime, direction = "Bearish Trend", "SELL"
        else:
            regime, direction = "Sideways Mode", "HOLD"
            
        reasons = []
        if trend_score == 100: reasons.append("Price exhibits structured exponential breakout alignment.")
        if latest['RSI'] > 60: reasons.append(f"Momentum expansion confirmed via RSI at {round(latest['RSI'], 1)}.")
        if volume_score == 100: reasons.append("Institutional volume expansion verified above 20-period VMA.")
        if not reasons: reasons.append("Market forces neutralizing; core parameters consolidation bound.")
        
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

# --- DHAN API INTEGRATION (V2 Corrected) ---
def pull_historical_dhan(security_id: str, exchange_segment: str, instrument_type: str):
    url = "https://api.dhan.co/v2/charts/historical"
    headers = {
        "access-token": DHAN_ACCESS_TOKEN, 
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": instrument_type,
        "expiryCode": 0,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 200:
            data = res.json()
            timestamps = data.get("timestamp", data.get("start_Time", []))
            
            if not timestamps:
                st.warning("Dhan returned a success code but no data. Check if token is active or if market is closed.")
                return []
                
            opens, highs, lows, closes, volumes = data.get("open", []), data.get("high", []), data.get("low", []), data.get("close", []), data.get("volume", [])
            
            formatted_candles = []
            for i in range(len(timestamps)):
                ts = int(timestamps[i])
                if ts < 1500000000:  
                    ts += 315532800 # Correct Dhan's epoch offset
                
                dt_str = pd.to_datetime(ts, unit='s', utc=True).tz_convert('Asia/Kolkata').isoformat()
                
                formatted_candles.append({
                    "timestamp": dt_str,
                    "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i], "volume": volumes[i]
                })
            return formatted_candles
        else:
            st.error(f"Dhan API Error {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Network/Parsing Exception: {e}")
    return []

# --- APP UI & SIDEBAR ---
st.title("⚡ Institutional AI Trading Copilot")

st.sidebar.header("Asset Selection Engine")

# PRE-DEFINED MAJOR INSTRUMENTS
PRESET_ASSETS = {
    "NIFTY 50 (Index)": {"id": "13", "seg": "IDX_I", "inst": "INDEX"},
    "BANK NIFTY (Index)": {"id": "25", "seg": "IDX_I", "inst": "INDEX"},
    "FINNIFTY (Index)": {"id": "27", "seg": "IDX_I", "inst": "INDEX"},
    "RELIANCE": {"id": "2885", "seg": "NSE_EQ", "inst": "EQUITY"},
    "HDFCBANK": {"id": "3456", "seg": "NSE_EQ", "inst": "EQUITY"},
    "ICICIBANK": {"id": "4963", "seg": "NSE_EQ", "inst": "EQUITY"},
    "TCS": {"id": "11536", "seg": "NSE_EQ", "inst": "EQUITY"},
    "INFY": {"id": "1594", "seg": "NSE_EQ", "inst": "EQUITY"},
    "SBIN": {"id": "3045", "seg": "NSE_EQ", "inst": "EQUITY"}
}

search_mode = st.sidebar.radio("Select Input Method:", ["Use Pre-defined List", "Custom Search (Any Instrument)"])

if search_mode == "Use Pre-defined List":
    selected_name = st.sidebar.selectbox("Choose Asset:", list(PRESET_ASSETS.keys()))
    active_sec_id = PRESET_ASSETS[selected_name]["id"]
    active_seg = PRESET_ASSETS[selected_name]["seg"]
    active_inst = PRESET_ASSETS[selected_name]["inst"]
    display_name = selected_name
else:
    st.sidebar.markdown("*(Find Dhan Security IDs in their API Security CSV)*")
    active_sec_id = st.sidebar.text_input("Security ID (e.g., 3456)", value="3456")
    active_seg = st.sidebar.selectbox("Exchange Segment", ["NSE_EQ", "IDX_I", "NSE_FNO", "BSE_EQ"])
    active_inst = st.sidebar.selectbox("Instrument Type", ["EQUITY", "INDEX", "OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"])
    display_name = f"Custom ID: {active_sec_id}"

st.sidebar.divider()

if st.sidebar.button("🔄 Sync Market Data Vector"):
    with st.spinner(f"Pulling historical data for {display_name}..."):
        candles = pull_historical_dhan(active_sec_id, active_seg, active_inst)
        if candles:
            try:
                # --- FIX 1: Build a LIST of payloads instead of upserting one by one ---
                # This is MUCH faster and properly handles the upsert contract.
                payload_list = []
                for candle in candles:
                    payload_list.append({
                        "symbol": str(active_sec_id),
                        "timestamp": candle["timestamp"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle["volume"]
                    })
                
                # --- FIX 2: Pass the LIST and catch potential API errors ---
                supabase.table("live_candles").upsert(payload_list, on_conflict="symbol,timestamp").execute()
                st.sidebar.success(f"✅ Synced {len(payload_list)} candles to cloud.")
            
            except Exception as e:
                # --- FIX 3: Display the actual Supabase error to the user ---
                error_msg = e.args[0] if e.args else str(e)
                st.sidebar.error(f"❌ Supabase Error: {error_msg}")
                # If you want to see the full raw error in the logs:
                # st.exception(e)
        else:
            st.sidebar.warning("Fetch failed. Ensure market is open or check API credentials.")

# --- DASHBOARD RENDER ---
# Pull data based on the Active Security ID
response = supabase.table("live_candles").select("*").eq("symbol", str(active_sec_id)).order("timestamp", desc=True).limit(300).execute()
raw_candles = response.data

if raw_candles:
    df = pd.DataFrame(raw_candles).iloc[::-1].reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Process Engine
    df = AdvancedQuantEngine.compute_indicators(df)
    copilot = AdvancedQuantEngine.process_ai_copilot(df)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### Terminal Interface: {display_name}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Candles", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], name='EMA 20', line=dict(color='#ff9f43', width=1.5)))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], name='EMA 50', line=dict(color='#0abde3', width=1.5)))
            
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#080B0E", plot_bgcolor="#080B0E")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### AI Copilot Analytics")
        metric_color = "#26a69a" if copilot['direction'] == "BUY" else "#ef5350" if copilot['direction'] == "SELL" else "#72767d"
        st.markdown(f"<div class='card'><h3 style='margin:0;color:gray;font-size:12px;'>CURRENT REGIME</h3><h2 style='margin:5px 0;color:{metric_color};'>{copilot['regime']}</h2></div>", unsafe_allow_html=True)
        
        st.metric(label="Model Probability Matrix", value=f"{copilot['score']}%")
        
        if copilot['direction'] != "HOLD":
            st.text_input("Execution Target", copilot['entry'], disabled=True)
            st.text_input("Invalidation Level (SL)", copilot['stop_loss'], disabled=True)
            st.text_input("Profit Target 1", copilot['target_1'], disabled=True)
            st.text_input("Profit Target 2", copilot['target_2'], disabled=True)
        else:
            st.info("Market bound inside compression corridors. System withholding trade signals.")
            
        st.markdown("### Engine Consensus Breakdown")
        for reason in copilot['reasons']:
            st.markdown(f"▪ *{reason}*")
else:
    st.info(f"No data stored yet for {display_name}. Click 'Sync Market Data Vector' to fetch from Dhan.")

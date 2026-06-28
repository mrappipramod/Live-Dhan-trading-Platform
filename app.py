import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client, Client

st.set_page_config(layout="wide", page_title="AI Trading Copilot", page_icon="⚡")

# --- INSTANCE STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #080B0E; color: #D1D4DC; }
    div[data-testid="stMetricValue"] { font-family: monospace; font-size: 32px; font-weight: bold; }
    .card { background-color: #12161A; padding: 20px; border-radius: 8px; border: 1px solid #23282E; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG & CREDENTIALS ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Configuration vectors missing. Please bind secrets in your Streamlit Cloud Dashboard.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- ADVANCED QUANTITATIVE ENGINE ---
class AdvancedQuantEngine:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates indicators without external dependencies to avoid environment friction"""
        if len(df) < 5:
            return df
        
        # Exponential Moving Averages
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
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
        """Executes full weighted multi-factor scoring model"""
        if df.empty or 'RSI' not in df.columns:
            return {"regime": "Sideways", "confidence": 50, "score": 50, "direction": "HOLD", "reasons": ["Initialization Data Deficit"]}
        
        latest = df.iloc[-1]
        
        # 1. Trend Matrix (30% weight)
        trend_score = 50
        if latest['close'] > latest['EMA_20'] and latest['EMA_20'] > latest['EMA_50']:
            trend_score = 100
        elif latest['close'] < latest['EMA_20'] and latest['EMA_20'] < latest['EMA_50']:
            trend_score = 0
            
        # 2. Momentum Matrix (30% weight)
        momentum_score = 50
        if 50 < latest['RSI'] < 70:
            momentum_score = 90
        elif latest['RSI'] >= 70:
            momentum_score = 40  # Overbought dampening
        elif 30 < latest['RSI'] <= 50:
            momentum_score = 20
            
        # 3. Volume Core (40% weight)
        volume_score = 50
        if latest['volume'] > latest['VMA_20'] * 1.3:
            volume_score = 100
            
        # Consolidated Weight Calculation
        final_score = (trend_score * 0.30) + (momentum_score * 0.30) + (volume_score * 0.40)
        
        # Define Regime Contexts
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
            
        # Reasons Output Breakdown
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

# --- FETCH & DATA INTERACTION SYNC ---
def pull_historical_dhan(symbol: str):
    url = "https://api.dhan.co/charts/historical"
    headers = {"access-token": DHAN_ACCESS_TOKEN, "Content-Type": "application/json"}
    payload = {
        "symbol": symbol,
        "exchangeSegment": "NSE_EQ",
        "instrumentType": "EQUITY",
        "expiryCode": 0,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    try:
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 200:
            return res.json().get("data", [])
    except:
        pass
    return []

# --- APP LAYOUT UI ---
st.title("⚡ Institutional AI Trading Copilot")

# Asset Vectors Expandable Configuration Menu
st.sidebar.header("Asset Spectrum Profile")
asset_dictionary = {
    "2885": "RELIANCE", "1333": "INFOSYS", "11536": "TCS", 
    "3456": "HDFCBANK", "4963": "ICICIBANK", "3045": "SBIN"
}
selected_asset = st.sidebar.selectbox("Active Asset Vector", list(asset_dictionary.keys()), format_func=lambda x: asset_dictionary[x])

if st.sidebar.button("🔄 Sync Market Data Vector"):
    with st.spinner(f"Pulling complete data pipeline matrix for {asset_dictionary[selected_asset]}..."):
        candles = pull_historical_dhan(selected_asset)
        if candles:
            for candle in candles:
                payload = {
                    "symbol": selected_asset,
                    "timestamp": datetime.fromtimestamp(candle[0]).isoformat(),
                    "open": candle[1], "high": candle[2], "low": candle[3], "close": candle[4], "volume": candle[5]
                }
                supabase.table("live_candles").upsert(payload, on_conflict="symbol,timestamp").execute()
            st.success("Cloud parameters synced successfully.")
        else:
            st.warning("Data fetch returned empty values. Verifying active Dhan API access tokens required.")

# Read Matrix
response = supabase.table("live_candles").select("*").eq("symbol", selected_asset).order("timestamp", desc=True).limit(300).execute()
raw_candles = response.data

if raw_candles:
    df = pd.DataFrame(raw_candles).iloc[::-1].reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate Data
    df = AdvancedQuantEngine.compute_indicators(df)
    copilot = AdvancedQuantEngine.process_ai_copilot(df)
    
    # Render layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### Terminal Interface: {asset_dictionary[selected_asset]}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Market Candlestick Vector", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ))
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], name='EMA 20', line=dict(color='#ff9f43', width=1.5)))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_50'], name='EMA 50', line=dict(color='#0abde3', width=1.5)))
            
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#080B0E", plot_bgcolor="#080B0E")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### AI Copilot Analytics")
        
        # Color coding metrics matrix values
        metric_color = "#26a69a" if copilot['direction'] == "BUY" else "#ef5350" if copilot['direction'] == "SELL" else "#72767d"
        st.markdown(f"<div class='card'><h3 style='margin:0;color:gray;font-size:12px;'>CURRENT REGIME</h3><h2 style='margin:5px 0;color:{metric_color};'>{copilot['regime']}</h2></div>", unsafe_allow_html=True)
        
        st.metric(label="Model Probability Score Matrix", value=f"{copilot['score']}%")
        
        st.markdown("### Action Specifications")
        if copilot['direction'] != "HOLD":
            st.text_input("Calculated Execution Target", copilot['entry'], disabled=True)
            st.text_input("Risk Invalidation Level (SL)", copilot['stop_loss'], disabled=True)
            st.text_input("Profit Extraction Target 1", copilot['target_1'], disabled=True)
            st.text_input("Profit Extraction Target 2", copilot['target_2'], disabled=True)
        else:
            st.info("Market status currently bound inside compression corridors. System withholding trade deployment signals.")
            
        st.markdown("### Engine Consensus Breakdown")
        for reason in copilot['reasons']:
            st.markdown(f"▪ *{reason}*")
else:
    st.warning("Database empty. Choose an instrument and run 'Sync Market Data Vector' to test live computational charts.")

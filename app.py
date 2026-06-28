import streamlit as st
import pandas as pd
import os
import requests
from supabase import create_client, Client
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="AI Trading Copilot")

# --- DATABASE & API INITIALIZATION ---
# Streamlit Cloud reads these securely from your dashboard settings
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing configuration secrets. Please add them in Streamlit Cloud settings.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- ENGINE LOGIC ---
def fetch_dhan_data(security_id: str):
    """Fetches recent data from Dhan REST API"""
    url = "https://api.dhan.co/charts/historical"
    headers = {
        "access-token": DHAN_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    payload = {
        "symbol": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrumentType": "EQUITY",
        "expiryCode": 0,
        "fromDate": (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "toDate": pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("data", [])
    except Exception:
        pass
    return []

def generate_ai_signal(df: pd.DataFrame) -> dict:
    """Calculates simple technical logic to output trading metrics"""
    if df.empty or len(df) < 5:
        return {"direction": "HOLD", "score": 50, "reasons": ["Insufficient data stream"]}
    
    latest_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    
    # Simple core momentum scoring
    score = 50
    reasons = []
    
    if latest_close > prev_close:
        score += 25
        reasons.append("Short-term price action is positive")
    else:
        score -= 25
        reasons.append("Short-term price action is soft")
        
    direction = "BUY" if score >= 75 else "SELL" if score <= 25 else "HOLD"
    
    return {
        "direction": direction,
        "score": score,
        "entry": f"{round(latest_close, 2)}",
        "stop_loss": f"{round(latest_close * 0.99, 2)}",
        "target_1": f"{round(latest_close * 1.01, 2)}",
        "target_2": f"{round(latest_close * 1.02, 2)}",
        "reasons": reasons
    }

# --- UI APP LAYOUT ---
st.title("⚡ Institutional AI Trading Copilot")

# Sidebar configurations
selected_asset = st.sidebar.selectbox("Active Asset Vector", ["2885", "1333"], format_func=lambda x: "RELIANCE" if x == "2885" else "INFOSYS")

# Manual Data Synchronizer for Cloud environment
if st.sidebar.button("🔄 Sync Market Data"):
    with st.spinner("Fetching latest data from Dhan..."):
        candles = fetch_dhan_data(selected_asset)
        for candle in candles:
            # Upsert into Supabase
            payload = {
                "symbol": selected_asset,
                "timestamp": datetime.fromtimestamp(candle[0]).isoformat(),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            }
            supabase.table("live_candles").upsert(payload, on_conflict="symbol,timestamp").execute()
    st.success("Database sync complete!")

# Fetch Data from Supabase to render
response = supabase.table("live_candles").select("*").eq("symbol", selected_asset).order("timestamp", desc=True).limit(100).execute()
data = response.data

if data:
    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
    ai_metrics = generate_ai_signal(df)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Asset Performance Window")
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Copilot Output Matrix")
        color = "#26a69a" if ai_metrics['direction'] == "BUY" else "#ef5350" if ai_metrics['direction'] == "SELL" else "#72767d"
        st.markdown(f"<h2 style='color: {color};'>SIGNAL: {ai_metrics['direction']}</h2>", unsafe_allow_html=True)
        st.metric("Model Confidence Score", f"{ai_metrics['score']}%")
        
        if ai_metrics['direction'] != "HOLD":
            st.text_input("Entry Price Level", ai_metrics['entry'], disabled=True)
            st.text_input("Invalidation Level (SL)", ai_metrics['stop_loss'], disabled=True)
            st.text_input("Target 1", ai_metrics['target_1'], disabled=True)
            st.text_input("Target 2", ai_metrics['target_2'], disabled=True)
            
        st.markdown("#### Structural Assertions")
        for reason in ai_metrics['reasons']:
            st.info(reason)
else:
    st.info("No data found in your cloud database yet. Click 'Sync Market Data' in the sidebar to populate.")

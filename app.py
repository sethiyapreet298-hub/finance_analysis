import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import analysis
import time
import json
import os

st.set_page_config(page_title="MasterTrade AI", layout="wide", page_icon="📈")

# --- CUSTOM CSS (Glassmorphism & Modern UI) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1a1a2e 0%, #16213e 90%);
        color: #ffffff;
    }
    
    /* Metric Card (Glassmorphism) */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .big-verdict {
        font-family: 'Inter', sans-serif;
        font-size: 56px;
        font-weight: 800;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    
    .confidence-score {
        font-size: 18px;
        font-weight: 600;
        color: #e0e0e0;
        letter-spacing: 1px;
    }
    
    /* News Cards */
    .news-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00AAFF;
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 0 12px 12px 0;
        transition: all 0.3s ease;
    }
    .news-card:hover {
        background: rgba(255, 255, 255, 0.08);
        padding-left: 20px;
    }
    .news-link {
        text-decoration: none;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 16px;
    }
    .news-meta {
        font-size: 12px;
        color: #aaa;
        margin-top: 5px;
    }
    
    /* Insights Sections */
    .insight-box {
        background: rgba(20, 20, 30, 0.6);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* DataFrame Styling */
    .stDataFrame {
       background: rgba(0,0,0,0.2); 
       border-radius: 10px;
    }
    
    /* Probability Card */
    .prob-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- WATCHLIST LOGIC ---
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return []

def save_watchlist(items):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(items, f)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# --- CACHING WRAPPERS ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_analysis(ticker, period, interval, api_key):
    return analysis.analyze_stock_comprehensive(ticker, period, interval, api_key)

@st.cache_data(ttl=3600) # Cache ticker resolution for 1 hour
def resolve_ticker(query, market_type):
    return analysis.lookup_ticker(query, market_type)
    
@st.cache_data(ttl=300)
def run_backtest(df):
    return analysis.backtest_stock(df)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ MasterTrade AI")
    st.caption("Advanced Algorithm & Market Intelligence")
    st.markdown("---")
    
    # Watchlist Section
    st.subheader("⭐ Watchlist")
    
    # Display Watchlist Items
    cols = st.columns(2)
    for i, symbol in enumerate(st.session_state.watchlist):
        if cols[i % 2].button(symbol, key=f"wl_{symbol}", use_container_width=True):
            st.session_state.ticker_input = symbol
            
    # Add Current Ticker to Watchlist Logic (Handle later in main flow)
    
    st.markdown("---")
    
    # Market Selector
    market_type = st.radio(
        "Market Scope", 
        ["US", "India", "Crypto", "Commodities"],
        horizontal=True,
        help="Select the market ecosystem to focus reliable search."
    )
    
    # Dynamic Input Label
    input_label = "Ticker / Company Name"
    examples = "AAPL, NVDA"
    if market_type == "India": examples = "Reliance, Tata Motors, INFY"
    elif market_type == "Crypto": examples = "Bitcoin, ETH, DOGE"
    elif market_type == "Commodities": examples = "Gold, Silver, Oil"
    
    # Input field with Session State handling
    if 'ticker_input' not in st.session_state:
        st.session_state.ticker_input = ""
        
    ticker_input = st.text_input(input_label, key="ticker_input", placeholder=f"e.g. {examples}")
    
    st.markdown("### ⏱ Timeframe")
    tf_mapping = {
        "Intraday (5m)": ("5m", "5d"),
        "Intraday (30m)": ("30m", "1mo"),
        "Swing (Daily)": ("1d", "1y"),
        "Investing (Weekly)": ("1wk", "2y"),
        "Macro (Monthly)": ("1mo", "5y")
    }
    tf_label = st.select_slider("Analysis Depth", options=list(tf_mapping.keys()), value="Swing (Daily)")
    interval, period = tf_mapping[tf_label]
    
    st.markdown("---")
    # AI Key
    api_key = st.text_input("Gemini API Key (Optional)", type="password", help="For AI-narrated stories.")
    
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info(f"**Mode:** {market_type} Market\n**Strategy:** {tf_label}")

# --- MAIN APP ---
if not ticker_input:
    # Landing Page
    # Landing Page
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 60px; background: -webkit-linear-gradient(#eee, #333); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            MasterTrade AI
        </h1>
        <h3 style="color: #aaa; margin-bottom: 20px;">Institutional-Grade Technical Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    # Native Streamlit Columns for Robustness
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 40px;">🇮🇳</div>
            <h3>Indian Markets</h3>
            <p style="color: #888; font-size: 14px;">NSE & BSE Support</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 40px;">🇺🇸</div>
            <h3>US Markets</h3>
            <p style="color: #888; font-size: 14px;">NYSE & NASDAQ</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 40px;">₿</div>
            <h3>Crypto</h3>
            <p style="color: #888; font-size: 14px;">Real-time Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="text-align: center; margin-top: 40px;">
        <p style="color: #666; font-size: 16px;">👈 Select a market and enter a ticker in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
elif run_btn or ticker_input:
    # Execution Flow
    try:
        # 1. Ticker Resolution
        with st.spinner(f"Searching {market_type} database..."):
            resolved_ticker = resolve_ticker(ticker_input, market_type)
            
        if resolved_ticker != ticker_input.upper() and resolved_ticker != ticker_input:
             st.toast(f"Found: {resolved_ticker}", icon="🎯")

        # 2. Main Analysis Layer
        with st.spinner(f"Crunching numbers for {resolved_ticker}..."):
            df, result, news, ai_insight, pattern_df = get_analysis(resolved_ticker, period, interval, api_key)
            
        # 2a. Backtesting Layer
        backtest_res = run_backtest(df)
            
        if df is None:
            st.error(f"❌ Could not retrieve data for **{resolved_ticker}**. Please check the ticker or try a different market.")
        else:
            time.sleep(0.2) # UX smoothing
            
            # --- HEADER ---
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"<h1 style='margin-bottom: 0;'>{resolved_ticker}</h1>", unsafe_allow_html=True)
                st.caption(f"Market: {market_type} | Data: Yahoo Finance | Interval: {interval}")
                
                # Watchlist Toggle
                if st.button("⭐ Add to Watchlist" if resolved_ticker not in st.session_state.watchlist else "❌ Remove from Watchlist", key="wl_toggle"):
                    if resolved_ticker not in st.session_state.watchlist:
                        st.session_state.watchlist.append(resolved_ticker)
                        st.session_state.watchlist = list(set(st.session_state.watchlist)) # Dedup
                        st.toast(f"Added {resolved_ticker} to Watchlist")
                    else:
                        st.session_state.watchlist.remove(resolved_ticker)
                        st.toast(f"Removed {resolved_ticker}")
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()

            with c2:
                # Live Price Badge
                current_price = result['current_price']
                st.markdown(f"""
                <div style="text-align: right; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;">
                    <span style="font-size: 24px; font-weight: bold;">${current_price:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)

            # --- 1. THE VERDICT ---
            st.markdown("### 🔥 Market Verdict")
            col_v1, col_v2, col_v3 = st.columns([1, 2, 1]) # Centered Verdict
            
            with col_v2:
                decision = result['decision']
                confidence = result['confidence']
                
                # Dynamic Color
                if "BUY" in decision: 
                    theme_color = "#00E676" # Bright Green
                    glow_color = "rgba(0, 230, 118, 0.4)"
                elif "SELL" in decision: 
                    theme_color = "#FF1744" # Bright Red
                    glow_color = "rgba(255, 23, 68, 0.4)"
                else: 
                    theme_color = "#FFC400" # Amber
                    glow_color = "rgba(255, 196, 0, 0.4)"
                
                st.markdown(f"""
                <div class="metric-container" style="border-top: 5px solid {theme_color}; box-shadow: 0 0 30px {glow_color};">
                    <div style="color: #aaa; letter-spacing: 2px; font-size: 14px; text-transform: uppercase;">Technical Consensus</div>
                    <div class="big-verdict" style="color: {theme_color};">{decision}</div>
                    <div class="confidence-score">Confidence Level: {confidence}%</div>
                    <div style="margin-top: 15px; font-size: 12px; color: #888;">
                        Analyzed {len(result['table_data'])} Indicators • {len(result['drivers']['bullish'])} Bullish signals vs {len(result['drivers']['bearish'])} Bearish
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if result['drivers']['bullish']:
                    st.caption(f"✅ **Strength:** {result['drivers']['bullish'][0]}")
                if result['drivers']['bearish']:
                    st.caption(f"⚠️ **Risk:** {result['drivers']['bearish'][0]}")
                
                # Yahoo Finance Link (Explicit)
                yf_link = f"https://finance.yahoo.com/quote/{resolved_ticker}"
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <a href="{yf_link}" target="_blank" style="color: #aaa; text-decoration: underline; font-size: 12px;">
                        Learn more about {resolved_ticker} on Yahoo Finance 🔗
                    </a>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- 2. DEEP DIVE (Charts + Data) ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📋 Technical Data", "🧠 Expert Outlook", "📰 News & Backtest"])
            
            # --- TAB 1: CHARTS ---
            with tab1:
                # Range Logic
                range_breaks = []
                if interval in ['1d', '1wk']:
                    range_breaks = [dict(bounds=["sat", "mon"])]
                
                # Main Price Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                
                # Overlays
                if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1.5), name='SMA 50'))
                if 'SMA_200' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#2979FF', width=2), name='SMA 200'))
                if 'VWAP_D' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_D'], line=dict(color='#FFD700', dash='dot'), name='VWAP'))
                
                fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", 
                                title=dict(text="Price Action & Trend", font=dict(size=20)),
                                margin=dict(l=0, r=0, t=40, b=0),
                                hovermode="x unified",
                                xaxis_rangebreaks=range_breaks,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Momentum Split
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    # MACD/RSI
                    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='#AA00FF'), name='RSI'), row=1, col=1)
                    fig2.add_hline(y=70, row=1, col=1, line_dash="dash", line_color="rgba(255,0,0,0.5)")
                    fig2.add_hline(y=30, row=1, col=1, line_dash="dash", line_color="rgba(0,255,0,0.5)")
                    
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], line=dict(color='#00E5FF'), name='MACD'), row=2, col=1)
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], line=dict(color='#FF6D00'), name='Signal'), row=2, col=1)
                    fig2.add_bar(x=df.index, y=df['MACDh_12_26_9'], name='Hist', marker_color='gray', row=2, col=1)
                    
                    fig2.update_layout(height=350, template="plotly_dark", showlegend=False, 
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis_rangebreaks=range_breaks)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                with c_m2:
                    # Stochastic
                    if 'STOCHk_14_3_3' in df.columns:
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(x=df.index, y=df['STOCHk_14_3_3'], line=dict(color='#00B0FF'), name='Stoch K'))
                        fig3.add_trace(go.Scatter(x=df.index, y=df['STOCHd_14_3_3'], line=dict(color='#FF1744'), name='Stoch D'))
                        fig3.add_hline(y=80, line_dash="dot", line_color="gray")
                        fig3.add_hline(y=20, line_dash="dot", line_color="gray")
                        fig3.update_layout(height=350, title="Stochastic Oscillator", template="plotly_dark",
                                        margin=dict(l=0, r=0, t=30, b=0),
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis_rangebreaks=range_breaks)
                        st.plotly_chart(fig3, use_container_width=True)

            # --- TAB 2: DATA TABLE ---
            with tab2:
                st.subheader("Deep Dive Metrics")
                table_data = result.get('table_data', [])
                if table_data:
                    tdf = pd.DataFrame(table_data)
                    
                    # Highlight Logic
                    def highlight_rows(row):
                        if row["Signal"] == "BULLISH": return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                        if row["Signal"] == "BEARISH": return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                        return [''] * len(row)

                    st.dataframe(
                        tdf,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Indicator": st.column_config.TextColumn("Metric", width="medium"),
                            "Value": st.column_config.TextColumn("Current Value", width="small"),
                            "Signal": st.column_config.TextColumn("Signal", width="small"),
                            "Reliability": st.column_config.ProgressColumn("Impact", min_value=1, max_value=10, format="%d"),
                            "Note": st.column_config.TextColumn("Analysis", width="large"),
                        }
                    )
                else:
                    st.info("No data available.")

            # --- TAB 3: INSIGHTS ---
            with tab3:
                st.markdown("### 🤖 Algorithmic Outlook")
                
                with st.container():
                   i_c1, i_c2 = st.columns(2)
                   with i_c1:
                       st.markdown(f"""
                       <div class="insight-box" style="border-left: 4px solid #BB86FC;">
                           <h4>🌗 Short Term (1-2 Weeks)</h4>
                           <p>{ai_insight['short_term']}</p>
                       </div>
                       """, unsafe_allow_html=True) 
                   with i_c2:
                       st.markdown(f"""
                       <div class="insight-box" style="border-left: 4px solid #03DAC6;">
                           <h4>🪐 Long Term (6+ Months)</h4>
                           <p>{ai_insight['long_term']}</p>
                       </div>
                       """, unsafe_allow_html=True)
                
                st.markdown("#### 📜 Logic Explanation")
                st.info(ai_insight['score_explanation'])

            # --- TAB 4: NEWS & BACKTEST ---
            with tab4:
                col_n, col_p = st.columns([1.5, 1])
                
                with col_n:
                    st.markdown(f"#### 🗞️ Recent Catalyst: {ai_insight['catalyst']}")
                    for n in news:
                        st.markdown(f"""
                        <div class="news-card">
                            <a href="{n['link']}" target="_blank" class="news-link">{n['title']}</a>
                            <div class="news-meta">{n['snippet']} • {n.get('pubDate','')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with col_p:
                    st.markdown("### 📊 Strategy Probability Analysis")
                    if backtest_res and backtest_res.get('status') == 'Active':
                        win_rate = backtest_res['win_rate']
                        wr_color = "#00ff00" if win_rate > 50 else "#ff4444"
                        
                        # Native Streamlit Components for Robustness
                        with st.container():
                            st.markdown(f"""
                            <div class="prob-card">
                                <div style="font-size: 14px; color: #888; margin-bottom: 5px;">HISTORICAL WIN RATE</div>
                                <div style="font-size: 36px; font-weight: 800; color: {wr_color}; line-height: 1;">{win_rate}%</div>
                                <div style="font-size: 12px; color: #aaa; margin-bottom: 15px;">{backtest_res['total_trades']} trades (1Y)</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics Row
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Return", f"{backtest_res['total_return']}%", border=True)
                            with m2:
                                st.metric("Max DD", f"{backtest_res['max_drawdown']}%", border=True)
                            with m3:
                                st.metric("Avg Profit", f"{backtest_res['avg_profit']}%", border=True)
                                
                            # Best Scenario (Simple Text)
                            if backtest_res['best_trade']:
                                entry_d = backtest_res['best_trade']['entry_date'].strftime('%Y-%m-%d')
                                profit = round(backtest_res['best_trade']['return_pct'], 1)
                                st.markdown(f"""
                                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                                    <span style="font-size: 12px; color: #aaa;">🏆 <b>Best Scenario:</b></span><br>
                                    <span style="font-size: 13px;">Entry: {entry_d} ➜ <span style="color: #00ff00; font-weight: bold;">+{profit}%</span></span>
                                </div>
                                """, unsafe_allow_html=True)

                        st.info("ℹ️ **Note:** This probability is calculated purely on mathematical statistics of past price action. It simulates a Trend-Following strategy on the last 365 days of data. Past performance is not a guarantee of future results. You are responsible for your own trades.")
                    else:
                        st.info("Insufficient historical data to calculate reliable probability metrics for this asset.")

    except Exception as e:
        st.error(f"App Runtime Error: {e}")
        import traceback
        st.code(traceback.format_exc())

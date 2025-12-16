import yfinance as yf
import pandas as pd
import pandas_ta as ta
from duckduckgo_search import DDGS
import google.generativeai as genai
import datetime

def fetch_data(ticker, period="2y", interval="1d"):
    """
    Fetches stock data with dynamic period adjustment for intraday.
    """
    try:
        # Intraday constraints for yfinance:
        # 1m: max 7d
        # 5m, 15m, 30m: max 60d
        # 1h: max 730d
        limit_period = period
        if interval in ['1m', '2m', '5m', '15m', '30m', '90m']:
            limit_period = "59d" # Safe buffer
        elif interval == '1h':
            limit_period = "729d" # Safe buffer

        stock = yf.Ticker(ticker)
        df = stock.history(period=limit_period, interval=interval)
        
        if df.empty:
            # Fallback for some indices or messy tickers
            return None
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_market_news(ticker, max_results=5):
    """
    Searches for latest news about the ticker using DuckDuckGo.
    Returns a list of dictionaries with 'title', 'link', 'source'.
    """
    try:
        results = DDGS().text(f"{ticker} stock news analysis why moving today", max_results=max_results)
        news_items = []
        if results:
            for r in results:
                news_items.append({
                    "title": r.get('title'),
                    "link": r.get('href'),
                    "snippet": r.get('body'),
                })
        return news_items
    except Exception as e:
        print(f"Search error: {e}")
        return []

def get_ai_insight(api_key, ticker, technical_summary, news_items):
    """
    Uses Gemini to synthesize technicals + news into a "Why" explanation.
    """
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        news_text = "\n".join([f"- {n['title']}: {n['snippet']}" for n in news_items])
        
        prompt = f"""
        You are a Wall Street Master Trader. Analyze {ticker} based on the following data:
        
        TECHNICAL INDICATORS:
        {technical_summary}
        
        LATEST NEWS/FUNDAMENTALS:
        {news_text}
        
        TASK:
        1. Explain WHY the stock is moving the way it is (match news to price action).
        2. Provide a 'Master Verdict' with a Confidence Score (0-100%).
        3. List key Long-term Risks and Potential Upside.
        
        Keep it concise, professional, and actionable. Use bullet points.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis Failed: {e}"

def calculate_expert_score(df, interval="1d"):
    """
    Calculates a comprehensive technical score with confidence logic and detailed breakdown.
    Returns at least 10 indicators with accurate values, sorted by reliability/impact.
    """
    if df is None or df.empty or len(df) < 50:
        return {
            "decision": "NEUTRAL", 
            "confidence": 0, 
            "current_price": 0,
            "summary": "Insufficient Data"
        }

    current_price = df['Close'].iloc[-1]
    
    # --- INDICATORS CALCULATION ---
    # 1. Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.willr(length=14, append=True) # Williams %R
    
    # 2. Trend
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.ema(length=9, append=True)  # Fast EMA
    df.ta.ema(length=20, append=True) # Short Trend
    df.ta.adx(length=14, append=True)
    df.ta.macds = df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.psar(append=True) # Parabolic SAR
    
    # 3. Volatility / Volume
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.obv(append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.cmf(length=20, append=True) # Chaikin Money Flow

    # 4. VWAP (Intraday)
    has_vwap = False
    if interval in ['5m', '15m', '30m', '1h']:
        try:
            df.ta.vwap(append=True)
            has_vwap = True
        except:
            pass
            
    # --- SAFE GETTERS ---
    def get_last(col_prefix):
        cols = [c for c in df.columns if c.startswith(col_prefix)]
        if cols:
            return df[cols[0]].iloc[-1]
        return None

    # Retrieve Values
    rsi = get_last('RSI')
    cci = get_last('CCI')
    stoch_k = get_last('STOCHk')
    willr = get_last('WILLR')
    
    sma_50 = get_last('SMA_50')
    sma_200 = get_last('SMA_200')
    ema_9 = get_last('EMA_9')
    ema_20 = get_last('EMA_20')
    adx = get_last('ADX')
    
    macd_val = get_last('MACD_')
    macd_sig = get_last('MACDs_')
    
    psar_l = get_last('PSARl') # Long
    psar_s = get_last('PSARs') # Short
    
    bb_upper = get_last('BBU')
    bb_lower = get_last('BBL')
    
    mfi = get_last('MFI')
    obv = get_last('OBV')
    cmf = get_last('CMF')
    atr = get_last('ATRr')
    
    vwap = get_last('VWAP') if has_vwap else None

    # --- SCORING & INSIGHT ENGINE ---
    score = 0
    max_score = 0
    
    bullish_factors = []
    bearish_factors = []
    neutral_factors = []
    
    # List to store full details for the table
    # Schema: {Indicator, Value, Signal, Impact(1-10), Reason}
    details = []
    
    def analyze(ind_name, val, weight, bullish_cond, bearish_cond, bull_r, bear_r, neutral_r):
        nonlocal score, max_score
        max_score += weight
        
        sig = "NEUTRAL"
        reason = neutral_r
        
        if bullish_cond:
            score += weight
            sig = "BULLISH"
            reason = bull_r
            bullish_factors.append(f"{ind_name}: {bull_r}")
        elif bearish_cond:
            score -= weight
            sig = "BEARISH"
            reason = bear_r
            bearish_factors.append(f"{ind_name}: {bear_r}")
        else:
            neutral_factors.append(f"{ind_name}: {neutral_r}")
            
        # Format Value
        val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        
        details.append({
            "Indicator": ind_name,
            "Value": val_str,
            "Signal": sig,
            "Reliability": weight,
            "Note": reason
        })

    # --- 1. TREND (High Reliability) ---
    if sma_200:
        analyze("SMA 200 (Long Term)", sma_200, 10, 
                current_price > sma_200, current_price < sma_200, 
                "Price > SMA 200 (Uptrend)", "Price < SMA 200 (Downtrend)", "At SMA Level")
                
    if ema_20:
        analyze("EMA 20 (Short Term)", ema_20, 8,
                current_price > ema_20, current_price < ema_20,
                "Price > EMA 20 (Strong Momentum)", "Price < EMA 20 (Weak Momentum)", "Consolidating")

    if macd_val is not None and macd_sig is not None:
        analyze("MACD (12,26,9)", macd_val - macd_sig, 8,
                macd_val > macd_sig, macd_val < macd_sig,
                "Histogram Positive (Bullish)", "Histogram Negative (Bearish)", "Flat")

    if adx:
        trend_str = "Strong" if adx > 25 else "Weak"
        # ADX doesn't give direction, just strength. We combine with EMA for direction context
        direction = "Bullish" if (ema_20 and current_price > ema_20) else "Bearish"
        analyze("ADX (Trend Strength)", adx, 7,
                adx > 25 and direction == "Bullish", adx > 25 and direction == "Bearish",
                "Strong Uptrend Strength", "Strong Downtrend Strength", "Weak Trend (<25)")

    if psar_l or psar_s:
        # PSAR logic: if PSARl exists (Long), price is above dots. If PSARs (Short), price below.
        val = psar_l if psar_l else psar_s
        is_bull = psar_l is not None and psar_l > 0
        analyze("Parabolic SAR", val, 6,
                is_bull, not is_bull,
                "Dots below price (Buy)", "Dots above price (Sell)", "N/A")

    # --- 2. MOMENTUM (Medium Reliability / Timing) ---
    if rsi:
        analyze("RSI (14)", rsi, 9,
                rsi < 30, rsi > 70,
                "Oversold (<30)", "Overbought (>70)", "Neutral Zone (30-70)")
                
    if stoch_k:
        analyze("Stochastic %K", stoch_k, 7,
                stoch_k < 20, stoch_k > 80,
                "Oversold (<20)", "Overbought (>80)", "Neutral")
                
    if cci:
        analyze("CCI (20)", cci, 6,
                cci < -100, cci > 100,
                "Oversold (<-100)", "Overbought (>100)", "Neutral")

    if willr:
        analyze("Williams %R", willr, 6,
                willr < -80, willr > -20,
                "Oversold (<-80)", "Overbought (>-20)", "Neutral")

    # --- 3. VOLUME & VOLATILITY (Confirmation) ---
    if mfi:
        analyze("MFI (Money Flow)", mfi, 7,
                mfi < 20, mfi > 80,
                "Smart Money Buying (Oversold)", "Smart Money Selling (Overbought)", "Neutral Flow")
                
    if cmf:
        analyze("Chaikin Money Flow", cmf, 6,
                cmf > 0.05, cmf < -0.05,
                "Inflows (Buying Pressure)", "Outflows (Selling Pressure)", "Neutral")
                
    if bb_upper and bb_lower:
        # Volatility squeeze or breakout?
        # Simple Mean Reversion logic
        analyze("Bollinger Bands", current_price, 6,
                current_price < bb_lower, current_price > bb_upper,
                "Price at Lower Band (Support)", "Price at Upper Band (Resistance)", "Inside Bands")
                
    if obv:
        # OBV is relative, harder to give a single signal without slope. 
        # We'll just display it with neutral weight unless we calc slope (omitted for speed)
        analyze("On-Balance Volume", obv, 4, False, False, "", "", "Cumulative Volume Metric")

    if atr:
         analyze("ATR (Volatility)", atr, 5, False, False, "", "", f"Daily Range is +/- {atr:.2f}")

    if vwap:
        analyze("VWAP (Intraday)", vwap, 8,
                current_price > vwap, current_price < vwap,
                "Price > VWAP (Bullish Control)", "Price < VWAP (Bearish Control)", "At VWAP")

    # --- FINAL SORTING ---
    # Sort by Reliability (High to Low), then by Signal Interest (Bull/Bear first)
    details.sort(key=lambda x: x['Reliability'], reverse=True)

    # --- DECISION LOGIC ---
    normalized_score = ((score + max_score) / (2 * max_score)) * 100 if max_score > 0 else 50
    
    decision = "HOLD"
    if normalized_score >= 60: decision = "BUY"
    elif normalized_score <= 40: decision = "SELL"
    
    strength = "WEAK"
    if 40 < normalized_score < 60: strength = "NEUTRAL"
    elif 60 <= normalized_score < 75: strength = "MODERATE"
    elif normalized_score >= 75: strength = "STRONG"
    elif 25 < normalized_score <= 40: strength = "MODERATE"
    elif normalized_score <= 25: strength = "STRONG"
    
    full_decision = f"{strength} {decision}" if decision != "HOLD" else "HOLD"

    return {
        "decision": full_decision,
        "confidence": round(normalized_score, 1),
        "current_price": current_price,
        "drivers": {
            "bullish": bullish_factors,
            "bearish": bearish_factors,
            "neutral": neutral_factors
        },
        "table_data": details,
        "metrics": {
            "rsi": rsi, "macd": macd_val, "macd_sig": macd_sig, 
            "sma200": sma_200, "sma50": sma_50, "ema20": ema_20,
            "adx": adx, "cci": cci
        }
    }

def generate_expert_outlook(analysis, news_items):
    """
    Generates a textual outlook based on the technical analysis and news.
    """
    m = analysis['metrics']
    drivers = analysis['drivers']
    
    # 1. Short-Term Scenarios (Days to Weeks)
    short_term_text = ""
    rsi = m.get('rsi')
    ema = m.get('ema20')
    price = analysis['current_price']
    
    if rsi and rsi < 30:
        short_term_text += "📉 **Oversold Bounce:** The stock is currently oversold (RSI < 30). This often precedes a short-term relief rally or 'dead cat bounce'. Watch for a reversal candle. "
    elif rsi and rsi > 70:
        short_term_text += "🔥 **Overheated:** Price is extended (RSI > 70). A pullback or consolidation is likely in the coming days to neutralize sentiment. "
    else:
        if ema and price > ema:
            short_term_text += "✅ **Momentum is Up:** Price is holding above the 20-day EMA, suggesting buyers are in control for the short term. "
        elif ema:
            short_term_text += "⚠️ **Momentum is Down:** Price is struggling below the 20-day EMA. Sellers are dominating intraday/short-term action. "
            
    # 2. Long-Term Scenarios (Months)
    long_term_text = ""
    sma200 = m.get('sma200')
    sma50 = m.get('sma50')
    
    if sma200:
        if price > sma200:
            long_term_text += "🚀 **Primary Uptrend:** The stock is in a healthy long-term uptrend (Above SMA 200). Dips should be viewed as buying opportunities. "
            if sma50 and sma50 > sma200:
                long_term_text += "Confirmed by a 'Golden Cross' formation. "
        else:
            long_term_text += "🐻 **Primary Downtrend:** The stock is in a secular decline (Below SMA 200). Rallies may be sold into. Caution is advised for long-term holders. "
            if sma50 and sma50 < sma200:
                long_term_text += "The 'Death Cross' confirms strong bearish grip. "

    # 3. Why the Score?
    b_count = len(drivers['bullish'])
    s_count = len(drivers['bearish'])
    score_expl = f"We analyzed **{b_count + s_count + len(drivers['neutral'])} technical factors**. "
    if b_count > s_count:
        score_expl += f"The **Bullish** case is dominant ({b_count} signals vs {s_count} bearish). Key drivers: {', '.join(drivers['bullish'][:2])}."
    elif s_count > b_count:
        score_expl += f"The **Bearish** case is dominant ({s_count} signals vs {b_count} bullish). Key risks: {', '.join(drivers['bearish'][:2])}."
    else:
        score_expl += "The market is currently **Indecisive**, with conflicting signals."

    # 4. News Catalyst
    catalyst = "No major specific news found."
    if news_items:
        top_news = news_items[0]
        catalyst = f"**Market Focus:** {top_news.get('title')} ({top_news.get('pubDate')})."

    return {
        "short_term": short_term_text,
        "long_term": long_term_text,
        "score_explanation": score_expl,
        "catalyst": catalyst
    }

def lookup_ticker(query, market_type="US"):
    """
    Attempts to find a ticker symbol from a company name query based on market type.
    """
    query = query.strip().upper()
    
    # Direct overrides for Commodities (Yahoo Finance symbols)
    COMMODITIES = {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "CRUDE OIL": "CL=F",
        "BRENT OIL": "BZ=F",
        "NATURAL GAS": "NG=F",
        "COPPER": "HG=F",
        "PLATINUM": "PL=F"
    }
    
    # Direct overrides for Crypto
    CRYPTO = {
        "BITCOIN": "BTC-USD",
        "ETHEREUM": "ETH-USD",
        "DOGECOIN": "DOGE-USD",
        "SOLANA": "SOL-USD",
        "RIPPLE": "XRP-USD",
        "CARDANO": "ADA-USD"
    }

    if market_type == "Commodities" and query in COMMODITIES:
        return COMMODITIES[query]
    
    if market_type == "Crypto":
        if query in CRYPTO: return CRYPTO[query]
        if not query.endswith("-USD"): return f"{query}-USD"
        return query

    # Heuristic: If it looks like a ticker, return it
    if " " not in query and (len(query) <= 8 or "." in query):
        if market_type == "India":
            if not query.endswith(".NS") and not query.endswith(".BO"):
                return f"{query}.NS" # Default to NSE
        return query
        
    # Search Fallback
    suffix_map = {
        "US": " stock symbol",
        "India": " share price nse ticker",
        "Crypto": " crypto ticker",
        "Commodities": " futures symbol yahoo finance"
    }
    
    search_query = f"{query} {suffix_map.get(market_type, 'stock symbol')}"
    
    try:
        results = DDGS().text(search_query, max_results=1)
        if results:
            text = results[0]['title'] + " " + results[0]['body']
            # Regex for Ticker inside parentheses e.g. (RELIANCE.NS)
            import re
            match = re.search(r'\((?P<ticker>[A-Z0-9.-]+)\)', text)
            if match:
                 t = match.group('ticker')
                 # Clean up common artifacts
                 return t.replace(")", "").replace("(", "")
    except:
        pass
    
    # Final Fallback
    if market_type == "India":
        return f"{query}.NS".replace(" ", "")
    
    return query

def find_similar_pattern(df, n_candles=60):
    """
    Finds a historical period where price action was similar.
    Returns: 
        - pattern_info: dict with date, score
        - historical_df: slice of DF containing the match + 30 future bars
    """
    if len(df) < n_candles * 4: # Need enough history
        return None, None
    
    # 1. Normalize recent price action (last n_candles)
    # Use Percent Change to match shape regardless of absolute price
    recent = df['Close'].iloc[-n_candles:]
    recent_norm = (recent - recent.mean()) / recent.std()
    
    best_corr = -1
    best_idx = 0
    
    # Slide through history, ensuring we have 30 days AFTER the window available
    history_closes = df['Close'].iloc[:-n_candles].values # Stop before overlapping with current
    
    search_limit = len(history_closes) - n_candles - 30 
    if search_limit < 1: return None, None
    
    # Optimization: Check every 2nd candle to speed up match
    for i in range(0, search_limit, 2):
        window = df['Close'].iloc[i : i+n_candles]
        window_norm = (window - window.mean()) / window.std()
        
        corr = recent_norm.corr(window_norm)
        
        if corr > best_corr:
            best_corr = corr
            best_idx = i
            
    if best_corr > 0.65: # Reasonable threshold
        # Extract Match + Future
        start_pos = best_idx
        end_pos = best_idx + n_candles + 30 # Show 30 candles into the future
        
        match_df = df.iloc[start_pos : end_pos].copy()
        
        # Add relative day counter for plotting overlay
        match_df['Day_Offset'] = range(-n_candles, len(match_df) - n_candles)
        
        # Metadata
        match_start = match_df.index[0].strftime('%Y-%m-%d')
        match_end = match_df.index[n_candles-1].strftime('%Y-%m-%d')
        
        return {
            "date": f"{match_start}",
            "score": round(best_corr * 100, 1),
            "hint": f"In {match_start}, similar behavior was followed by a move to ${match_df['Close'].iloc[-1]:.2f}"
        }, match_df
    
    return None, None

def get_market_news(ticker):
    """
    Fetches official news from YFinance.
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news
        results = []
        for n in news:
            # Handle nested structure (yfinance new API format)
            content = n.get('content', n) # Fallback to n if no content key
            
            title = content.get('title')
            
            # Link extraction
            link = content.get('link')
            click_through = content.get('clickThroughUrl')
            if not link and click_through:
                link = click_through.get('url')
            
            pub_date = content.get('pubDate') or content.get('providerPublishTime')
            provider = content.get('provider') or {}
            publisher = provider.get('displayName') or "Yahoo Finance"
            
            if title and link:
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": f"Published by {publisher}",
                    "pubDate": pub_date
                })
        return results[:5]
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"News error: {e}")
        return []

def analyze_stock_comprehensive(ticker, period, interval, api_key=None):
    df = fetch_data(ticker, period, interval)
    if df is None:
        return None, None, None, None, None
        
    analysis_res = calculate_expert_score(df, interval)
    
    # Pattern Match
    pattern_info, pattern_df = find_similar_pattern(df)
    analysis_res['pattern'] = pattern_info
    
    # Web Search for Context
    news = get_market_news(ticker)
    
    # Rule-Based Outlook (Replaces AI)
    outlook = generate_expert_outlook(analysis_res, news)
    
    return df, analysis_res, news, outlook, pattern_df
    
def backtest_stock(df, initial_capital=10000):
    """
    Performs a rigorous historical backtest of a Trend Following strategy.
    Strategy: BUY when Price > SMA_50 AND MACD > Signal. SELL when MACD < Signal.
    Returns statistical metrics: Win Rate, Total Return, Max Drawdown, etc.
    """
    if df is None or len(df) < 100:
        return None

    # Ensure we have necessary indicators (Should already be calculated, but safe to re-ensure)
    # We use vector operations for speed and precision
    try:
        if 'SMA_50' not in df.columns: df.ta.sma(length=50, append=True)
        if 'MACD_12_26_9' not in df.columns: df.ta.macd(fast=12, slow=26, signal=9, append=True)
    except:
        return None

    capital = initial_capital
    position = 0 # 0 = Flat, >0 = Number of shares
    entry_price = 0
    
    trades = [] # List of {entry_date, entry_price, exit_date, exit_price, profit, return_pct}
    equity_curve = [initial_capital]
    
    # We ignore the last candle as it might be incomplete/current
    # We iterate through the dataframe. 
    # NOTE: vectorized backtesting is faster, but looping allows for complex "state" (e.g. holding)
    
    # Signals
    # SMA 50 as trend filter. MACD as momentum trigger.
    # Logic: Only buy if Trend is UP (Price > SMA50) AND Momentum shifts UP (MACD > Signal)
    
    sma_col = 'SMA_50'
    macd_col = 'MACD_12_26_9'
    sig_col = 'MACDs_12_26_9'
    close_col = 'Close'
    dates = df.index
    
    # Pre-fetch numpy arrays for speed and precision
    closes = df[close_col].values
    smas = df[sma_col].values
    macds = df[macd_col].values
    sigs = df[sig_col].values
    
    for i in range(1, len(df) - 1):
        price = closes[i]
        date = dates[i]
        
        # Skip if NaN
        if pd.isna(smas[i]) or pd.isna(macds[i]):
            continue
            
        # BUY SIGNAL
        if position == 0:
            # Trend Filter: Price > SMA 50
            # Trigger: MACD crosses above Signal (Bullish Crossover)
            # We check if MACD[i] > Sig[i] AND MACD[i-1] <= Sig[i-1]
            bullish_cross = macds[i] > sigs[i] and macds[i-1] <= sigs[i-1]
            trend_up = price > smas[i]
            
            if bullish_cross and trend_up:
                position = capital / price
                entry_price = price
                entry_date = date
                capital = 0 # All in
                
        # SELL SIGNAL
        elif position > 0:
            # Exit when Momentum dies (MACD crosses below Signal) OR Stop Loss
            bearish_cross = macds[i] < sigs[i] and macds[i-1] >= sigs[i-1]
            
            if bearish_cross:
                revenue = position * price
                profit = revenue - (position * entry_price)
                pct_change = ((price - entry_price) / entry_price) * 100
                
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": date,
                    "exit_price": price,
                    "profit": profit,
                    "return_pct": pct_change
                })
                
                capital = revenue
                position = 0
                
        # Update Equity Curve (Mark to Market)
        current_equity = capital if position == 0 else (position * price)
        equity_curve.append(current_equity)

    # Metrics Calculation
    if not trades:
        return {
            "win_rate": 0,
            "total_return": 0,
            "total_trades": 0,
            "max_drawdown": 0,
            "best_trade": None,
            "status": "No historical trades found with this strategy."
        }
        
    wins = [t for t in trades if t['profit'] > 0]
    losses = [t for t in trades if t['profit'] <= 0]
    
    win_rate = (len(wins) / len(trades)) * 100
    
    # Calculate Total Return
    final_equity = equity_curve[-1]
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Max Drawdown
    peak = initial_capital
    max_dd = 0
    for val in equity_curve:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
        
    max_drawdown_pct = max_dd * 100
    
    # Best Scenario
    best_trade = max(trades, key=lambda x: x['return_pct']) if trades else None
    
    return {
        "win_rate": round(win_rate, 1),
        "total_return": round(total_return_pct, 1),
        "total_trades": len(trades),
        "max_drawdown": round(max_drawdown_pct, 1),
        "avg_profit": round(pd.DataFrame(trades)['return_pct'].mean(), 1),
        "best_trade": best_trade,
        "status": "Active"
    }

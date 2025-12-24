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
            limit_period = "60d" # Max for Yahoo Intraday
        elif interval == '1h':
            limit_period = "730d" 
        elif interval in ['1d', '1wk', '1mo']:
             limit_period = "max" # Get full history for Backtesting

        stock = yf.Ticker(ticker)
        df = stock.history(period=limit_period, interval=interval)
        
        if df.empty:
            return None
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None



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

def get_market_regime(market_type):
    """
    Checks the broad market trend (SPY, NIFTY, BTC) to filter signals.
    Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    try:
        ticker_map = {
            "US": "SPY",
            "India": "^NSEI",
            "Crypto": "BTC-USD",
            "Commodities": "GC=F" # Gold as general sentiment proxy? Or maybe simple Neutral.
        }
        
        index_ticker = ticker_map.get(market_type, "SPY")
        df = yf.Ticker(index_ticker).history(period="1y", interval="1d")
        
        if df.empty or len(df) < 200:
            return "NEUTRAL"
            
        current_price = df['Close'].iloc[-1]
        sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
        
        if current_price > sma_200:
            return "BULLISH"
        else:
            return "BEARISH"
    except:
        return "NEUTRAL"

def calculate_expert_score(df, interval="1d", market_type="US"):
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
    
    # Adaptive Logic for SMA 200
    has_sma200 = False
    if len(df) >= 200:
        df.ta.sma(length=200, append=True)
        has_sma200 = True
        
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

    # --- SCORING ENGINE (WEIGHTED GROUPS) ---
    # Weights: Trend (40%), Momentum (30%), Volatility (10%), Volume (20%)
    
    score_trend = 0
    score_momentum = 0
    score_volatility = 0
    score_volume = 0
    
    bullish_factors = []
    bearish_factors = []
    neutral_factors = []
    details = []
    
    def log_detail(name, val, sig, impact, note):
        details.append({
            "Indicator": name,
            "Value": f"{val:.2f}" if isinstance(val, (int, float)) else str(val),
            "Signal": sig,
            "Reliability": impact,
            "Note": note
        })
        if sig == "BULLISH": bullish_factors.append(f"{name}: {note}")
        elif sig == "BEARISH": bearish_factors.append(f"{name}: {note}")
        else: neutral_factors.append(f"{name}: {note}")

    # --- GROUP A: TREND (40%) ---
    # SMA 200, EMA 20, ADX
    # Max Trend Score: 10
    
    # 1. Long Term Trend (SMA 200) -> 4 pts
    if sma_200:
        if current_price > sma_200:
            score_trend += 4
            log_detail("SMA 200 (Trend)", sma_200, "BULLISH", 10, "Price > SMA 200 (Uptrend)")
        elif current_price < sma_200:
            score_trend -= 4
            log_detail("SMA 200 (Trend)", sma_200, "BEARISH", 10, "Price < SMA 200 (Downtrend)")
        else:
            log_detail("SMA 200 (Trend)", sma_200, "NEUTRAL", 10, "Price at SMA 200")

    # 2. Short Term Trend (EMA 20) -> 3 pts
    if ema_20:
        if current_price > ema_20:
            score_trend += 3
            log_detail("EMA 20 (Momentum)", ema_20, "BULLISH", 8, "Price > EMA 20")
        elif current_price < ema_20:
            score_trend -= 3
            log_detail("EMA 20 (Momentum)", ema_20, "BEARISH", 8, "Price < EMA 20")
            
    # 3. Trend Strength (ADX) -> 3 pts
    if adx:
        trend_dir = "BULLISH" if (ema_20 and current_price > ema_20) else "BEARISH"
        if adx > 25:
            if trend_dir == "BULLISH":
                score_trend += 3
                log_detail("ADX Strength", adx, "BULLISH", 7, "Strong Uptrend")
            else:
                score_trend -= 3
                log_detail("ADX Strength", adx, "BEARISH", 7, "Strong Downtrend")
        else:
            log_detail("ADX Strength", adx, "NEUTRAL", 5, "Weak Trend (<25)")

    # Normalize Trend Score to 40% (Range -10 to +10 maps to -40 to +40)
    final_trend = score_trend * 4 

    # --- GROUP B: MOMENTUM (30%) ---
    # RSI, Stoch, MACD, CCI. VOTE SYSTEM: 1 Vote Max.
    # Logic: Count Bullish vs Bearish signals. Majority wins the full 30 points.
    
    mom_bull = 0
    mom_bear = 0
    
    # RSI
    if rsi:
        if rsi < 30: 
            mom_bull += 1
            log_detail("RSI (14)", rsi, "BULLISH", 9, "Oversold (<30)")
        elif rsi > 70: 
            mom_bear += 1
            log_detail("RSI (14)", rsi, "BEARISH", 9, "Overbought (>70)")
        else:
            log_detail("RSI (14)", rsi, "NEUTRAL", 6, "Neutral Zone")
            
    # Stoch
    if stoch_k:
        if stoch_k < 20: 
            mom_bull += 1
            log_detail("Stochastic", stoch_k, "BULLISH", 7, "Oversold (<20)")
        elif stoch_k > 80: 
            mom_bear += 1
            log_detail("Stochastic", stoch_k, "BEARISH", 7, "Overbought (>80)")
        else:
            log_detail("Stochastic", stoch_k, "NEUTRAL", 5, "Neutral")

    # MACD
    if macd_val is not None and macd_sig is not None:
        if macd_val > macd_sig:
            mom_bull += 1
            log_detail("MACD", macd_val-macd_sig, "BULLISH", 8, "Bullish Crossover/Hist")
        else:
            mom_bear += 1
            log_detail("MACD", macd_val-macd_sig, "BEARISH", 8, "Bearish Crossover/Hist")
            
    # Calculate Momentum Score
    final_momentum = 0
    if mom_bull > mom_bear:
        final_momentum = 30
    elif mom_bear > mom_bull:
        final_momentum = -30
        
    # --- GROUP C: VOLUME (20%) ---
    # OBV, MFI, CMF
    
    vol_score_raw = 0
    # MFI -> 7 pts
    if mfi:
        if mfi < 20: 
            vol_score_raw += 7
            log_detail("MFI (Money Flow)", mfi, "BULLISH", 7, "Smart Money Buying")
        elif mfi > 80: 
            vol_score_raw -= 7
            log_detail("MFI (Money Flow)", mfi, "BEARISH", 7, "Smart Money Selling")
        else:
            log_detail("MFI (Money Flow)", mfi, "NEUTRAL", 5, "Neutral Flow")
            
    # CMF -> 7 pts
    if cmf:
        if cmf > 0.05: 
            vol_score_raw += 7
            log_detail("Chaikin Money Flow", cmf, "BULLISH", 6, "Inflows (Accumulation)")
        elif cmf < -0.05: 
            vol_score_raw -= 7
            log_detail("Chaikin Money Flow", cmf, "BEARISH", 6, "Outflows (Distribution)")
        else:
            log_detail("Chaikin Money Flow", cmf, "NEUTRAL", 5, "Neutral")

    # OBV Slope -> 6 pts
    if obv is not None and len(df) > 20:
        obv_series = df['OBV']
        obv_ema = obv_series.ewm(span=20).mean().iloc[-1]
        if obv > obv_ema:
            vol_score_raw += 6
            log_detail("On-Balance Volume", obv, "BULLISH", 6, "Volume Trend Rising")
        else:
            vol_score_raw -= 6
            log_detail("On-Balance Volume", obv, "BEARISH", 6, "Volume Trend Falling")

    final_volume = vol_score_raw 
    
    # --- GROUP D: VOLATILITY (10%) ---
    # BBands, ATR, VIX
    
    final_volatility = 0
    
    # 1. Bollinger Bands -> 4 pts
    if bb_upper and bb_lower:
        if current_price < bb_lower:
            final_volatility += 4
            log_detail("Bollinger Bands", current_price, "BULLISH", 6, "Price at Support")
        elif current_price > bb_upper:
            final_volatility -= 4
            log_detail("Bollinger Bands", current_price, "BEARISH", 6, "Price at Resistance")
        else:
            log_detail("Bollinger Bands", current_price, "NEUTRAL", 5, "Inside Bands")
            
    # 2. ATR -> 3 pts
    if atr:
        atr_pct = (atr / current_price) * 100
        if atr_pct > 3.0:
            final_volatility -= 3
            log_detail("ATR (Volatility)", atr, "BEARISH", 4, f"High Volatility ({atr_pct:.1f}%)")
        else:
            final_volatility += 3
            log_detail("ATR (Volatility)", atr, "NEUTRAL", 4, f"Normal Volatility ({atr_pct:.1f}%)")

    # 3. VIX -> 3 pts
    vix_ticker = "^VIX" if market_type == "US" else "^INDIAVIX" if market_type == "India" else None
    if vix_ticker:
        try:
            vix_df = yf.Ticker(vix_ticker).history(period="5d")
            if not vix_df.empty:
                vix_val = vix_df['Close'].iloc[-1]
                if vix_val > 30:
                    final_volatility -= 3
                    log_detail(f"VIX ({market_type})", vix_val, "BEARISH", 8, "Extreme Fear (>30)")
                elif vix_val < 20:
                    final_volatility += 3
                    log_detail(f"VIX ({market_type})", vix_val, "BULLISH", 8, "Market Stable (<20)")
                else:
                    log_detail(f"VIX ({market_type})", vix_val, "NEUTRAL", 5, "Normal Regime")
        except:
             pass

    # --- FINAL AGGREGATION ---
    total_score = final_trend + final_momentum + final_volume + final_volatility
    # Range: -100 to +100
    
    # Normalize to 0-100 for display
    normalized_score = (total_score + 100) / 2
    
    decision = "HOLD"
    if normalized_score >= 60: decision = "BUY"
    elif normalized_score <= 40: decision = "SELL"
    
    strength = "WEAK"
    if 40 < normalized_score < 60: strength = "NEUTRAL"
    elif 60 <= normalized_score < 75: strength = "MODERATE"
    elif normalized_score >= 75: strength = "STRONG"
    elif 25 < normalized_score <= 40: strength = "MODERATE"
    elif normalized_score <= 25: strength = "STRONG"
    
    # --- MARKET REGIME FILTER ---
    # Downgrade STRONG BUY in Bear Market
    regime = get_market_regime(market_type)
    regime_msg = ""
    
    warnings = []
    if len(df) < 200:
        warnings.append("⚠️ Data limited: Switched to Short-Term Analysis (50-SMA) due to lack of history.")
    
    if regime == "BEARISH" and "STRONG BUY" in f"{strength} {decision}":
        strength = "MODERATE"
        regime_msg = f" (Downgraded due to Bearish {market_type} Market)"
        if market_type == "US": regime_msg = " (Downgraded: SPY < SMA200)"
        if market_type == "India": regime_msg = " (Downgraded: NIFTY < SMA200)"
        
    full_decision = f"{strength} {decision}{regime_msg}" if decision != "HOLD" else "HOLD"

    details.sort(key=lambda x: x['Reliability'], reverse=True)

    return {
        "decision": full_decision,
        "confidence": round(normalized_score), # Round to whole number
        "current_price": current_price,
        "warnings": warnings,
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
    query = query.strip()
    q_upper = query.upper()
    
    # Direct overrides
    COMMODITIES = {
        "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE OIL": "CL=F", "BRENT OIL": "BZ=F",
        "NATURAL GAS": "NG=F", "COPPER": "HG=F", "PLATINUM": "PL=F"
    }
    CRYPTO = {
        "BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "DOGECOIN": "DOGE-USD",
        "SOLANA": "SOL-USD", "RIPPLE": "XRP-USD", "CARDANO": "ADA-USD",
        "BNB": "BNB-USD", "TETHER": "USDT-USD"
    }

    if market_type == "Commodities" and q_upper in COMMODITIES:
        return COMMODITIES[q_upper]
    
    if market_type == "Crypto":
        if q_upper in CRYPTO: return CRYPTO[q_upper]
        if not q_upper.endswith("-USD") and " " not in query: return f"{q_upper}-USD"
        # If space, fall through to search

    # --- HEURISTIC CHECK ---
    # If it looks like a ticker, return it immediately to save time
    is_likely_ticker = False
    
    # Condition A: It is already uppercase and no spaces (e.g. AAPL, TSLA)
    if " " not in query and query.isupper():
        is_likely_ticker = True
    # Condition B: Contains explicit ticker chars like dot or number (e.g. BRK.B, 500123)
    elif " " not in query and ("." in query or any(char.isdigit() for char in query)):
        is_likely_ticker = True
    # Condition C: Explicit suffix
    elif query.lower().endswith(".ns") or query.lower().endswith(".bo"):
        is_likely_ticker = True
        
    if is_likely_ticker:
        return q_upper

    # --- SEARCH OVERRIDES ---
    # Common big names manual fix (Optimization)
    manual_map = {
        "APPLE": "AAPL", "TESLA": "TSLA", "MICROSOFT": "MSFT", 
        "GOOGLE": "GOOGL", "AMAZON": "AMZN", "NVIDIA": "NVDA", 
        "META": "META", "NETFLIX": "NFLX", "RELIANCE": "RELIANCE.NS",
        "TATA MOTORS": "TATAMOTORS.NS", "HDFC": "HDFCBANK.NS", "INFOSYS": "INFY.NS"
    }
    if q_upper in manual_map:
        return manual_map[q_upper]

    # --- SEARCH FALLBACK ---
    suffix_map = {
        "US": " stock symbol",
        "India": " share price nse ticker",
        "Crypto": " crypto ticker",
        "Commodities": " futures symbol yahoo finance"
    }
    
    search_query = f"{query} {suffix_map.get(market_type, 'stock symbol')}"
    print(f"DEBUG: Searching for '{search_query}'...")
    
    try:
        # Using DuckDuckGo Search
        # Attempt 1: Look for parenthesis pattern (e.g. "Apple Inc. (AAPL)")
        results = DDGS().text(search_query, max_results=1)
        if results:
            title = results[0]['title'] 
            body = results[0]['body']
            combined_text = f"{title} {body}"
            
            import re
            # Pattern 1: (TICKER) - most common in search results
            match = re.search(r'\(\s?(?P<ticker>[A-Z0-9.-]+)\s?\)', combined_text)
            if match:
                t = match.group('ticker')
                # Clean up
                t = t.replace("NYSE:", "").replace("NASDAQ:", "").strip()
                if market_type == "India" and not t.endswith(".NS") and not t.endswith(".BO"):
                     # Some results might just say RELIANCE without suffix
                     return f"{t}.NS"
                return t
                
            # Pattern 2: "Symbol: TICKER"
            match_sym = re.search(r'Symbol:\s?(?P<ticker>[A-Z0-9.-]+)', combined_text, re.IGNORECASE)
            if match_sym:
                 t = match_sym.group('ticker')
                 if market_type == "India" and not t.endswith(".NS"): return f"{t}.NS"
                 return t

    except Exception as e:
        print(f"Lookup Warning: {e}")
    
    # Fallback if search fails but it was a single word
    if " " not in query:
        if market_type == "India":
            return f"{q_upper}.NS"
        return q_upper
        
    return q_upper

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

def analyze_stock_comprehensive(ticker, period, interval, market_type="US", api_key=None):
    df = fetch_data(ticker, period, interval)
    if df is None:
        return None, None, None, None, None
        
    analysis_res = calculate_expert_score(df, interval, market_type)
    
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

    # Signal Generation (Simulated at T-1 Close for execution at T Open)
    # 1. Check Sufficiency
    if len(df) < 50:
         return {
            "status": "Insufficient Data (<50 candles)",
            "win_rate": 0, "total_return": 0, "total_trades": 0, "max_drawdown": 0, "sharpe_ratio": 0
        }

    # 2. Prepare Vectorized Columns (Subset first to avoid dropping rows due to unused indicators like SMA200)
    # We only need Open, Close, SMA_50, MACD, Signal
    req_cols = ['Open', 'Close', 'SMA_50', 'MACD_12_26_9', 'MACDs_12_26_9']
    
    # Check if they exist
    for c in req_cols:
        if c not in df.columns:
            return {
                "status": f"Missing Indicator: {c}",
                "win_rate": 0, "total_return": 0, "total_trades": 0, "max_drawdown": 0, "sharpe_ratio": 0
            }
            
    df_bt = df[req_cols].copy()
    
    # Create shifted columns for T-1 Logic
    df_bt['Prev_MACD'] = df_bt['MACD_12_26_9'].shift(1)
    df_bt['Prev_Signal'] = df_bt['MACDs_12_26_9'].shift(1)
    df_bt['Prev2_MACD'] = df_bt['MACD_12_26_9'].shift(2)
    df_bt['Prev2_Signal'] = df_bt['MACDs_12_26_9'].shift(2)
    df_bt['Prev_Close'] = df_bt['Close'].shift(1)
    df_bt['Prev_SMA50'] = df_bt['SMA_50'].shift(1)
    
    # Now Safe Drop
    df_bt.dropna(inplace=True)
    
    if len(df_bt) < 30:
        return {
            "status": "Insufficient Data (After dropping NaNs)",
            "win_rate": 0, "total_return": 0, "total_trades": 0, "max_drawdown": 0, "sharpe_ratio": 0
        }

    capital = initial_capital
    position = 0
    entry_price = 0

    trades = [] 
    equity_curve = [initial_capital]
    
    # Vectorized logic is tricky with state (holding vs not).
    # We use a loop on the clean dataset.
    
    opens = df_bt['Open'].values
    prev_macds = df_bt['Prev_MACD'].values
    prev_sigs = df_bt['Prev_Signal'].values
    prev2_macds = df_bt['Prev2_MACD'].values
    prev2_sigs = df_bt['Prev2_Signal'].values
    prev_closes = df_bt['Prev_Close'].values
    prev_smas = df_bt['Prev_SMA50'].values
    dates = df_bt.index
    
    for i in range(len(df_bt)):
        current_date = dates[i]
        execution_price = opens[i] # Trade at Open
        
        # Strategy Logic:
        # BUY IF: at Prev Close, Price > SMA50 AND MACD Crossed Above Signal
        
        # Check Crossover occurred at T-1
        # (Prev_MACD > Prev_Signal) AND (Prev2_MACD <= Prev2_Signal)
        bullish_cross = (prev_macds[i] > prev_sigs[i]) and (prev2_macds[i] <= prev2_sigs[i])
        trend_up = prev_closes[i] > prev_smas[i]
        
        if position == 0:
            if bullish_cross and trend_up:
                position = capital / execution_price
                entry_price = execution_price
                entry_date = current_date
                capital = 0 
                
        elif position > 0:
            # SELL IF: at Prev Close, MACD Crossed Below Signal
            bearish_cross = (prev_macds[i] < prev_sigs[i]) and (prev2_macds[i] >= prev2_sigs[i])
            
            if bearish_cross:
                revenue = position * execution_price
                
                # Transaction Cost (0.1% Slippage/Clerical Fees)
                raw_pct = ((execution_price - entry_price) / entry_price) * 100
                pct_change = raw_pct - 0.1
                
                # Adjust absolute profit (approx)
                cost = (position * entry_price) * 0.001
                profit = revenue - (position * entry_price) - cost 
                
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": execution_price,
                    "profit": profit,
                    "return_pct": pct_change
                })
                
                capital = revenue
                position = 0
        
        # Mark to Market (using Close of current day)
        current_val = capital if position == 0 else (position * df_bt['Close'].iloc[i])
        equity_curve.append(current_val)
        
    # --- METRICS ---
    if not trades:
         return {
            "win_rate": 0, "total_return": 0, "total_trades": 0, "max_drawdown": 0, "sharpe_ratio": 0,
            "status": "No Trades Triggered"
        }

    trades_df = pd.DataFrame(trades)
    win_rate = (len(trades_df[trades_df['profit'] > 0]) / len(trades_df)) * 100
    total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
    
    # Max DD
    peak = initial_capital
    max_dd = 0
    for val in equity_curve:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
        
    # Sharpe Ratio (Daily Returns)
    equity_series = pd.Series(equity_curve)
    daily_returns = equity_series.pct_change().dropna()
    
    risk_free_daily = 0.04 / 252 # 4% Annual
    excess_returns = daily_returns - risk_free_daily
    
    if excess_returns.std() > 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * (252**0.5)
    else:
        sharpe = 0
        
    warning = ""
    if len(trades) < 30:
        warning = "⚠️ Low Statistical Significance (N < 30)"

    return {
        "win_rate": round(win_rate, 0),
        "total_return": round(total_return, 1),
        "total_trades": len(trades),
        "max_drawdown": round(max_dd * 100, 1),
        "avg_profit": round(trades_df['return_pct'].mean(), 1),
        "best_trade": max(trades, key=lambda x: x['return_pct']),
        "sharpe_ratio": round(sharpe, 2),
        "warning": warning,
        "status": "Active"
    }

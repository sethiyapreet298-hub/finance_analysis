import analysis
import pandas as pd

def test_logic():
    print("Testing Ticker Lookup...")
    try:
        sym = analysis.lookup_ticker("Apple")
        print(f"Lookup 'Apple' -> {sym}")
        assert sym == "AAPL"
    except Exception as e:
        print(f"Lookup failed: {e}")

    print("\nTesting Daily...")
    # Note: unpacking 5 items now
    df, res, news, ai, pat_df = analysis.analyze_stock_comprehensive("AAPL", "1y", "1d", None)
    if df is not None:
        print(f"Daily Close: {df['Close'].iloc[-1]}")
        print(f"Daily Verdict: {res['decision']} ({res['confidence']}%)")
        
        # Check Table Data Education
        table = res.get('table_data', [])
        if table:
            print(f"First Ind Edu: {table[0].get('Education')}")
            
        # Check Pattern DF
        if pat_df is not None:
             print(f"Pattern DF Shape: {pat_df.shape}")
             print(f"Historical Match Date: {res['pattern']['date']}")
        else:
             print("No pattern match found (might be short history)")
             
        print(f"News Items: {len(news)}")
    else:
        print("Daily Failed")

    print("\nTesting Intraday (5m)...")
    # Intraday might not return pattern df effectively if history is short in fetch
    df_intra, res_intra, _, _, _ = analysis.analyze_stock_comprehensive("AAPL", "1d", "5m", None)
    if df_intra is not None:
         print(f"Intraday Close: {df_intra['Close'].iloc[-1]}")
    else:
        print("Intraday Failed")

if __name__ == "__main__":
    test_logic()

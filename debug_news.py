import yfinance as yf
t = yf.Ticker("AAPL")
news = t.news
import json
if news:
    print(json.dumps(news[0], indent=2, default=str))
else:
    print("No news found")

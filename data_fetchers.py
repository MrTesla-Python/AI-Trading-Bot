import yfinance as yf
import pandas as pd

df = yf.download("NQ=F", period="1mo", interval="5m")
df.dropna(inplace=True)
df.to_csv("nq_5m_data.csv")
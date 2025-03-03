##########################################################
# 1) FORCE THE PATCH ON BOKEH FIRST, BEFORE ANYTHING ELSE
##########################################################
import bokeh.models.formatters as fm

_OriginalDTF = fm.DatetimeTickFormatter

class PatchedDatetimeTickFormatter(_OriginalDTF):
    def __init__(self, *args, **kwargs):
        # If ANY interval is a list, pick the first item
        intervals = [
            'microseconds','milliseconds','seconds','minsec',
            'minutes','hourmin','hours','days','months','years'
        ]
        for key in intervals:
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, list) and len(val) > 0:
                    kwargs[key] = val[0]  # e.g. pick "%d %b"
        super().__init__(*args, **kwargs)

fm.DatetimeTickFormatter = PatchedDatetimeTickFormatter


##########################################################
# 2) NOW IMPORT BACKTESTING (WHICH EVENTUALLY IMPORTS BOKEH)
##########################################################
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy

# --- Indicators
def parabolic_sar(high, low, acceleration=0.02, maximum=0.2):
    psar = np.zeros(len(high))
    psar[0] = low[0]
    direction = 1
    af = acceleration
    ep = high[0]
    for i in range(1, len(high)):
        prev_psar = psar[i-1]
        if direction == 1:
            psar[i] = prev_psar + af*(ep - prev_psar)
            if psar[i] > low[i]:
                direction = -1
                psar[i] = ep
                af = acceleration
                ep = low[i]
        else:
            psar[i] = prev_psar - af*(prev_psar - ep)
            if psar[i] < high[i]:
                direction = 1
                psar[i] = ep
                af = acceleration
                ep = high[i]
        if direction == 1 and high[i] > ep:
            ep = high[i]
            af = min(af + acceleration, maximum)
        elif direction == -1 and low[i] < ep:
            ep = low[i]
            af = min(af + acceleration, maximum)
    return pd.Series(psar, index=high.index)

def lucid_sar(high, low, acceleration=0.01, maximum=0.1):
    return parabolic_sar(high, low, acceleration, maximum)

def true_range(df):
    high = df["high"]
    low  = df["low"]
    close= df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def chop_index(df, length=14):
    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()
    tr = true_range(df)
    numer = (hh - ll) / tr
    return 100 * np.log10(numer) / np.log10(length)

# --- Strategy
class FiveMinBotScalper(Strategy):
    sarAcc   = 0.32
    sarMax   = 0.2
    lucidAcc = 0.01
    lucidMax = 0.1
    chopLen  = 14
    chopTh   = 55.0
    maShortL = 50
    maLongL  = 200
    slPct    = 1.0
    tpPct    = 2.0

    def init(self):
        high  = self.data.High
        low   = self.data.Low
        close = self.data.Close

        self.maShort = self.I(lambda c: pd.Series(c).ewm(span=self.maShortL).mean().values, close)
        self.maLong  = self.I(lambda c: pd.Series(c).ewm(span=self.maLongL).mean().values, close)

        def psar_func(h, l):
            return parabolic_sar(pd.Series(h), pd.Series(l), self.sarAcc, self.sarMax).values
        self.psar = self.I(psar_func, high, low)

        def lucid_sar_func(h, l):
            return lucid_sar(pd.Series(h), pd.Series(l), self.lucidAcc, self.lucidMax).values
        self.lucidSar = self.I(lucid_sar_func, high, low)

        def chop_func(h, l, c):
            tmp = pd.DataFrame({"high": h, "low": l, "close": c})
            return chop_index(tmp, self.chopLen).values
        self.chop = self.I(chop_func, high, low, close)

    def next(self):
        psarVal     = self.psar[-1]
        lucidSarVal = self.lucidSar[-1]
        chopVal     = self.chop[-1]
        maShortVal  = self.maShort[-1]
        maLongVal   = self.maLong[-1]
        price       = self.data.Close[-1]

        trending    = chopVal < self.chopTh
        above200    = price > maLongVal
        below200    = price < maLongVal
        goldenCross = maShortVal > maLongVal
        deathCross  = maShortVal < maLongVal

        sarBullish   = price > psarVal
        sarBearish   = price < psarVal
        lucidBullish = price > lucidSarVal
        lucidBearish = price < lucidSarVal

        longCond  = trending and above200 and goldenCross and sarBullish and lucidBullish
        shortCond = trending and below200 and deathCross  and sarBearish  and lucidBearish

        exitLongCond  = sarBearish or lucidBearish
        exitShortCond = sarBullish or lucidBullish

        if not self.position:
            if longCond:
                sl = price * (1.0 - self.slPct/100)
                tp = price * (1.0 + self.tpPct/100)
                self.buy(sl=sl, tp=tp)
            elif shortCond:
                sl = price * (1.0 + self.slPct/100)
                tp = price * (1.0 - self.tpPct/100)
                self.sell(sl=sl, tp=tp)
        else:
            if self.position.is_long and exitLongCond:
                self.position.close()
            elif self.position.is_short and exitShortCond:
                self.position.close()

def main():
    df = yf.download("NQ=F", period="10d", interval="5m")
    df.dropna(how="all", inplace=True)

    # Flatten multi-index if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(level) for level in col if level) for col in df.columns
        ]
    # drop "Adj Close"
    to_drop = [c for c in df.columns if "adj close" in c.lower()]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)

    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if "open" in lc:
            rename_map[col] = "Open"
        elif "high" in lc:
            rename_map[col] = "High"
        elif "low" in lc:
            rename_map[col] = "Low"
        elif "close" in lc:
            rename_map[col] = "Close"
        elif "volume" in lc:
            rename_map[col] = "Volume"

    df.rename(columns=rename_map, inplace=True)

    needed = ["Open","High","Low","Close","Volume"]
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=needed, inplace=True)

    bt = Backtest(df, FiveMinBotScalper, cash=100_000, commission=.000, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot()

if __name__ == "__main__":
    main()

###############################################################
# PATCH: Ensure Plot_OldSync has an 'mpyplot' attribute (for matplotlib)
###############################################################
try:
    import backtrader.plot.plot as btplot
    import matplotlib.pyplot as plt
    if not hasattr(btplot.Plot_OldSync, 'mpyplot'):
        btplot.Plot_OldSync.mpyplot = plt
        print("Patched Plot_OldSync: 'mpyplot' attribute set to matplotlib.pyplot")
except Exception as e:
    print("Could not patch Plot_OldSync:", e)

###############################################################
# THE REST OF THE CODE
###############################################################
import math
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt

###############################################################################
# 1) CUSTOM INDICATORS
###############################################################################

class ChopIndex(bt.Indicator):
    """
    Chop Index = 100 * log10((HH - LL) / ATR(1)) / log10(length)
    where:
      - HH = highest high over 'chop_len' bars,
      - LL = lowest low over 'chop_len' bars,
      - ATR(1) is approximated using the current bar's true range.
    """
    params = (("chop_len", 14),)
    lines = ("chop",)

    def __init__(self):
        self.addminperiod(self.p.chop_len)

    def next(self):
        if len(self.data) < self.p.chop_len:
            self.lines.chop[0] = 0
            return
        
        length = self.p.chop_len
        recent_high = max(list(self.data.high.get(0, length)))
        recent_low  = min(list(self.data.low.get(0, length)))
        if len(self.data.close) > 1:
            prev_close = self.data.close[-1]
        else:
            prev_close = self.data.close[0]
        tr1 = self.data.high[0] - self.data.low[0]
        tr2 = abs(self.data.high[0] - prev_close)
        tr3 = abs(self.data.low[0]  - prev_close)
        tr_current = max(tr1, tr2, tr3)
        if tr_current == 0:
            self.lines.chop[0] = 0
            return
        numer = (recent_high - recent_low) / tr_current
        self.lines.chop[0] = 100 * math.log10(numer) / math.log10(length)

class LucidSAR(bt.Indicator):
    """
    Custom Lucid SAR indicator.
    Uses parameters: acceleration = 0.01, maximum = 0.1.
    """
    params = (("acceleration", 0.01), ("accel_max", 0.1))
    lines = ("lucid_sar",)

    def __init__(self):
        self.addminperiod(1)
        self._direction = 1
        self._af = self.p.acceleration
        self._ep = self.data.high[0]
        self._sar = self.data.low[0]

    def next(self):
        prev_sar = self._sar
        direction = self._direction
        af = self._af
        ep = self._ep

        if direction == 1:
            curr_sar = prev_sar + af * (ep - prev_sar)
            if curr_sar > self.data.low[0]:
                direction = -1
                curr_sar = ep
                af = self.p.acceleration
                ep = self.data.low[0]
            else:
                if self.data.high[0] > ep:
                    ep = self.data.high[0]
                    af = min(af + self.p.acceleration, self.p.accel_max)
        else:
            curr_sar = prev_sar - af * (prev_sar - ep)
            if curr_sar < self.data.high[0]:
                direction = 1
                curr_sar = ep
                af = self.p.acceleration
                ep = self.data.high[0]
            else:
                if self.data.low[0] < ep:
                    ep = self.data.low[0]
                    af = min(af + self.p.acceleration, self.p.accel_max)
        self._sar = curr_sar
        self._direction = direction
        self._af = af
        self._ep = ep
        self.lines.lucid_sar[0] = curr_sar

###############################################################################
# 2) STRATEGY: FIVE MIN BOT SCALPER (Replicating Your Pine Script Logic)
###############################################################################
class FiveMinBotScalper(bt.Strategy):
    params = dict(
        sar_acc=0.32,
        sar_max=0.2,
        lucid_acc=0.01,
        lucid_max=0.1,
        chop_len=14,
        chop_th=55.0,
        ma_short_len=50,
        ma_long_len=200,
        sl_pct=1.0,   # Stop Loss percentage
        tp_pct=2.0    # Take Profit percentage
    )
    
    def __init__(self):
        if len(self.data) < self.p.ma_long_len:
            return
        self.sar = bt.ind.ParabolicSAR(self.data, af=self.p.sar_acc, afmax=self.p.sar_max)
        self.lucid_sar = LucidSAR(self.data, acceleration=self.p.lucid_acc, accel_max=self.p.lucid_max)
        self.chop = ChopIndex(self.data, chop_len=self.p.chop_len)
        self.ma_short = bt.ind.EMA(self.data.close, period=self.p.ma_short_len)
        self.ma_long  = bt.ind.EMA(self.data.close, period=self.p.ma_long_len)
    
    def next(self):
        if len(self.data) < self.p.ma_long_len:
            return
        price = self.data.close[0]
        trending = self.chop[0] < self.p.chop_th
        above200 = price > self.ma_long[0]
        below200 = price < self.ma_long[0]
        goldenCross = self.ma_short[0] > self.ma_long[0]
        deathCross = self.ma_short[0] < self.ma_long[0]
        sarBullish = price > self.sar[0]
        sarBearish = price < self.sar[0]
        lucidBullish = price > self.lucid_sar[0]
        lucidBearish = price < self.lucid_sar[0]
        
        longCond = trending and above200 and goldenCross and sarBullish and lucidBullish
        shortCond = trending and below200 and deathCross and sarBearish and lucidBearish
        
        exitLongCond = sarBearish or lucidBearish
        exitShortCond = sarBullish or lucidBullish
        
        if not self.position:
            if longCond:
                sl_price = price * (1.0 - self.p.sl_pct/100)
                tp_price = price * (1.0 + self.p.tp_pct/100)
                self.buy_bracket(stopprice=sl_price, limitprice=tp_price)
            elif shortCond:
                sl_price = price * (1.0 + self.p.sl_pct/100)
                tp_price = price * (1.0 - self.p.tp_pct/100)
                self.sell_bracket(stopprice=sl_price, limitprice=tp_price)
        else:
            if self.position.size > 0 and exitLongCond:
                self.close()
            elif self.position.size < 0 and exitShortCond:
                self.close()

###############################################################################
# 3) DATA FETCH & PREPARATION (using start and end dates)
###############################################################################
def fetch_data(symbol="NQ=F", start=None, end=None, period="5d", interval="5m"):
    # If start and end are provided, use them; else use period.
    if start is not None and end is not None:
        df = yf.download(symbol, start=start, end=end, interval=interval)
    else:
        df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(level) for level in col if level) for col in df.columns]
    else:
        df.columns = [str(col) for col in df.columns]
    to_drop = [c for c in df.columns if "adj close" in c.lower()]
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors="ignore")
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
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns: {df.columns.tolist()}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=needed, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

###############################################################################
# 4) RUN THE BACKTEST
###############################################################################
def run_backtest(symbol="NQ=F", period = "5d", interval="5m"):
    # If start and end are provided, use them. Else use period.
    df = fetch_data(symbol, period=period, interval=interval)
    print("Sample of Processed Data:")
    print(df.head())
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(FiveMinBotScalper)
    
    datafeed = bt.feeds.PandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Minutes,
        compression=5,  # 5-minute bars
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume"
    )
    cerebro.adddata(datafeed)
    
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0002)
    
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: ${:.2f}".format(final_value))
    
    cerebro.plot(iplot=False)

###############################################################################
# 5) MAIN
###############################################################################
if __name__ == "__main__":
    # Use start and end to specify one month (adjust if needed)
    run_backtest(symbol="NQ=F", period="5d", interval="5m")

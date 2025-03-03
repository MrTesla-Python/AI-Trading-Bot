# strategy_code.py

from backtesting import Strategy
import pandas as pd
import numpy as np
from custom_indicators import (
    parabolic_sar,
    lucid_sar,
    chop_index,
)

class FiveMinBotScalper(Strategy):
    # Default parameters
    sarAcc   = 0.32
    sarMax   = 0.20
    lucidAcc = 0.01
    lucidMax = 0.10
    chopLen  = 14
    chopTh   = 55.0
    maShortL = 50
    maLongL  = 200
    slPct    = 1.0   # stop loss %
    tpPct    = 2.0   # take profit %

    def init(self):
        # Access columns through self.data
        high  = self.data.High
        low   = self.data.Low
        close = self.data.Close

        # ----- MAs -----
        # backtesting.py's 'I' function expects a callable that returns a numpy array
        # If you want to do an EMA, you can do:
        self.maShort = self.I(
            lambda c: pd.Series(c).ewm(span=self.maShortL, adjust=False).mean(),
            close
        )

        self.maLong = self.I(
            lambda c: pd.Series(c).ewm(span=self.maLongL, adjust=False).mean(),
            close
        )

        # ----- Parabolic SAR -----
        # Similarly, define a function that takes high & low arrays:
        def psar_func(h, l):
            return parabolic_sar(pd.Series(h), pd.Series(l), self.sarAcc, self.sarMax)
        self.psar = self.I(psar_func, high, low)

        # ----- Lucid SAR -----
        def lucid_sar_func(h, l):
            return lucid_sar(pd.Series(h), pd.Series(l), self.lucidAcc, self.lucidMax)
        self.lucidSar = self.I(lucid_sar_func, high, low)

        # ----- Chop Index -----
        # Chop index in your custom_indicators expects a DataFrame with [high, low, close].
        # We can feed it as well, but the recommended approach in backtesting.py is
        # to do it purely on arrays or piecewise. Here’s a simple approach:
        def chop_func(h, l, c):
            # Build a small DF from these numpy arrays
            tmp = pd.DataFrame({"high": h, "low": l, "close": c})
            return chop_index(tmp, self.chopLen).values
        self.chop = self.I(chop_func, high, low, close)

    def next(self):
        # current (most recent) values
        psarVal     = self.psar[-1]
        lucidSarVal = self.lucidSar[-1]
        chopVal     = self.chop[-1]
        maLongVal   = self.maLong[-1]
        maShortVal  = self.maShort[-1]
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

        # Entry conditions
        longCond  = trending and above200 and goldenCross and sarBullish and lucidBullish
        shortCond = trending and below200 and deathCross  and sarBearish  and lucidBearish

        # Exit conditions
        exitLongCond  = sarBearish or lucidBearish
        exitShortCond = sarBullish or lucidBullish

        if not self.position:
            # No current position → check if we open new
            if longCond:
                sl = price * (1.0 - self.slPct / 100.0)
                tp = price * (1.0 + self.tpPct / 100.0)
                self.buy(sl=sl, tp=tp)
            elif shortCond:
                sl = price * (1.0 + self.slPct / 100.0)
                tp = price * (1.0 - self.tpPct / 100.0)
                self.sell(sl=sl, tp=tp)

        else:
            # Position exists → check exit
            if self.position.is_long and exitLongCond:
                self.position.close()
            elif self.position.is_short and exitShortCond:
                self.position.close()

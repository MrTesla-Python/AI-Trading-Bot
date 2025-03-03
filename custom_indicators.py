# custom_indicators.py

import numpy as np
import pandas as pd
import math

def pivot_high(series, pivot_len=2):
    """
    Replicates ta.pivothigh(high, pivotLen, pivotLen).
    Returns a boolean Series where pivot highs occur.
    """
    ph = []
    for i in range(len(series)):
        if i < pivot_len or i > len(series) - pivot_len - 1:
            ph.append(False)
            continue
        
        window = series[i - pivot_len : i + pivot_len + 1]
        center_val = series[i]
        
        if center_val == window.max():
            ph.append(True)
        else:
            ph.append(False)
    return pd.Series(ph, index=series.index)

def pivot_low(series, pivot_len=2):
    """
    Replicates ta.pivotlow(low, pivotLen, pivotLen).
    Returns a boolean Series where pivot lows occur.
    """
    pl = []
    for i in range(len(series)):
        if i < pivot_len or i > len(series) - pivot_len - 1:
            pl.append(False)
            continue
        
        window = series[i - pivot_len : i + pivot_len + 1]
        center_val = series[i]
        
        if center_val == window.min():
            pl.append(True)
        else:
            pl.append(False)
    return pd.Series(pl, index=series.index)

def parabolic_sar(high, low, acceleration=0.02, maximum=0.2):
    """
    Basic Parabolic SAR calculation.
    If you want a more standard, tested version, see ta.trend.PSARIndicator in 'ta' library.
    """
    psar = np.zeros(len(high))
    psar[0] = low[0]
    
    # direction: +1 for bullish, -1 for bearish
    direction = 1
    af = acceleration
    ep = high[0]  # extreme point

    for i in range(1, len(high)):
        prev_psar = psar[i-1]
        if direction == 1:
            # Bullish
            psar[i] = prev_psar + af*(ep - prev_psar)
            if psar[i] > low[i]:
                # Flip direction
                direction = -1
                psar[i] = ep
                af = acceleration
                ep = low[i]
        else:
            # Bearish
            psar[i] = prev_psar - af*(prev_psar - ep)
            if psar[i] < high[i]:
                # Flip direction
                direction = 1
                psar[i] = ep
                af = acceleration
                ep = high[i]

        # Update extreme point
        if direction == 1 and high[i] > ep:
            ep = high[i]
            af = min(af + acceleration, maximum)
        elif direction == -1 and low[i] < ep:
            ep = low[i]
            af = min(af + acceleration, maximum)

    return pd.Series(psar, index=high.index)

def lucid_sar(high, low, acceleration=0.01, maximum=0.1):
    """
    Lucid SAR is conceptually the same calculation
    as PSAR but with different default params.
    """
    return parabolic_sar(high, low, acceleration=acceleration, maximum=maximum)

def true_range(df):
    """
    True Range = max of:
     - current_high - current_low
     - abs(current_high - prev_close)
     - abs(current_low  - prev_close)
    """
    high = df['high']
    low  = df['low']
    close= df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def chop_index(df, length=14):
    """
    100 * log10( (highestHigh - lowestLow) / ATR(1) ) / log10(length)
    """
    highest_high = df['high'].rolling(length).max()
    lowest_low   = df['low'].rolling(length).min()
    
    tr = true_range(df)
    atr1 = tr  # 1-bar ATR ~ TR

    numer = (highest_high - lowest_low) / atr1
    chop  = 100 * np.log10(numer) / np.log10(length)
    
    return chop

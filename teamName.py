import numpy as np

nInst = 51

LOOKBACK = 40 # I played with this variable and 16 gives best score on eval.py      

GROSS = 10000000000     # total dollars deployed across all positions each day - larger it is, the higher our score (until position limits are hit)


def getMyPosition(prcSoFar):
    # prcSoFar has shape (51 assets, numDays). Last column is today.
    nins, nt = prcSoFar.shape

    # not enough history yet -> hold nothing (instead of crashinng)
    if nt < LOOKBACK + 1:
        return np.zeros(nins)

    window = prcSoFar[:, -LOOKBACK:]          # last LOOKBACK days for each asset
    mean = window.mean(axis=1)                
    std = window.std(axis=1)                  
    std[std == 0] = 1                         # avoid divide-by-zero on flat assets

    today = prcSoFar[:, -1]
    zscore = (today - mean) / std             # + = above average, - = below

    # Signal: bet AGAINST the stretch. High z -> short (negative), low z -> long
    signal = -zscore

    # Turn signal into dollar allocations that add up (in absolute size) to GROSS, so a stronger signal gets a bigger share of money
    total = np.abs(signal).sum()
    if total == 0:
        return np.zeros(nins)
    dollars = (signal / total) * GROSS

    positions = (dollars / today).astype(int)

    # should return 51 integers (number of shares we should own of each asset at end of day - NOT how many you should buy/sell)
    return positions

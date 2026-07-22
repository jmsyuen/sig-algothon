"""
Algothon 2026 - statistical-arbitrage strategy.

Idea
----
ALGO (instrument 0) is essentially the market index (~0.99 correlation with the
cross-sectional average return) and carries a 10x position limit with 1/5 the
commission, so it is used purely as a cheap market hedge.

Alpha comes from IDIOSYNCRATIC mean-reversion: after projecting out the top few
common factors (estimated by PCA on a trailing return window), each instrument's
residual "spread" reverts. We go long instruments whose residual has fallen and
short those whose residual has risen, size by residual volatility, blend several
factor-counts / look-backs for robustness, gate gross exposure by how well the
signal has worked recently, and hedge the net market beta with ALGO.

Only numpy is used (no extra requirements.txt needed).
"""

import numpy as np

# ---- parameters (chosen for robustness across sub-periods, not peak fit) ----
_KS        = (4, 5, 6)     # numbers of PCA factors to remove (blended)
_WS        = (18, 20, 22)  # residual reversion look-backs, in days (blended)
_FIT_WIN   = 130           # trailing window for PCA + beta estimation
_DOLLAR    = 65000         # gross sizing knob (positions clip to per-instrument $ limits)
_SMOOTH    = 0.5           # EMA on target positions to damp turnover/commissions
_GATE_WIN  = 20            # look-back for the regime gate
_GATE_FLOOR = 0.15         # minimum fraction of exposure kept when the gate is "off"

# ---- persistent state across daily calls (guarded so re-runs reset cleanly) ----
_state = {"last_t": -1, "prev": None, "sh": [], "rh": []}


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n, t = prcSoFar.shape

    # reset state if called non-sequentially (e.g. a fresh backtest in same process)
    if t <= _state["last_t"]:
        _state["prev"], _state["sh"], _state["rh"] = None, [], []
    _state["last_t"] = t

    if t < _FIT_WIN + 2:
        return np.zeros(n, dtype=int)

    logP = np.log(prcSoFar)
    R = np.diff(logP, axis=1)                 # daily log returns, shape (n, t-1)
    algo = R[0]

    Rw = R[:, -_FIT_WIN:]
    mean = Rw.mean(axis=1, keepdims=True)
    Rc = Rw - mean                            # de-meaned trailing returns

    # single SVD reused for every K in the blend
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)

    sig = np.zeros(n)
    cnt = 0
    resid5 = None
    for K in _KS:
        Load = U[:, :K]                       # factor directions
        resid = Rc - Load @ (Load.T @ Rc)     # idiosyncratic returns
        if K == 5:
            resid5 = resid
        rvol = resid.std(axis=1) + 1e-9
        for W in _WS:
            cum = resid[:, -W:].sum(axis=1)   # residual spread over W days
            sig += -(cum / (rvol * np.sqrt(W)))
            cnt += 1
    sig /= cnt
    sig[0] = 0.0                              # no idiosyncratic bet in ALGO
    if resid5 is None:                        # safety if 5 not in _KS
        Load = U[:, :5]
        resid5 = Rc - Load @ (Load.T @ Rc)

    # --- regime gate: scale exposure by how well the signal predicted recent days ---
    g = 1.0
    sh, rh = _state["sh"], _state["rh"]
    sh.append(sig.copy())
    rh.append(resid5[:, -1].copy())           # residual return that just realized
    if len(sh) > _GATE_WIN + 1:
        pnls = np.array([sh[k].dot(rh[k + 1])
                         for k in range(len(sh) - _GATE_WIN - 1, len(sh) - 1)])
        sd = pnls.std()
        if sd > 0:
            tsr = pnls.mean() / sd            # trailing daily signal Sharpe
            g = _GATE_FLOOR + (1.0 - _GATE_FLOOR) / (1.0 + np.exp(-4.0 * tsr))
    if len(sh) > _GATE_WIN + 40:              # keep memory bounded
        _state["sh"] = sh[-(_GATE_WIN + 40):]
        _state["rh"] = rh[-(_GATE_WIN + 40):]

    px = prcSoFar[:, -1]
    dollarPos = _DOLLAR * g * sig
    pos = dollarPos / px

    # hedge net market beta with ALGO (cheap, large limit)
    a = algo[-_FIT_WIN:]
    beta = (R[:, -_FIT_WIN:] @ a) / (a.dot(a) + 1e-12)
    beta[0] = 1.0
    dollarPos[0] = 0.0
    pos[0] = -np.sum(beta * dollarPos) / px[0]

    # damp turnover with an EMA on the target position
    if _state["prev"] is not None:
        pos = _SMOOTH * _state["prev"] + (1.0 - _SMOOTH) * pos
    _state["prev"] = pos.copy()

    return pos.astype(int)

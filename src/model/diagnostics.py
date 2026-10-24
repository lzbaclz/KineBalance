"""MCMC convergence diagnostics: Gelman--Rubin R-hat and effective sample size.

Pure-NumPy, no external probabilistic-programming dependency.

* ``gelman_rubin_rhat`` -- the potential scale reduction factor (Gelman & Rubin
  1992) across ``m`` independent chains; values near 1 (commonly ``< 1.01``)
  indicate between-chain agreement.
* ``effective_sample_size`` -- autocorrelation-based ESS for a single chain via
  Geyer's (1992) initial monotone positive sequence estimator (paired
  autocorrelations ``rho_{2m} + rho_{2m+1}`` summed while positive, with the
  running minimum enforcing monotonicity); the autocovariance is computed by FFT.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gelman_rubin_rhat", "effective_sample_size", "ess_per_chain"]


def gelman_rubin_rhat(chains: np.ndarray) -> np.ndarray:
    """Gelman--Rubin potential scale reduction factor across chains.

    Parameters
    ----------
    chains : numpy.ndarray
        Shape ``(m, n)`` for a scalar parameter (``m`` chains of length ``n``),
        or ``(m, n, P)`` for ``P`` parameters monitored jointly.

    Returns
    -------
    numpy.ndarray
        R-hat per parameter: scalar for ``(m, n)`` input, shape ``(P,)`` for
        ``(m, n, P)``. Parameters with zero within-chain variance return ``1.0``.
    """
    x = np.asarray(chains, dtype=np.float64)
    squeeze = False
    if x.ndim == 2:
        x = x[..., None]
        squeeze = True
    if x.ndim != 3:
        raise ValueError(f"chains must be (m, n) or (m, n, P); got ndim={x.ndim}")
    m, n, _ = x.shape
    if m < 2:
        raise ValueError("R-hat needs at least 2 chains")
    if n < 2:
        raise ValueError("R-hat needs chains of length >= 2")

    chain_means = x.mean(axis=1)                       # (m, P)
    grand_mean = chain_means.mean(axis=0)              # (P,)
    B = n / (m - 1.0) * ((chain_means - grand_mean) ** 2).sum(axis=0)  # between
    W = x.var(axis=1, ddof=1).mean(axis=0)             # within (P,)
    var_hat = (n - 1.0) / n * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
    # Degenerate within-chain variance: chains stuck at one common constant have
    # converged (R-hat = 1); chains stuck at DIFFERENT constants have not (R-hat = inf).
    rhat = np.where(W <= 0, np.where(B > 0, np.inf, 1.0), rhat)
    return float(rhat[0]) if squeeze else rhat


def effective_sample_size(x: np.ndarray) -> float:
    """Autocorrelation-based effective sample size of a single chain.

    Uses Geyer's (1992) initial monotone positive sequence estimator: the
    autocorrelations are summed in the pairs ``rho_{2m} + rho_{2m+1}`` (m = 0, 1,
    ...) until a pair sum turns non-positive, and the pair sums are forced to be
    non-increasing, which keeps the estimate stable in the noisy tail.

    Parameters
    ----------
    x : numpy.ndarray
        A single chain of shape ``(n,)``.

    Returns
    -------
    float
        Effective sample size in ``(0, n]``. A constant chain returns ``1.0``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return float(n)
    x = x - x.mean()
    if not np.any(x):
        return 1.0
    # Autocovariance via FFT (zero-padded to avoid circular wrap-around).
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n].real
    acov /= n
    if acov[0] <= 0:
        return 1.0
    rho = acov / acov[0]
    # Geyer (1992) initial monotone positive sequence: Gamma_m = rho_{2m} + rho_{2m+1}
    # (m = 0, 1, ...) is summed while it stays positive, and the running minimum
    # enforces monotonicity; tau = -1 + 2 * sum_m Gamma_m.
    gamma_sum = 0.0
    running_min = np.inf
    m = 0
    while 2 * m + 1 < n:
        g = rho[2 * m] + rho[2 * m + 1]
        if g <= 0:
            break
        running_min = min(running_min, g)
        gamma_sum += running_min
        m += 1
    tau = -1.0 + 2.0 * gamma_sum
    ess = n / tau if tau > 0 else float(n)
    return float(min(max(ess, 1.0), n))


def ess_per_chain(chains: np.ndarray) -> np.ndarray:
    """ESS for each (chain, parameter) given ``(m, n)`` or ``(m, n, P)`` draws."""
    x = np.asarray(chains, dtype=np.float64)
    if x.ndim == 2:
        return np.array([effective_sample_size(x[i]) for i in range(x.shape[0])])
    if x.ndim != 3:
        raise ValueError(f"chains must be (m, n) or (m, n, P); got ndim={x.ndim}")
    m, _, P = x.shape
    out = np.empty((m, P))
    for i in range(m):
        for j in range(P):
            out[i, j] = effective_sample_size(x[i, :, j])
    return out

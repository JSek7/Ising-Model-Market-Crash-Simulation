import numpy as np


import numpy as np


class IsingAnalysis:
    def __init__(self, window=10, max_lag=50):
        self.window = window
        self.max_lag = max_lag

    def first_returns(self, m):
        m = np.asarray(m)
        if len(m) < 2:
            return np.array([])
        return np.diff(m)

    def rolling_volatility(self, r, window=None):
        if window is None:
            window = self.window

        r = np.asarray(r)
        vol = np.full(len(r), np.nan)

        for t in range(window, len(r)):
            vol[t] = np.std(r[t - window:t])

        return vol

    def non_overlapping_volatility(self, r, window=None, ddof=1):
        if window is None:
            window = self.window

        r = np.asarray(r)
        n_blocks = len(r) // window

        if n_blocks == 0:
            return np.array([])

        trimmed = r[: n_blocks * window]
        blocks = trimmed.reshape(n_blocks, window)
        return np.std(blocks, axis=1, ddof=ddof)

    def autocorrelation_function(self, x, max_lag=None):
        if max_lag is None:
            max_lag = self.max_lag

        x = np.asarray(x)
        x = x[~np.isnan(x)]

        if len(x) == 0:
            return np.array([])

        x = x - np.mean(x)
        var = np.var(x)

        if np.isclose(var, 0):
            return np.zeros(max_lag + 1)

        N = len(x)
        acf = np.zeros(max_lag + 1)

        for tau in range(max_lag + 1):
            if tau >= N:
                acf[tau] = np.nan
            else:
                cov = np.sum(x[:N - tau] * x[tau:]) / (N - tau)
                acf[tau] = cov / var

        return acf

    def susceptibility(self, mags, T, size):
        mags = np.asarray(mags)
        avg_m = np.mean(mags)
        return (size ** 2 / T) * (np.mean(mags ** 2) - avg_m ** 2)

    def heat_capacity(self, energies, T, size):
        energies = np.asarray(energies)
        return (1.0 / (size ** 2 * T ** 2)) * (
            np.mean(energies ** 2) - np.mean(energies) ** 2
        )

    def binder_cumulant(self, mags):
        mags = np.asarray(mags)
        denom = 3 * np.mean(mags ** 2) ** 2
        if np.isclose(denom, 0):
            return np.nan
        return 1 - (np.mean(mags ** 4) / denom)

    def summarise(self, raw_result, size):
        T = raw_result["Temperature"]
        alpha = raw_result["Alpha"]
        mags = np.asarray(raw_result["Magnetisation"])
        energies = np.asarray(raw_result["Energy"])

        abs_mags = np.abs(mags)
        first_returns = self.first_returns(mags)
        abs_returns = np.abs(first_returns)
        sq_returns = first_returns ** 2

        rolling_vol = (
            self.rolling_volatility(first_returns)
            if len(first_returns) >= self.window
            else np.array([])
        )
        block_vol = (
            self.non_overlapping_volatility(first_returns)
            if len(first_returns) >= self.window
            else np.array([])
        )

        return {
            "Temperature": T,
            "Alpha": alpha,
            "Magnetisation": mags,
            "Abs_Magnetisation": abs_mags,
            "Energy": energies,
            "First_Return": first_returns,
            "Abs_Return": abs_returns,
            "Squared_Return": sq_returns,
            "Rolling_Vol": rolling_vol,
            "Non_Rol_Vol": block_vol,
            "acf_m": self.autocorrelation_function(mags),
            "acf_r": self.autocorrelation_function(first_returns),
            "acf_abs_r": self.autocorrelation_function(abs_returns),
            "acf_sq_r": self.autocorrelation_function(sq_returns),
            "acf_rolling_vol": self.autocorrelation_function(rolling_vol),
            "acf_block_vol": self.autocorrelation_function(block_vol),
            "Average_Magnetisation": np.mean(mags) if len(mags) > 0 else np.nan,
            "Abs_Average_Magnetisation": np.mean(abs_mags) if len(abs_mags) > 0 else np.nan,
            "Susceptibility": self.susceptibility(mags, T, size) if len(mags) > 0 else np.nan,
            "Heat_Capacity": self.heat_capacity(energies, T, size) if len(energies) > 0 else np.nan,
            "Binder_Cumulant": self.binder_cumulant(mags) if len(mags) > 0 else np.nan,
        }
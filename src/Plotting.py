import numpy as np
import matplotlib.pyplot as plt


class IsingPlotter:
    def __init__(self, results):
        self.results = results

    # ---------------------------
    # Helper: detect structure
    # ---------------------------
    def _is_bornholdt(self):
        first_val = next(iter(self.results.values()))
        return isinstance(first_val, dict)

    # ---------------------------
    # Magnetisation vs Temperature
    # ---------------------------
    def plot_magnetisation_vs_T(self):
        plt.figure()

        if self._is_bornholdt():
            for alpha in next(iter(self.results.values())).keys():
                temps = []
                mags = []

                for T in self.results:
                    temps.append(T)
                    mags.append(self.results[T][alpha]["Abs_Average_Magnetisation"])

                plt.plot(temps, mags, marker="o", label=f"alpha={alpha}")

        else:
            temps = list(self.results.keys())
            mags = [self.results[T]["Abs_Average_Magnetisation"] for T in temps]

            plt.plot(temps, mags, marker="o")

        plt.xlabel("Temperature")
        plt.ylabel(r"$\langle |M| \rangle$")
        plt.title("Magnetisation vs Temperature")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # Susceptibility vs T
    # ---------------------------
    def plot_susceptibility(self):
        plt.figure()

        if self._is_bornholdt():
            for alpha in next(iter(self.results.values())).keys():
                temps = []
                chi = []

                for T in self.results:
                    temps.append(T)
                    chi.append(self.results[T][alpha]["Susceptibility"])

                plt.plot(temps, chi, marker="o", label=f"alpha={alpha}")

        else:
            temps = list(self.results.keys())
            chi = [self.results[T]["Susceptibility"] for T in temps]

            plt.plot(temps, chi, marker="o")

        plt.xlabel("Temperature")
        plt.ylabel(r"$\chi$")
        plt.title("Susceptibility vs Temperature")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # Time series
    # ---------------------------
    def plot_time_series(self, T, alpha=None):
        plt.figure(figsize=(10, 4))

        if self._is_bornholdt():
            data = self.results[T][alpha]
            title = f"T={T}, alpha={alpha}"
        else:
            data = self.results[T]
            title = f"T={T}"

        plt.plot(data["Magnetisation"], label="Magnetisation")
        plt.xlabel("Time")
        plt.ylabel("M")
        plt.title(f"Magnetisation Time Series ({title})")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # Returns time series
    # ---------------------------
    def plot_returns(self, T, alpha=None):
        plt.figure(figsize=(10, 4))

        if self._is_bornholdt():
            data = self.results[T][alpha]
            title = f"T={T}, alpha={alpha}"
        else:
            data = self.results[T]
            title = f"T={T}"

        plt.plot(data["First_Return"])
        plt.title(f"Returns ({title})")
        plt.xlabel("Time")
        plt.ylabel("r_t")
        plt.grid()
        plt.show()

    # ---------------------------
    # Distribution of returns
    # ---------------------------
    def plot_return_distribution(self, T, alpha=None, bins=50):
        plt.figure()

        if self._is_bornholdt():
            data = self.results[T][alpha]
            title = f"T={T}, alpha={alpha}"
        else:
            data = self.results[T]
            title = f"T={T}"

        r = data["First_Return"]

        plt.hist(r, bins=bins, density=True)
        plt.title(f"Return Distribution ({title})")
        plt.xlabel("Returns")
        plt.ylabel("Density")
        plt.grid()
        plt.show()

    # ---------------------------
    # ACF plots
    # ---------------------------
    def plot_acf(self, T, alpha=None):
        plt.figure(figsize=(12, 4))

        if self._is_bornholdt():
            data = self.results[T][alpha]
            title = f"T={T}, alpha={alpha}"
        else:
            data = self.results[T]
            title = f"T={T}"

        plt.plot(data["acf_r"], label="ACF returns")
        plt.plot(data["acf_abs_r"], label="ACF |returns|")
        plt.plot(data["acf_sq_r"], label="ACF squared returns")

        plt.title(f"Autocorrelation ({title})")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # Volatility comparison
    # ---------------------------
    def plot_volatility(self, T, alpha=None):
        plt.figure(figsize=(10, 4))

        if self._is_bornholdt():
            data = self.results[T][alpha]
            title = f"T={T}, alpha={alpha}"
        else:
            data = self.results[T]
            title = f"T={T}"

        plt.plot(data["Rolling_Vol"], label="Rolling Vol")
        plt.title(f"Volatility ({title})")
        plt.xlabel("Time")
        plt.ylabel("Vol")
        plt.legend()
        plt.grid()
        plt.show()
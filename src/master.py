# src/master.py

from src.ising import IsingModel
from src.analysis import IsingAnalysis


class IsingExperiment:
    def __init__(self, size, temperatures, alphas=None, update_rule="bornholdt", seed=None):
        self.size = size
        self.temperatures = temperatures if isinstance(temperatures, (list, tuple)) else [temperatures]
        self.alphas = alphas if alphas is not None else [1.5]
        self.update_rule = update_rule
        self.seed = seed

        self.analysis = IsingAnalysis()

    def run_single(self, T, alpha, n_equil=1000, n_steps=1500, sample_freq=10):
        model = IsingModel(
            size=self.size,
            T=T,
            alpha=alpha if alpha is not None else 1.5,
            update_rule=self.update_rule,
            seed=self.seed,
        )

        raw = model.run(n_equil=n_equil, n_steps=n_steps, sample_freq=sample_freq, T=T, alpha=alpha)
        analysed = self.analysis.summarise(raw, size=self.size)

        return analysed

    def run_grid(self, n_equil=1000, n_steps=1500, sample_freq=10):
        results = {}

        if self.update_rule == "ising":
            for T in self.temperatures:
                print(f"Running T={T}")
                results[T] = self.run_single(T, None, n_equil=n_equil, n_steps=n_steps, sample_freq=sample_freq)

        elif self.update_rule == "bornholdt":
            for T in self.temperatures:
                print(f"Running T={T}")
                results[T] = {}
                for alpha in self.alphas:
                    print(f"alpha={alpha}")
                    results[T][alpha] = self.run_single(T, alpha, n_equil=n_equil, n_steps=n_steps, sample_freq=sample_freq)

        else:
            raise ValueError("update_rule must be 'ising' or 'bornholdt'")

        return results
import numpy as np
import pandas as pd 
import math 

import numpy as np


class IsingModel:
    def __init__(
        self,
        size,
        T,
        alpha=1.5,
        mode="random",
        update_rule="bornholdt",
        h=0.0,
        J=1.0,
        seed=None,
        rng=None,
    ):
        self.size = size
        self.T = T
        self.alpha = alpha
        self.mode = mode
        self.update_rule = update_rule
        self.h = h
        self.J = J
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        self.spins = self.create_grid()

    def create_grid(self):
        if self.mode == "random":
            spins = self.rng.choice([-1, 1], size=(self.size, self.size))
        elif self.mode == "up":
            spins = np.ones((self.size, self.size), dtype=int)
        elif self.mode == "down":
            spins = -np.ones((self.size, self.size), dtype=int)
        else:
            raise ValueError("mode must be 'random', 'up', or 'down'")
        return spins

    def reset(self):
        self.spins = self.create_grid()

    def magnetisation(self):
        return np.mean(self.spins)

    def total_energy(self):
        right = np.roll(self.spins, shift=-1, axis=1)
        down = np.roll(self.spins, shift=-1, axis=0)

        interaction = -self.J * np.sum(self.spins * (right + down))
        field_term = -self.h * np.sum(self.spins)
        return interaction + field_term

    def delta_energy(self, i, j):
        L = self.size
        s = self.spins[i, j]

        up = self.spins[(i - 1) % L, j]
        down = self.spins[(i + 1) % L, j]
        left = self.spins[i, (j - 1) % L]
        right = self.spins[i, (j + 1) % L]

        S = up + down + left + right
        return 2 * s * (self.J * S + self.h)

    def delta_energy_bornholdt(self, i, j):
        L = self.size
        s = self.spins[i, j]

        up = self.spins[(i - 1) % L, j]
        down = self.spins[(i + 1) % L, j]
        left = self.spins[i, (j - 1) % L]
        right = self.spins[i, (j + 1) % L]

        abs_m = np.abs(np.mean(self.spins))
        S = up + down + left + right

        return 2 * s * (self.J * S + self.h) - (2 * self.alpha * abs_m)

    def metropolis_sweep(self, T=None):
        if T is None:
            T = self.T

        L = self.size
        N = self.size * self.size

        for _ in range(N):
            i = self.rng.integers(0, L)
            j = self.rng.integers(0, L)

            if self.update_rule == "ising":
                dE = self.delta_energy(i, j)
            elif self.update_rule == "bornholdt":
                dE = self.delta_energy_bornholdt(i, j)
            else:
                raise ValueError("update_rule must be 'ising' or 'bornholdt'")

            if dE <= 0 or self.rng.random() < np.exp(-dE / T):
                self.spins[i, j] *= -1

    def heatbath_sweep_bornholdt(self, T=None, alpha=None):
        if T is None:
            T = self.T
        if alpha is None:
            alpha = self.alpha

        N = self.size * self.size

        for _ in range(N):
            i = self.rng.integers(0, self.size)
            j = self.rng.integers(0, self.size)

            s = self.spins[i, j]

            up = self.spins[(i - 1) % self.size, j]
            down = self.spins[(i + 1) % self.size, j]
            left = self.spins[i, (j - 1) % self.size]
            right = self.spins[i, (j + 1) % self.size]

            nn_sum = up + down + left + right
            m = np.mean(self.spins)

            local_field = self.J * nn_sum - alpha * s * abs(m) + self.h
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * local_field / T))

            self.spins[i, j] = 1 if self.rng.random() < p_plus else -1

    def sweep(self, T=None, alpha=None):
        if self.update_rule == "ising":
            self.metropolis_sweep(T=T)
        elif self.update_rule == "bornholdt":
            self.heatbath_sweep_bornholdt(T=T, alpha=alpha)
        else:
            raise ValueError("update_rule must be 'ising' or 'bornholdt'")

    def run(self, n_equil=1000, n_steps=1500, sample_freq=10, T=None, alpha=None, verbose=True):
        if T is None:
            T = self.T
        if alpha is None:
            alpha = self.alpha

        if verbose:
            print(f"[T={T}, alpha={alpha}] Running equilibration ({n_equil} sweeps)...")

        for _ in range(n_equil):
            self.sweep(T=T, alpha=alpha)

        if verbose:
            print(f"[T={T}, alpha={alpha}] Running main simulation ({n_steps} sweeps)...")

        mags = []
        energies = []

        checkpoints = {
            int(n_steps * 0.25),
            int(n_steps * 0.50),
            int(n_steps * 0.75),
            int(n_steps * 1.00),
        }

        for step in range(n_steps):
            self.sweep(T=T, alpha=alpha)

            if step % sample_freq == 0:
                mags.append(float(self.magnetisation()))
                energies.append(float(self.total_energy()))

            if verbose and (step + 1) in checkpoints:
                pct = int(100 * (step + 1) / n_steps)
                print(f"[T={T}, alpha={alpha}] {pct}% complete")

        if verbose:
            print(f"[T={T}, alpha={alpha}] Simulation complete.\n")

        return {
            "Temperature": T,
            "Alpha": alpha,
            "Magnetisation": np.array(mags),
            "Energy": np.array(energies),
        }
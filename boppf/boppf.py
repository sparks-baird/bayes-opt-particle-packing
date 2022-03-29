"""Perform Bayesian optimization on the particle packing simulation parameters."""
from os.path import join
from pathlib import Path
from boppf.utils.ax import optimize_ppf
from multiprocessing import cpu_count


class BOPPF:
    def __init__(
        self,
        dummy=False,
        particles=int(1.5e6),
        n_sobol=None,
        n_bayes=1000,
        save_dir="results",
        savename="experiment.json",
        max_parallel=cpu_count(),  # hyperthreading
    ) -> None:
        self.particles = particles
        self.n_sobol = n_sobol
        self.n_bayes = n_bayes
        self.dummy = dummy
        print(f"maximum number of parallel jobs: {max_parallel}")
        self.max_parallel = max_parallel

        if dummy:
            save_dir = join(save_dir, "dummy")

        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.savepath = join(save_dir, savename)

    def optimize(self, X_train, y_train, return_ax_client=False):

        if self.dummy:
            X_train = X_train.head(10)
            y_train = y_train.head(10)

        # %% optimization
        self.ax_client, best_parameters, mean, covariance = optimize_ppf(
            X_train,
            y_train,
            particles=self.particles,
            n_sobol=self.n_sobol,
            n_bayes=self.n_bayes,
            savepath=self.savepath,
            max_parallel=self.max_parallel,
        )

        if return_ax_client:
            return best_parameters, mean, covariance, self.ax_client
        else:
            return best_parameters, mean, covariance

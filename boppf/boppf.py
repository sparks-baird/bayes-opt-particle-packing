"""Perform Bayesian optimization on the particle packing simulation parameters."""
from os import getcwd
from os.path import join
from pathlib import Path

import torch
from boppf.utils.ax import optimize_ppf
from psutil import cpu_count
import ray
import boppf


class BOPPF:
    def __init__(
        self,
        dummy=False,
        particles=int(1.5e6),
        n_sobol=None,
        n_bayes=1000,
        save_dir="results",
        savename="experiment.json",
        max_parallel="cpu_count",  # hyperthreading or not
        include_logical_cores=False,
        debug=False,
        torch_device=torch.device("cuda"),
        use_saas=False,
        seed=10,
    ) -> None:
        self.particles = particles
        self.n_sobol = n_sobol
        self.n_bayes = n_bayes
        self.dummy = dummy

        if isinstance(max_parallel, str) and max_parallel == "cpu_count":
            max_parallel = max(1, cpu_count(logical=include_logical_cores))

        print(f"maximum number of parallel jobs: {max_parallel}")

        self.max_parallel = max_parallel
        self.torch_device = torch_device
        self.use_saas = use_saas

        if dummy:
            save_dir = join(save_dir, "dummy")

        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.savepath = join(save_dir, savename)

        if debug:
            ray.init(local_mode=True)
        else:
            ray.init()

        self.seed = seed

    def optimize(self, X_train, y_train, return_ax_client=False):
        # %% optimization
        self.ax_client, best_parameters, mean, covariance = optimize_ppf(
            X_train,
            y_train,
            particles=self.particles,
            n_sobol=self.n_sobol,
            n_bayes=self.n_bayes,
            savepath=self.savepath,
            max_parallel=self.max_parallel,
            torch_device=self.torch_device,
            use_saas=self.use_saas,
            seed=self.seed,
        )

        if return_ax_client:
            return best_parameters, mean, covariance, self.ax_client
        else:
            return best_parameters, mean, covariance


# %% Code Graveyard
# runtime_env = {"working_dir": getcwd(), "py_modules": [join("boppf", "utils")]}
# ray.init(local_mode=True, runtime_env=runtime_env)

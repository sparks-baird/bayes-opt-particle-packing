"""Perform Bayesian optimization on the particle packing simulation parameters."""
from os import getcwd, path
from pathlib import Path
from typing import Optional

import torch
from boppf.utils.ax import optimize_ppf
from psutil import cpu_count
import ray
import boppf


class BOPPF:
    def __init__(
        self,
        dummy=False,
        particles: Optional[int] = int(2.5e4),
        n_sobol: Optional[int] = None,
        n_bayes: int = 100,
        save_dir="results",
        savename="experiment.json",
        max_parallel="cpu_count",
        include_logical_cores=False,  # hyperthreading or not
        debug=False,
        torch_device=torch.device("cuda"),
        use_saas=False,
        seed=10,
        data_augmentation=False,
        remove_composition_degeneracy=True,
        remove_scaling_degeneracy=False,
        use_order_constraint=False,
        ray_verbosity=3,
        multi_fidelity: bool = False,
        lower_particles: Optional[int] = None,
        upper_particles: Optional[int] = None,
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
            save_dir = path.join(save_dir, "dummy")

        if use_saas and multi_fidelity:
            raise NotImplementedError(
                "SAASBO (use_saas) and multi_fidelity not compatible."
            )

        if use_saas:
            save_dir = path.join(save_dir, "saas")

        if multi_fidelity:
            save_dir = path.join(save_dir, "multi_fidelity")

        save_dir = path.join(
            save_dir,
            f"particles={particles}",
            f"max_parallel={max_parallel}",
            f"n_sobol={n_sobol},n_bayes={n_bayes},seed={seed}",
            f"augment={data_augmentation},drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
        )

        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir
        self.savename = savename

        ray.shutdown()
        if debug:
            ray.init(local_mode=True, num_cpus=max_parallel)
        else:
            ray.init(num_cpus=max_parallel)

        self.seed = seed

        self.data_augmentation = data_augmentation
        self.remove_composition_degeneracy = remove_composition_degeneracy
        self.remove_scaling_degeneracy = remove_scaling_degeneracy
        self.use_order_constraint = use_order_constraint
        self.ray_verbosity = ray_verbosity

        self.multi_fidelity = multi_fidelity

        if not multi_fidelity and (
            lower_particles is not None or upper_particles is not None
        ):
            raise ValueError(
                "lower_particles and upper_particles should be None if not using multi-fidelity optimization."
            )

        if multi_fidelity and lower_particles is None:
            self.lower_particles: Optional[int] = int(2.5e4)
        else:
            self.lower_particles = lower_particles

        if multi_fidelity and upper_particles is None:
            self.upper_particles: Optional[int] = int(2.5e5)
        else:
            self.upper_particles = upper_particles

    def optimize(self, X_train, y_train, return_ax_client=False):
        # %% optimization
        self.ax_client, best_parameters, mean, covariance = optimize_ppf(
            X_train,
            y_train,
            particles=self.particles,
            n_sobol=self.n_sobol,
            n_bayes=self.n_bayes,
            save_dir=self.save_dir,
            max_parallel=self.max_parallel,
            torch_device=self.torch_device,
            use_saas=self.use_saas,
            seed=self.seed,
            data_augmentation=self.data_augmentation,
            remove_composition_degeneracy=self.remove_composition_degeneracy,
            remove_scaling_degeneracy=self.remove_scaling_degeneracy,
            use_order_constraint=self.use_order_constraint,
            ray_verbosity=self.ray_verbosity,
            multi_fidelity=self.multi_fidelity,
            lower_particles=self.lower_particles,
            upper_particles=self.upper_particles,
        )

        if return_ax_client:
            return best_parameters, mean, covariance, self.ax_client
        else:
            return best_parameters, mean, covariance


# %% Code Graveyard
# runtime_env = {"working_dir": getcwd(), "py_modules": [join("boppf", "utils")]}
# ray.init(local_mode=True, runtime_env=runtime_env)

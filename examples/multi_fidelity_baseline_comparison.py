"""Reproduce paper results."""
from psutil import cpu_count
import torch
from tqdm import tqdm
from boppf.boppf import BOPPF
from boppf.utils.data import DUMMY_SEEDS, SEEDS
from time import time
import numpy as np
from os import path

dummy = True

device_str = "cpu"  # "cuda" or "cpu"
use_saas = False

if dummy:
    # https://stackoverflow.com/questions/49529372/force-gpu-memory-limit-in-pytorch
    # torch.cuda.set_per_process_memory_fraction(0.25, "cuda")
    torch.cuda.empty_cache()
    n_sobol = 2
    n_bayes = 3
    particles = 1000
    max_parallel = 2
    debug = False
    random_seeds = DUMMY_SEEDS
else:
    n_sobol = 10
    n_bayes = 100 - n_sobol
    particles = int(2.5e5)
    # save one CPU for my poor, overworked machine
    max_parallel = max(1, cpu_count(logical=False))
    debug = False
    random_seeds = SEEDS

for seed in tqdm(random_seeds, postfix="seed"):
    boppf = BOPPF(
        dummy=dummy,
        n_sobol=n_sobol,
        n_bayes=n_bayes,
        particles=particles,
        max_parallel=max_parallel,
        torch_device=torch.device(device_str),
        use_saas=use_saas,
        data_augmentation=False,
        debug=debug,
        seed=seed,
        ray_verbosity=0,
        remove_composition_degeneracy=False,
        remove_scaling_degeneracy=False,
        use_order_constraint=False,
        save_dir=path.join("results", "multi-fidelity-baseline"),
    )

    t0 = time()
    best_parameters, means, covariances, ax_client = boppf.optimize(
        np.array([]), np.array([]), return_ax_client=True
    )
    print("elapsed (s): ", time() - t0)

1 + 1

"""Reproduce paper results."""
from pprint import pprint
from warnings import warn
from psutil import cpu_count
import torch
from tqdm import tqdm
from boppf.boppf import BOPPF
from boppf.utils.data import DUMMY_SEEDS, SEEDS, COMBS_KWARGS, load_data
from time import time

data_dir = "data"
fname = "packing-fraction.csv"  # not used
X_train, y_train = load_data(fname=fname, folder=data_dir)

dummy = False

device_str = "cpu"  # "cuda" or "cpu"
use_saas = False
use_random = True

if dummy:
    # https://stackoverflow.com/questions/49529372/force-gpu-memory-limit-in-pytorch
    # torch.cuda.set_per_process_memory_fraction(0.25, "cuda")
    torch.cuda.empty_cache()
    n_sobol = 2
    n_bayes = 3
    particles = 1000
    n_train_keep = 0
    X_train = X_train.head(n_train_keep)
    y_train = y_train.head(n_train_keep)
    max_parallel = 2
    debug = False
    random_seeds = DUMMY_SEEDS
else:
    n_sobol = 10
    n_bayes = 100 - n_sobol
    particles = int(2.5e4)
    n_train_keep = 0
    X_train = X_train.head(n_train_keep)
    y_train = y_train.head(n_train_keep)
    # save one CPU for my poor, overworked machine
    max_parallel = 5  # max(1, cpu_count(logical=False))
    debug = False
    random_seeds = SEEDS

if use_random:
    n_bayes = n_sobol + n_bayes
    n_sobol = 0
    warn(
        f"Adding n_sobol to n_bayes and overwriting n_sobol to 0 since use_random=True. n_sobol: {n_sobol}, n_bayes: {n_bayes}"
    )

t_start = time()
for kwargs in tqdm(COMBS_KWARGS, postfix="combs"):
    for seed in tqdm(random_seeds, postfix="seed"):
        print(kwargs, ", seed: ", seed)

        boppf = BOPPF(
            dummy=dummy,
            n_sobol=n_sobol,
            n_bayes=n_bayes,
            particles=particles,
            max_parallel=max_parallel,
            torch_device=torch.device(device_str),
            use_saas=use_saas,
            use_random=use_random,
            data_augmentation=False,
            debug=debug,
            seed=seed,
            ray_verbosity=0,
            **kwargs,
        )

        t0 = time()
        best_parameters, means, covariances, ax_client = boppf.optimize(
            X_train, y_train, return_ax_client=True
        )
        print("elapsed (s): ", time() - t0)

print("total elapsed time (s)", time() - t_start)
1 + 1

# %% code graveyard
# kwargs = dict(
#     remove_scaling_degeneracy=False,
#     remove_composition_degeneracy=True,
#     use_order_constraint=False,
# )

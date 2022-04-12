"""Reproduce paper results."""
from psutil import cpu_count
import torch
from boppf.boppf import BOPPF
from boppf.utils.data import load_data
from time import time

data_dir = "data"
fname = "packing-fraction.csv"
X_train, y_train = load_data(fname=fname, folder=data_dir)

dummy = False

device_str = "cuda"  # "cuda" or "cpu"
use_saas = True

random_seed = 11

if dummy:
    # https://stackoverflow.com/questions/49529372/force-gpu-memory-limit-in-pytorch
    # torch.cuda.set_per_process_memory_fraction(0.25, "cuda")
    torch.cuda.empty_cache()
    n_sobol = 1
    n_bayes = 16
    particles = 1000
    n_train_keep = 100
    X_train = X_train.head(n_train_keep)
    y_train = y_train.head(n_train_keep)
else:
    n_sobol = 10
    n_bayes = 100 - n_sobol
    particles = int(1.5e6)
    n_train_keep = 0
    X_train = X_train.head(n_train_keep)
    y_train = y_train.head(n_train_keep)

# save one CPU for my poor, overworked machine
max_parallel = max(1, cpu_count(logical=True) - 2)

boppf = BOPPF(
    dummy=dummy,
    n_sobol=n_sobol,
    n_bayes=n_bayes,
    particles=particles,
    max_parallel=max_parallel,
    torch_device=torch.device(device_str),
    use_saas=use_saas,
)

t0 = time()
best_parameters, means, covariances, ax_client = boppf.optimize(
    X_train, y_train, return_ax_client=True
)
print("elapsed (s): ", time() - t0)

1 + 1

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

device_str = "cpu"  # "cpu"

if dummy:
    n_sobol = 5
    n_bayes = 100 - 5
    particles = 100
else:
    X_train = X_train.head(2000)
    y_train = y_train.head(2000)
    n_sobol = 5
    n_bayes = 100 - 5
    particles = 1000
    # n_sobol = 16
    # n_bayes = 100 - 16
    # particles = int(1.5e6)

# save one CPU for my poor machine
max_parallel = max(1, cpu_count(logical=False) - 1)

boppf = BOPPF(
    dummy=dummy,
    n_sobol=n_sobol,
    n_bayes=n_bayes,
    particles=particles,
    max_parallel=max_parallel,
    torch_device=torch.device(device_str),
)

t0 = time()
best_parameters, means, covariances, ax_client = boppf.optimize(
    X_train, y_train, return_ax_client=True
)
print("elapsed (s): ", time() - t0)

1 + 1

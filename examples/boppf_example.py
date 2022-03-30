"""Reproduce paper results."""
import torch
from boppf.boppf import BOPPF
from boppf.utils.data import load_data
from time import time

data_dir = "data"
fname = "packing-fraction.csv"
X_train, y_train = load_data(fname=fname, folder=data_dir)

dummy = False

device_str = "cuda"  # "cpu"

if dummy:
    n_sobol = 2
    n_bayes = 3
    particles = 100
else:
    n_sobol = 16
    n_bayes = 100 - 16
    particles = int(1.5e6)

boppf = BOPPF(
    dummy=dummy,
    n_sobol=n_sobol,
    n_bayes=n_bayes,
    particles=particles,
    include_logical_cores=False,
    torch_device=torch.device(device_str),
)

t0 = time()
best_parameters, means, covariances, ax_client = boppf.optimize(
    X_train, y_train, return_ax_client=True
)
print("elapsed (s): ", time() - t0)

1 + 1

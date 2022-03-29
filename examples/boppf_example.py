"""Reproduce paper results."""
from boppf.boppf import BOPPF
from boppf.utils.data import load_data

data_dir = "data"
fname = "packing-fraction.csv"
X_train, y_train = load_data(fname=fname, folder=data_dir)

dummy = True

if dummy:
    n_sobol = 2
    n_bayes = 3
    particles = 10
else:
    n_sobol = 16
    n_bayes = 1000 - 16
    particles = int(1.5e6)

boppf = BOPPF(dummy=dummy, n_sobol=n_sobol, n_bayes=n_bayes, particles=particles)
best_parameters, means, covariances, ax_client = boppf.optimize(
    X_train, y_train, return_ax_client=True
)

1 + 1

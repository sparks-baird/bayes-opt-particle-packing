"""Multi-fidelity experiments."""
import numpy as np
from psutil import cpu_count
import torch
from boppf.boppf import BOPPF
from time import time

dummy = True

device_str = "cuda"  # "cuda" or "cpu"
use_saas = False
multi_fidelity = True

random_seed = 11

if dummy:
    # https://stackoverflow.com/questions/49529372/force-gpu-memory-limit-in-pytorch
    # torch.cuda.set_per_process_memory_fraction(0.25, "cuda")
    torch.cuda.empty_cache()
    n_sobol = 1
    n_bayes = 16
    lower_particles = int(2.5e1)
    upper_particles = int(2.5e2)
    max_parallel = 2
else:
    n_sobol = 10
    n_bayes = 100 - n_sobol
    lower_particles = int(2.5e4)
    upper_particles = int(2.5e5)
    max_parallel = max(1, cpu_count(logical=False) - 1)

boppf = BOPPF(
    dummy=dummy,
    n_sobol=n_sobol,
    n_bayes=n_bayes,
    particles=None,
    max_parallel=max_parallel,
    torch_device=torch.device(device_str),
    use_saas=use_saas,
    multi_fidelity=multi_fidelity,
    lower_particles=lower_particles,
    upper_particles=upper_particles,
    remove_composition_degeneracy=False,
    remove_scaling_degeneracy=False,
    use_order_constraint=False,
    debug=False,
)

t0 = time()
best_parameters, means, covariances, ax_client = boppf.optimize(
    np.array([]), np.array([]), return_ax_client=True
)
print("elapsed (s): ", time() - t0)

1 + 1

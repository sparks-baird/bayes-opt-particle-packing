"""Depends on a file from paper_plotting.py."""
from os import path
from pathlib import Path
import pickle
from uuid import uuid4
import numpy as np
from psutil import cpu_count

import pandas as pd
from tqdm import tqdm
from boppf.utils.data import (
    COMBS_KWARGS,
    DUMMY_SEEDS,
    SEEDS,
    # mean_names,
    # std_names,
    # frac_names,
)

from os import chdir, getcwd
from boppf.utils.particle_packing import particle_packing_simulation
import ray

from boppf.utils.plotting import df_to_rounded_csv

dummy = False
use_saas = False
interact_contour = False
if dummy:
    n_sobol = 2
    n_bayes = 3
    particles = 1000
    n_train_keep = 0
    max_parallel = 2
    debug = False
    random_seeds = DUMMY_SEEDS
else:
    n_sobol = 10
    n_bayes = 100 - n_sobol
    particles = int(2.5e4)
    n_train_keep = 0
    # save one CPU for my poor, overworked machine
    max_parallel = 5  # SparksOne has 8 cores
    debug = False
    random_seeds = SEEDS

tab_dir_base = "tables"

if use_saas:
    tab_dir_base = path.join(tab_dir_base, "saas")

tab_dir_base = path.join(
    tab_dir_base,
    f"particles={particles}",
    f"max_parallel={max_parallel}",
    f"n_sobol={n_sobol},n_bayes={n_bayes}",
)

main_path = path.join(tab_dir_base, "main_df")
with open(main_path + ".pkl", "rb") as f:
    main_df = pickle.load(f)

# overwrite
# particles = int(2.5e4)
# particles = int(100)
nvalreps = 50


@ray.remote  # comment for debugging
def validate_prediction(kwargs, seed):
    remove_composition_degeneracy = kwargs["remove_composition_degeneracy"]
    remove_scaling_degeneracy = kwargs["remove_scaling_degeneracy"]
    use_order_constraint = kwargs["use_order_constraint"]

    # tab_dir = path.join(
    #     tab_dir_base,
    #     f"n_sobol={n_sobol},n_bayes={n_bayes}",
    #     f"augment=False,drop_last={rm_comp},drop_scaling={rm_scl},order={order}",
    # )

    sub_df = main_df.query(
        f"rm_comp == {remove_composition_degeneracy} and rm_scl == {remove_scaling_degeneracy} and order == {use_order_constraint} and seed == {seed}"
    )

    uid = str(uuid4())[0:8]

    mean_names = ["$\tilde{x}_1$", "$\tilde{x}_2$", "$\tilde{x}_3$"]
    std_names = ["$s_1$", "$s_2$", "$s_3$"]
    frac_names = ["$p_1$", "$p_2$", "$p_3$"]
    means = sub_df[mean_names].values.tolist()[0]
    stds = sub_df[std_names].values.tolist()[0]
    fractions = sub_df[frac_names].values.tolist()[0]

    cwd = getcwd()
    chdir("boppf/utils")

    vol_fracs = []
    for _ in range(nvalreps):
        vol_frac = particle_packing_simulation(
            uid=uid, particles=particles, means=means, stds=stds, fractions=fractions,
        )
        vol_fracs.append(vol_frac)
    avg_vol_frac = np.mean(vol_fracs)
    std_vol_frac = np.std(vol_fracs)

    chdir(cwd)

    df = pd.DataFrame(
        dict(
            remove_scaling_degeneracy=remove_scaling_degeneracy,
            remove_composition_degeneracy=remove_composition_degeneracy,
            use_order_constraint=use_order_constraint,
            **sub_df[mean_names + std_names + frac_names].to_dict(),
            seed=seed,
            vol_frac=avg_vol_frac,
            std=std_vol_frac,
        )
    )
    tab_dir = path.join(
        tab_dir_base,
        f"augment=False,drop_last={kwargs['remove_composition_degeneracy']},drop_scaling={kwargs['remove_scaling_degeneracy']},order={kwargs['use_order_constraint']}",
    )
    Path(tab_dir).mkdir(exist_ok=True, parents=True)
    df.to_csv(
        path.join(
            tab_dir, f"val_result_{seed}_particles={particles}_nvalreps={nvalreps}.csv"
        ),
        index=False,
    )

    df2 = pd.DataFrame(dict(vol_fracs=vol_fracs))
    df2.to_csv(
        path.join(
            tab_dir,
            f"val_result_{seed}_particles={particles}_nvalreps={nvalreps}_repeats.csv",
        )
    )

    return df


ray.shutdown()
ray.init(num_cpus=10)
# https://stackoverflow.com/questions/5236364/how-to-parallelize-list-comprehension-calculations-in-python
dfs = ray.get(
    [
        # validate_prediction(kwargs, seed)  # uncomment for debugging
        validate_prediction.remote(kwargs, seed)  # comment for debugging
        for kwargs in tqdm(COMBS_KWARGS)
        for seed in tqdm(random_seeds)
    ]
)

val_df = pd.concat(dfs, ignore_index=True)
val_df.to_csv(
    path.join(tab_dir_base, f"val_results_unrounded_particles={particles}.csv")
)
df_to_rounded_csv(
    val_df, save_dir=tab_dir_base, save_name=f"val_results_particles={particles}.csv"
)


1 + 1

# %% Code Graveyard
# validate_prediction(COMBS_KWARGS[0])
# vol_fracs = []
# dfs = []
# means = []
# stds = []

# for kwargs in COMBS_KWARGS:
#     for seed in random_seeds:
#         cwd = getcwd()
#         chdir("boppf/utils")
#         df = validate_prediction(kwargs, seed)
# vol_frac = ray.get(validate_prediction.remote(kwargs, seed))

# avg = np.mean(vol_fracs)
# std = np.std(vol_fracs)

# stds.append(std)
# means.append(avg)

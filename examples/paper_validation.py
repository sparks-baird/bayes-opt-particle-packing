from os import path
import pickle
from uuid import uuid4
from psutil import cpu_count

import pandas as pd
from boppf.utils.data import (
    COMBS_KWARGS,
    DUMMY_SEEDS,
    mean_names,
    std_names,
    frac_names,
)

from os import chdir, getcwd
from boppf.utils.particle_packing import particle_packing_simulation
import ray

from boppf.utils.plotting import df_to_rounded_csv

dummy = False
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
    n_bayes = 50 - n_sobol
    particles = int(2.5e4)
    n_train_keep = 0
    # save one CPU for my poor, overworked machine
    max_parallel = 8  # SparksOne has 8 cores
    debug = False
    random_seeds = [10, 11, 12, 13, 14]

tab_dir_base = path.join(
    "tables", f"particles={particles}", f"max_parallel={max_parallel}"
)

main_path = path.join(tab_dir_base, "main_df")
with open(main_path + ".pkl", "rb") as f:
    main_df = pickle.load(f)

# overwrite for dummy run
particles = 1000


@ray.remote
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

    means = sub_df[mean_names].values.tolist()
    stds = sub_df[std_names].values.tolist()
    fractions = sub_df[frac_names].values.tolist()

    cwd = getcwd()
    chdir("boppf/utils")

    vol_frac = particle_packing_simulation(
        uid=uid, particles=particles, means=means, stds=stds, fractions=fractions
    )

    chdir(cwd)

    df = pd.DataFrame(dict(kwargs=kwargs, seed=seed, vol_frac=vol_frac))

    df.to_csv(
        path.join(
            tab_dir_base,
            f"n_sobol={n_sobol},n_bayes={n_bayes}",
            f"augment=False,drop_last={kwargs['remove_composition_degeneracy']},drop_scaling={kwargs['remove_scaling_degeneracy']},order={kwargs['use_order_constraint']}",
            "val_result_{seed}.csv",
        )
    )

    return df

ray.shutdown()
ray.init(num_cpus=cpu_count(logical=False))
# https://stackoverflow.com/questions/5236364/how-to-parallelize-list-comprehension-calculations-in-python
dfs = ray.get(
    [
        validate_prediction.remote(kwargs, seed)
        for kwargs in COMBS_KWARGS
        for seed in random_seeds
    ]
)

val_df = pd.concat(dfs, axis=1, ignore_index=True)
val_df.to_csv(path.join(tab_dir_base, "val_results_unrounded.csv"))
df_to_rounded_csv(val_df, path.join(tab_dir_base, "val_results.csv"))


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


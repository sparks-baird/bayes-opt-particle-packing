from os import path

import pandas as pd
import plotly.express as px
from boppf.utils.data import COMBS_KWARGS, DUMMY_SEEDS
from boppf.utils.plotting import plot_and_save

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

fig_dir_base = path.join(
    "figures",
    f"particles={particles}",
    f"max_parallel={max_parallel}",
    f"n_sobol={n_sobol},n_bayes={n_bayes}",
)

tab_dir_base = path.join(
    "tables",
    f"particles={particles}",
    f"max_parallel={max_parallel}",
    f"n_sobol={n_sobol},n_bayes={n_bayes}",
)

# overwrite
particles = int(2.5e4)
nvalreps = 50

mapper = dict(
    remove_scaling_degeneracy="rm_scl",
    remove_composition_degeneracy="rm_comp",
    use_order_constraint="order",
)
val_df = pd.read_csv(
    path.join(tab_dir_base, f"val_results_unrounded_particles={particles}.csv",)
).rename(columns=mapper)

lbls = []
means = []
stds = []
for kwargs in COMBS_KWARGS:
    remove_composition_degeneracy = kwargs["remove_composition_degeneracy"]
    remove_scaling_degeneracy = kwargs["remove_scaling_degeneracy"]
    use_order_constraint = kwargs["use_order_constraint"]
    sub_df = val_df.query(
        f"rm_comp == {remove_composition_degeneracy} and rm_scl == {remove_scaling_degeneracy} and order == {use_order_constraint}"
    )
    tmp_lbl = []
    if remove_composition_degeneracy:
        tmp_lbl.append("comp")
    if remove_scaling_degeneracy:
        tmp_lbl.append("size")
    if use_order_constraint:
        tmp_lbl.append("order")
    if tmp_lbl == []:
        lbl = "bounds<br>-only"
    else:
        lbl = "<br>".join(tmp_lbl)

    mean = sub_df["vol_frac"].mean()
    std = sub_df["vol_frac"].std()

    lbls.append(lbl)
    means.append(mean)
    stds.append(std)

df = pd.DataFrame(dict(lbl=lbls, vol_frac=means, std=stds))
df = df.sort_values(by="vol_frac", ascending=False)

fig = px.scatter(
    df,
    x="lbl",
    y="vol_frac",
    error_y="std",
    labels=dict(vol_frac="Volume Fraction (greater is better)"),
)

fig.update_xaxes(title_text="Search Space Type")

plot_and_save(
    path.join(fig_dir_base, "val_results"),
    fig,
    mpl_kwargs=dict(size=16, width_inches=4.0, height_inches=4.0),
    show=True,
)

1 + 1

# %% Code Graveyard
# grp = val_df.groupby(by=["rm_scl", "rm_comp", "order"])
# grp["vol_frac"].mean()

# tickangle=45,
# title_font = {"size": 20},
# title_standoff = 25


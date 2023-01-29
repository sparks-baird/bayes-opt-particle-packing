"""Depends on output from paper_validation.py"""
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import offline
from boppf.utils.data import COMBS_KWARGS, DUMMY_SEEDS, SEEDS
from boppf.utils.plotting import plot_and_save
from scipy.stats import ttest_ind

nvalreps = 50

dummy = False
interact_contour = False
plot_both = False
use_saas = False
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
    max_parallel = 5
    debug = False
    random_seeds = SEEDS


def get_df(n_sobol, n_bayes, particles, max_parallel, use_saas=False):

    fig_dir_base = "figures"
    tab_dir_base = "tables"

    # if dummy:
    #     fig_dir_base = path.join(fig_dir_base, "dummy")
    #     tab_dir_base = path.join(tab_dir_base, "dummy")

    if use_saas:
        fig_dir_base = path.join(fig_dir_base, "saas")
        tab_dir_base = path.join(tab_dir_base, "saas")

    fig_dir_base = path.join(
        fig_dir_base,
        f"particles={particles}",
        f"max_parallel={max_parallel}",
        f"n_sobol={n_sobol},n_bayes={n_bayes}",
    )

    tab_dir_base = path.join(
        tab_dir_base,
        f"particles={particles}",
        f"max_parallel={max_parallel}",
        f"n_sobol={n_sobol},n_bayes={n_bayes}",
    )

    # overwrite
    # particles = int(2.5e4)

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
    sub_dfs = {}
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
        sub_dfs[lbl] = sub_df

    df = pd.DataFrame(dict(lbl=lbls, vol_frac=means, std=stds))
    df = df.sort_values(by="vol_frac", ascending=False)
    return fig_dir_base, df, sub_dfs


# return fig_dir_base to avoid local variable error (and alternative: more complex
# refactor)
fig_dir_base, df, sub_dfs = get_df(
    n_sobol, n_bayes, particles, max_parallel, use_saas=False
)
df["type"] = "GPEI"
if use_saas:
    fig_dir_base, saas_df = get_df(
        n_sobol, n_bayes, particles, max_parallel, use_saas=True
    )
    saas_df["type"] = "SAAS"


if use_saas:
    cat_df = pd.concat([df, saas_df])
else:
    cat_df = df

# fig = px.scatter(
#     cat_df,
#     x="type",
#     y="vol_frac",
#     facet_col="lbl",
#     color="type",
#     error_y="std",
#     labels=dict(
#         vol_frac="Best In-sample Validated Volume Fraction (greater is better)"
#     ),
#     width=450,
#     height=450,
# )

for key, sub_df in sub_dfs.items():
    sub_df["lbl"] = key

main_df = pd.concat(list(sub_dfs.values()), axis=0)

fig = px.box(
    main_df,
    x="lbl",
    y="vol_frac",
    points="all",
    width=450,
    height=450,
    labels=dict(
        vol_frac="Best In-sample Validated Volume Fraction (greater is better)"
    ),
)

fig.update_layout(showlegend=False)
# decrease the font size of the y-axis label
fig.update_yaxes(title_font=dict(size=10))

# fig.update_xaxes(title_text="Search Space Type")
fig.update_xaxes(title_text="")
fig.update_xaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("lbl=", "")))

Path(fig_dir_base).mkdir(exist_ok=True, parents=True)

fig_path = path.join(fig_dir_base, "val_results")
fig.write_image(fig_path + ".png", scale=3)
offline.plot(fig)

# plot_and_save(
#     path.join(fig_dir_base, "val_results"),
#     fig,
#     mpl_kwargs=dict(size=16, width_inches=8.0, height_inches=4.5),
#     show=True,
# )

# %% T-test
lbls = df["lbl"]
num_lbls = len(lbls)
ttest_results = np.zeros((num_lbls, num_lbls))
for i, i_lbl in enumerate(lbls):
    a = sub_dfs[i_lbl]["vol_frac"].values
    for j, j_lbl in enumerate(lbls):
        b = sub_dfs[j_lbl]["vol_frac"].values
        _, ttest_results[i, j] = ttest_ind(a, b, equal_var=False)

fig = px.imshow(
    ttest_results,
    x=lbls,
    y=lbls,
    color_continuous_scale="RdBu_r",
    width=450,
    height=450,
)
# heatmap with values as text labels
fig.update_traces(text=ttest_results, texttemplate="%{text:.2f}")
# name the color-axis as "t-test p-value"
fig.update_layout(coloraxis_colorbar=dict(title="t-test p-value"))

fig_path = path.join(fig_dir_base, "val_results_ttest")
fig.write_image(fig_path + ".png", scale=3)
offline.plot(fig)

# plot_and_save(
#     path.join(fig_dir_base, "val_results_ttest"),
#     fig,
#     mpl_kwargs=dict(size=16, width_inches=8.0, height_inches=4.5),
#     show=True,
# )

1 + 1

# %% Code Graveyard
# grp = val_df.groupby(by=["rm_scl", "rm_comp", "order"])
# grp["vol_frac"].mean()

# tickangle=45,
# title_font = {"size": 20},
# title_standoff = 25

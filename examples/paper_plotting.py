"""Depends on paper_results.py"""
from os import path
from ast import literal_eval
from pathlib import Path
import pickle
from warnings import warn
import plotly.express as px
import json

# TODO: interested to see CV comparison between GPEI and SAASBO
from ax.modelbridge.factory import get_GPEI
from ax import Models
from plotly import offline
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.plot.marginal_effects import plot_marginal_effects
from ax.plot.slice import interact_slice_plotly, plot_slice_plotly
from ax.plot.contour import interact_contour_plotly
from ax.service.ax_client import AxClient
import numpy as np
import pandas as pd
from boppf.utils.data import (
    COMBS_KWARGS,
    DUMMY_SEEDS,
    SEEDS,
    frac_names,
    get_parameters,
    mean_names,
    std_names,
    param_mapper,
)
from boppf.utils.plotting import (
    my_plot_feature_importance_by_feature_plotly,
    my_optimization_trace_single_method_plotly,
    my_std_optimization_trace_single_method_plotly,
    df_to_rounded_csv,
    plot_and_save,
    plot_distribution,
    to_plotly,
)
from ax.plot.feature_importances import plot_feature_importance_by_feature_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_ind

dummy = False
interact_contour = False
use_random = False
use_saas = False
if dummy:
    n_sobol = 2
    n_bayes = 3
    cutoff = 4
    particles = 1000
    n_train_keep = 0
    max_parallel = 2
    debug = False
    random_seeds = DUMMY_SEEDS
else:
    n_sobol = 10
    n_bayes = 100 - n_sobol
    cutoff = 50  # per reviewer comment (e.g., 20, 40, 60)
    particles = int(2.5e4)
    n_train_keep = 0
    # save one CPU for my poor, overworked machine
    max_parallel = 5  # SparksOne has 8 cores
    debug = False
    random_seeds = SEEDS

if use_random:
    n_bayes = n_sobol + n_bayes
    n_sobol = 0
    warn(
        f"Adding n_sobol to n_bayes and overwriting n_sobol to 0 since use_random=True. n_sobol: {n_sobol}, n_bayes: {n_bayes}"
    )

dir_base = "results"

if dummy:
    dir_base = path.join(dir_base, "dummy")

if use_saas:
    dir_base = path.join(dir_base, "saas")

if use_random:
    dir_base = path.join(dir_base, "random")

target_lbl = "vol. packing fraction"
optimization_direction = "maximize"

fig_dir_base = "figures"
tab_dir_base = "tables"

if use_saas:
    fig_dir_base = path.join(fig_dir_base, "saas")
    tab_dir_base = path.join(tab_dir_base, "saas")

if use_random:
    fig_dir_base = path.join(fig_dir_base, "random")
    tab_dir_base = path.join(tab_dir_base, "random")

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

dfs = []
kwargs_dfs = []
best_par_dfs = []
observed_sets = {}
predicted_sets = {}
for kwargs in COMBS_KWARGS:
    remove_composition_degeneracy = kwargs["remove_composition_degeneracy"]
    remove_scaling_degeneracy = kwargs["remove_scaling_degeneracy"]
    use_order_constraint = kwargs["use_order_constraint"]

    # NOTE: slightly changing the directory structure
    tab_dir = path.join(
        tab_dir_base,
        f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
    )
    Path(tab_dir).mkdir(exist_ok=True, parents=True)

    experiments = []
    ax_feature_importances = []
    best_parameters = []
    best_preds = []
    best_sems = []
    raw_means = []
    raw_sems = []
    kwargs_list = []
    observed = {}
    predicted = {}
    for seed in random_seeds:
        load_dir = path.join(
            dir_base,
            f"particles={particles}",
            f"max_parallel={max_parallel}",
            f"n_sobol={n_sobol},n_bayes={n_bayes},seed={seed}",
            f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
        )
        fpath = path.join(load_dir, "experiment.json")
        ax_client = AxClient.load_from_json_file(filepath=fpath)
        [ax_client.experiment._trials.pop(i) for i in range(cutoff, n_sobol + n_bayes)]

        experiment = ax_client.experiment
        experiments.append(experiment)

        # NOTE: slightly changing the directory structure
        # TODO: remove "seed" from the filenames since it's in the dir structure now
        # at least for a later manuscript
        fig_dir = path.join(
            fig_dir_base,
            f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
            f"seed={seed}",
        )

        # try:
        fig = my_optimization_trace_single_method_plotly(
            experiment,
            ylabel=target_lbl,
            optimization_direction=optimization_direction,
        )
        # except Exception as e:
        #     print(e)
        #     print(f"seed {seed} failed for kwargs {kwargs}")
        #     continue  # skip to next iteration
        fig.update_yaxes(range=[0.2, 0.85])
        Path(fig_dir).mkdir(exist_ok=True, parents=True)

        plot_and_save(
            path.join(fig_dir, "best_objective_plot"), fig, update_legend=True
        )

        metric = ax_client.objective_name
        ax_client.fit_model()
        model = ax_client.generation_strategy.model
        # model = get_GPEI(experiment, experiment.fetch_data())
        if not use_random:
            ax_feature_importances.append(model.feature_importances(metric))
            fig = plot_feature_importance_by_feature_plotly(model)

            fig_path = path.join(fig_dir, "feature_importances")
            plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12))

            x_range = [0.525, 0.85]
            y_range = x_range
            cv = cross_validate(model)
            observed[seed] = [cv[i].observed.data.means[0] for i in range(len(cv))]
            predicted[seed] = [cv[i].predicted.means[0] for i in range(len(cv))]
            fig = interact_cross_validation_plotly(cv)
            fig.update_xaxes(title_text="Actual Vol. Packing Fraction", range=x_range)
            fig.update_yaxes(
                title_text="Predicted Vol. Packing Fraction", range=y_range
            )
            fig_path = path.join(fig_dir, "cross_validate")
            plot_and_save(
                fig_path, fig, mpl_kwargs=dict(width_inches=4.0, size=20), show=False
            )
            # TODO: ask about feature covariance matrix on Ax GitHub
            slice_dir = path.join(fig_dir, "slice")
            Path(slice_dir).mkdir(exist_ok=True, parents=True)
            fig_path = path.join(slice_dir, "slice")
            param_names = experiment.parameters.keys()

        # can take a while to loop through, so don't plot for dummy
        if not dummy and not use_random:
            for name in param_names:
                fig = plot_slice_plotly(model, name, metric)
                fig.update_layout(title_text="")
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                plot_and_save(
                    fig_path + f"_{name}",
                    fig,
                    mpl_kwargs=dict(
                        width_inches=1.68, height_inches=1.68
                    ),  # for 5x5 grid
                    show=False,
                )
        if not use_random:
            fig_path = path.join(fig_dir, "interact_slice")
            fig = interact_slice_plotly(model)
            plot_and_save(
                fig_path,
                fig,
                mpl_kwargs=dict(width_inches=5.0, height_inches=3.0),
                show=False,
            )

            fig_path = path.join(fig_dir, "contour_2d")
            fig = to_plotly(
                ax_client.get_contour_plot(
                    param_x=frac_names[0], param_y=frac_names[1], metric_name=metric
                )
            )
            # rename x-axis to $p_1$ and rename y-axis to $p_2$
            fig.update_xaxes(title_text="$p_1$")
            fig.update_yaxes(title_text="$p_2$")
            plot_and_save(
                fig_path,
                fig,
                mpl_kwargs=dict(size=16, width_inches=6.5, height_inches=4.0),
                show=False,
            )

        if interact_contour and not use_random:
            fig_path = path.join(fig_dir, "interact_contour_2d")
            fig = interact_contour_plotly(model=model, metric_name=metric)
            plot_and_save(fig_path, fig, show=False)

            fig = to_plotly(plot_marginal_effects(model, metric))
            # fig.update_yaxes(title_text="Percent worse than experimental average")
            # https://stackoverflow.com/a/63586646/13697228
            fig.layout["yaxis"].update(
                title_text="Percent higher than experimental average"
            )
            fig.update_layout(title_text="")
            fig_path = path.join(fig_dir, "marginal_effects")
            plot_and_save(
                fig_path,
                fig,
                mpl_kwargs=dict(size=16, height_inches=3.33, width_inches=10),
                show=False,
            )

        (trial_idx, best_parameter, (best_pred, best_sem)) = ax_client.get_best_trial()
        best_pred = best_pred[metric]
        best_sem = best_sem[metric][metric]

        # raw data
        data_row = experiment.lookup_data_for_trial(trial_idx)[0].df.iloc[0]
        raw_mean = data_row["mean"]
        raw_sem = data_row["sem"]

        raw_means.append(raw_mean)
        raw_sems.append(raw_sem)
        best_parameters.append(best_parameter)
        best_preds.append(best_pred)
        best_sems.append(best_sem)
        kwargs_list.append(kwargs)

    key = str(
        dict(
            remove_composition_degeneracy=kwargs["remove_composition_degeneracy"],
            remove_scaling_degeneracy=kwargs["remove_scaling_degeneracy"],
            use_order_constraint=kwargs["use_order_constraint"],
        ),
    )
    observed_sets[key] = observed
    predicted_sets[key] = predicted

    kwargs_dfs.append(pd.DataFrame(kwargs_list))
    par_df = pd.DataFrame(best_parameters)
    if remove_scaling_degeneracy:
        last_mean = mean_names[-1]
        sub_df = par_df.filter(regex=f"(.+)_div_{last_mean}")
        for name in sub_df:
            orig_name = name.replace(f"_div_{last_mean}", "")
            par_df[orig_name] = par_df[name] * 10.0  # NOTE: hardcoded
            par_df[last_mean] = 10.0
        for name in sub_df:
            par_df.pop(name)
    if remove_composition_degeneracy:
        last_frac = frac_names[-1]
        par_df[last_frac] = 1.0 - par_df[frac_names[0:-1]].sum(axis=1)

    # par_df = par_df.rename(columns=param_mapper)

    # distribution plots
    for seed, (_, sub_df) in zip(random_seeds, par_df.iterrows()):
        fig_dir = path.join(
            fig_dir_base,
            f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
            f"seed={seed}",
        )
        fig = plot_distribution(
            sub_df[mean_names].values,
            sub_df[std_names].values,
            sub_df[frac_names].values,
        )
        plot_and_save(path.join(fig_dir, "dist"), fig, update_legend=True)

    best_par_dfs.append(par_df)
    result_df = pd.DataFrame(
        dict(
            seed=random_seeds,
            raw_mean=raw_means,
            raw_sem=raw_sems,
            best_pred=best_preds,
            best_sem=best_sems,
        )
    )
    df = pd.concat((par_df, result_df), axis=1)
    df.drop(columns=["raw_sem"], inplace=True)
    df_to_rounded_csv(df, save_dir=tab_dir, save_name="best_results.csv")
    df.to_csv(path.join(tab_dir, "best_results_unrounded.csv"))
    dfs.append(df)

    fig = my_std_optimization_trace_single_method_plotly(
        experiments, ylabel=target_lbl, optimization_direction=optimization_direction
    )
    fig.update_yaxes(range=[0.625, 0.78])
    plot_and_save(
        path.join(
            fig_dir_base,
            f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
            "best_objective_std_plot",
        ),
        fig,
        update_legend=True,
        show=False,
    )

    feat_df = pd.DataFrame(ax_feature_importances).T
    feat_df = feat_df.rename(columns=param_mapper)
    feat_df["mean"] = feat_df.mean(axis=1)
    feat_df["std"] = feat_df.std(axis=1)
    avg_ax_importances = feat_df["mean"].to_dict()
    std_ax_importances = feat_df["std"].to_dict()
    fig = my_plot_feature_importance_by_feature_plotly(
        model=None,
        feature_importances=avg_ax_importances,
        error_x=std_ax_importances,
        metric_names=[metric],
    )
    fig_path = path.join(
        fig_dir_base,
        f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
        "avg_feature_importances",
    )
    plot_and_save(fig_path, fig, mpl_kwargs=dict(size=16), show=False)

main_kwargs_df = pd.concat(kwargs_dfs, ignore_index=True)
main_df = pd.concat(dfs, ignore_index=True)
pars = {*mean_names, *std_names, *frac_names}
all_columns = {*list(main_df.columns)}
other_columns = all_columns - pars
main_df = main_df[mean_names + std_names + frac_names + list(other_columns)]
main_df = main_df.rename(columns=param_mapper)
mapper = dict(
    remove_scaling_degeneracy="rm_scl",
    remove_composition_degeneracy="rm_comp",
    use_order_constraint="order",
)
main_df = (
    pd.concat((main_kwargs_df, main_df), axis=1).rename(columns=mapper).reset_index()
)
# mapper didn't seem to actually rename as I expected it to

main_path = path.join(tab_dir_base, "main_df")
with open(main_path + ".pkl", "wb") as f:
    pickle.dump(main_df, f)

main_df.to_csv(main_path + ".csv")

for seed in random_seeds:
    sub_df = main_df[main_df.seed == seed]
    df_to_rounded_csv(sub_df, save_dir=tab_dir, save_name=f"best_of_seed={seed}.csv")


with open(path.join(tab_dir_base, "observed.json"), "w") as f:
    json.dump(observed_sets, f)

with open(path.join(tab_dir_base, "predicted.json"), "w") as f:
    json.dump(predicted_sets, f)

# %% average and std dev of best predictions for each search space
grp_df = main_df.groupby(["rm_scl", "rm_comp", "order"])
best_pred_dfs = [grp_df.get_group(x)["best_pred"] for x in grp_df.groups]
best_pred_means = grp_df.mean()["best_pred"]
best_pred_stds = grp_df.std()["best_pred"]

lbls = []


def make_lbl(kwargs):
    remove_composition_degeneracy = kwargs["remove_composition_degeneracy"]
    remove_scaling_degeneracy = kwargs["remove_scaling_degeneracy"]
    use_order_constraint = kwargs["use_order_constraint"]
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
    return lbl


for kwargs in COMBS_KWARGS:
    lbl = make_lbl(kwargs)
    lbls.append(lbl)

best_pred_dfs = {lbl: df for lbl, df in zip(lbls, best_pred_dfs)}

pred_df = pd.DataFrame(dict(lbl=lbls, vol_frac=best_pred_means, std=best_pred_stds))
pred_df = pred_df.sort_values(by="vol_frac", ascending=False)
pred_df["type"] = "GPEI"

for key, sub_df in best_pred_dfs.items():
    sub_df = sub_df.to_frame()
    sub_df["lbl"] = key
    best_pred_dfs[key] = sub_df

main_best_pred_df = pd.concat(best_pred_dfs.values(), axis=0, ignore_index=True)

fig = px.box(
    main_best_pred_df,
    x="lbl",
    y="best_pred",
    points="all",
    width=450,
    height=450,
    labels=dict(best_pred="Best In-sample Predicted Volume Fraction"),
)

# fig = px.scatter(
#     pred_df,
#     x="type",
#     y="vol_frac",
#     facet_col="lbl",
#     color="type",
#     error_y="std",
#     labels=dict(vol_frac="Best In-sample Predicted Volume Fraction"),
#     width=450,
#     height=450,
# )
# remove legend
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="")
# fig.update_xaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("lbl=", "")))

Path(fig_dir_base).mkdir(exist_ok=True, parents=True)

fig_path = path.join(fig_dir_base, "pred_results")
fig.write_image(fig_path + ".png", scale=3)
offline.plot(fig)
# plot_and_save(
#     fig_path,
#     fig,
#     mpl_kwargs=dict(size=16, width_inches=8.0, height_inches=4.5),
#     show=True,
# )

# %% cross validation MAEs
mean_maes = []
std_maes = []
dummy_maes = []
dummy_std_maes = []
mean_scaled_errors = []
std_scaled_errors = []
scaled_err_dfs = {}
lbls = []
for param_str, observation in observed_sets.items():
    kwargs = literal_eval(param_str)
    prediction = predicted_sets[param_str]
    obs_df = pd.DataFrame(observation)
    preds_df = pd.DataFrame(prediction)
    mae_df = abs(obs_df - preds_df).mean(axis=0)
    dummy_mae_df = abs(obs_df - obs_df.mean(axis=0)).mean(axis=0)
    mean_maes.append(mae_df.mean())
    std_maes.append(mae_df.std())
    dummy_maes.append(dummy_mae_df.mean())
    dummy_std_maes.append(dummy_mae_df.std())
    scaled_err_df = mae_df / dummy_mae_df
    lbl = make_lbl(kwargs)
    scaled_err_dfs[lbl] = scaled_err_df
    scaled_err_df.name = "scaled_error"
    mean_scaled_errors.append(scaled_err_df.mean())
    std_scaled_errors.append(scaled_err_df.std())
    lbls.append(lbl)

maes_df = pd.DataFrame(
    dict(lbl=lbls, scaled_error=mean_scaled_errors, std=std_scaled_errors)
)
maes_df = maes_df.sort_values(by="scaled_error", ascending=True)
maes_df["type"] = "GPEI"

for key, sub_df in scaled_err_dfs.items():
    sub_df = sub_df.to_frame()
    sub_df["lbl"] = key
    scaled_err_dfs[key] = sub_df

main_scaled_err_df = pd.concat(scaled_err_dfs.values(), axis=0, ignore_index=True)

fig = px.box(
    main_scaled_err_df,
    x="lbl",
    y="scaled_error",
    points="all",
    width=450,
    height=450,
    labels=dict(scaled_error="Cross-Validation Scaled MAE (lower is better)"),
)

# fig = px.scatter(
#     maes_df,
#     x="type",
#     y="scaled_error",
#     facet_col="lbl",
#     color="type",
#     error_y="std",
#     labels=dict(scaled_error="Cross-Validation Scaled MAE (lower is better)"),
#     width=450,
#     height=450,
# )
# remove legend
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="")
# fig.update_xaxes(showticklabels=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("lbl=", "")))

fig_path = path.join(fig_dir_base, "cv_results")
fig.write_image(fig_path + ".png", scale=3)
offline.plot(fig)
# plot_and_save(
#     fig_path,
#     fig,
#     mpl_kwargs=dict(size=16, width_inches=4.5, height_inches=4.5),
#     show=True,
# )

# %% Best Pred T-test
# lbls = ["comp", "order", "comp<br>order", "bounds<br>-only"]  # HACK: hardcoded
num_lbls = len(lbls)
pred_ttest_results = np.zeros((num_lbls, num_lbls))
for i, i_lbl in enumerate(lbls):
    a = best_pred_dfs[i_lbl]["best_pred"].values
    for j, j_lbl in enumerate(lbls):
        b = best_pred_dfs[j_lbl]["best_pred"].values
        _, pred_ttest_results[i, j] = ttest_ind(a, b, equal_var=False)

fig = px.imshow(
    pred_ttest_results,
    x=lbls,
    y=lbls,
    color_continuous_scale="RdBu_r",
    width=450,
    height=450,
)
# heatmap with values as text labels
fig.update_traces(text=pred_ttest_results, texttemplate="%{text:.2f}")
# name the color-axis as "t-test p-value"
fig.update_layout(coloraxis_colorbar=dict(title="t-test p-value"))

fig_path = path.join(fig_dir_base, "pred_results_ttest")
fig.write_image(fig_path + ".png", scale=3)
offline.plot(fig)

# plot_and_save(
#     path.join(fig_dir_base, "pred_results_ttest"),
#     fig,
#     mpl_kwargs=dict(size=16, width_inches=4.5, height_inches=4.5),
#     show=True,
# )

# %% CV T-test
# lbls = ["comp<br>order", "order", "bounds<br>-only", "comp"]  # HACK: hardcoded
for kwargs in COMBS_KWARGS:
    lbl = make_lbl(kwargs)
    lbls.append(lbl)

num_lbls = len(lbls)
cv_ttest_results = np.zeros((num_lbls, num_lbls))
for i, i_lbl in enumerate(lbls):
    a = scaled_err_dfs[i_lbl]["scaled_error"].values
    for j, j_lbl in enumerate(lbls):
        b = scaled_err_dfs[j_lbl]["scaled_error"].values
        _, cv_ttest_results[i, j] = ttest_ind(a, b, equal_var=False)

fig = px.imshow(
    cv_ttest_results,
    x=lbls,
    y=lbls,
    color_continuous_scale="RdBu_r",
    width=450,
    height=450,
)
# heatmap with values as text labels
fig.update_traces(text=cv_ttest_results, texttemplate="%{text:.2f}")
# name the color-axis as "t-test p-value"
fig.update_layout(coloraxis_colorbar=dict(title="t-test p-value"))

fig_path = path.join(fig_dir_base, "cv_results_ttest")
fig.write_image(fig_path + ".png", scale=3)
offline.plot(fig)
# plot_and_save(
#     path.join(fig_dir_base, "cv_results_ttest"),
#     fig,
#     mpl_kwargs=dict(size=16, width_inches=4.5, height_inches=4.5),
#     show=True,
# )

"""
Bash Helper Commands for copying figures to paper repo
------------------------------------------------------
Based on https://askubuntu.com/a/333641/1186612
First, open Windows Terminal with an Ubuntu shell
cd /mnt/c/Users/sterg/Documents/GitHub/
rsync -av --exclude="**/*.html" sparks-baird/bayes-opt-particle-packing/figures/particles\=25000/max_parallel\=5/ sgbaird/bayes-opt-particle-packing-papers/figures/particles\=25000/max_parallel\=5/
"""
1 + 1
# %% Code Graveyard
# exp = ax_client.experiment
# data = exp.fetch_data()
# m = get_GPEI(exp, data)
# feature_importances = m.feature_importances(target_name)
# print(DataFrame(feature_importances, index=[0]))

# best_pred, best_sem = ax_client.generation_strategy.model.predict(
#     [ObservationFeatures(best_parameter)]
# )
# best_pred = best_pred[metric][0]
# best_sem = best_sem[metric][metric][0]

# ax_client_tmp = AxClient(generation_strategy=ax_client.generation_strategy)

# _, parameters, _, _, _, _, _ = get_parameters(
#     remove_composition_degeneracy=remove_composition_degeneracy,
#     remove_scaling_degeneracy=remove_scaling_degeneracy,
# )

# ax_client_tmp.create_experiment(name="particle_packing", parameters=parameters)

# create an ax_client with only the first 50 iterations
# AxClient(
#     experiment=ax_client.experiment,
#     data=ax_client.experiment.fetch_trials_data(range(0, cutoff)),
# )

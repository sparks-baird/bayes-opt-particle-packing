from os import path
from pathlib import Path
import pickle

# TODO: interested to see CV comparison between GPEI and SAASBO
from ax.modelbridge.factory import get_GPEI
from ax import Models
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.plot.marginal_effects import plot_marginal_effects
from ax.plot.slice import interact_slice_plotly, plot_slice_plotly
from ax.plot.contour import interact_contour_plotly
from ax.service.ax_client import AxClient
import pandas as pd
from boppf.utils.data import (
    COMBS_KWARGS,
    DUMMY_SEEDS,
    SEEDS,
    frac_names,
    mean_names,
    std_names,
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

dummy = False
interact_contour = False
use_saas = True
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

dir_base = "results"

if dummy:
    dir_base = path.join(dir_base, "dummy")

if use_saas:
    dir_base = path.join(dir_base, "saas")

target_lbl = "vol. packing fraction"
optimization_direction = "maximize"

fig_dir_base = "figures"
tab_dir_base = "tables"

if use_saas:
    fig_dir_base = path.join(fig_dir_base, "saas")
    tab_dir_base = path.join(tab_dir_base, "saas")

fig_dir_base = path.join(
    fig_dir_base, f"particles={particles}", f"max_parallel={max_parallel}"
)
tab_dir_base = path.join(
    tab_dir_base, f"particles={particles}", f"max_parallel={max_parallel}"
)

dfs = []
kwargs_dfs = []
best_par_dfs = []
for kwargs in COMBS_KWARGS:
    remove_composition_degeneracy = kwargs["remove_composition_degeneracy"]
    remove_scaling_degeneracy = kwargs["remove_scaling_degeneracy"]
    use_order_constraint = kwargs["use_order_constraint"]

    # NOTE: slightly changing the directory structure
    tab_dir = path.join(
        tab_dir_base,
        f"n_sobol={n_sobol},n_bayes={n_bayes}",
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
        experiment = ax_client.experiment
        experiments.append(experiment)

        # NOTE: slightly changing the directory structure
        # TODO: remove "seed" from the filenames since it's in the dir structure now
        # at least for a later manuscript
        fig_dir = path.join(
            fig_dir_base,
            f"n_sobol={n_sobol},n_bayes={n_bayes}",
            f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
            f"seed={seed}",
        )

        fig = my_optimization_trace_single_method_plotly(
            experiment,
            ylabel=target_lbl,
            optimization_direction=optimization_direction,
        )
        fig.update_yaxes(range=[0.2, 0.85])
        Path(fig_dir).mkdir(exist_ok=True, parents=True)

        plot_and_save(
            path.join(fig_dir, "best_objective_plot"), fig, update_legend=True
        )

        metric = ax_client.objective_name
        model = ax_client.generation_strategy.model
        # model = get_GPEI(experiment, experiment.fetch_data())
        ax_feature_importances.append(model.feature_importances(metric))
        fig = plot_feature_importance_by_feature_plotly(model)

        fig_path = path.join(fig_dir, "feature_importances")
        plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12))

        x_range = [0.525, 0.8]
        y_range = x_range
        cv = cross_validate(model)
        fig = interact_cross_validation_plotly(cv)
        fig.update_xaxes(title_text="Actual Vol. Packing Fraction", range=x_range)
        fig.update_yaxes(title_text="Predicted Vol. Packing Fraction", range=y_range)
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
        if not dummy:
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
        plot_and_save(
            fig_path,
            fig,
            mpl_kwargs=dict(size=16, width_inches=6.5, height_inches=4.0),
            show=False,
        )

        if interact_contour:
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

    # distribution plots
    for seed, (_, sub_df) in zip(random_seeds, par_df.iterrows()):
        fig_dir = path.join(
            fig_dir_base,
            f"n_sobol={n_sobol},n_bayes={n_bayes}",
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
    fig.update_yaxes(range=[0.575, 0.775])
    plot_and_save(
        path.join(
            fig_dir_base,
            f"n_sobol={n_sobol},n_bayes={n_bayes}",
            f"augment=False,drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",
            "best_objective_std_plot",
        ),
        fig,
        update_legend=True,
        show=False,
    )

    feat_df = pd.DataFrame(ax_feature_importances).T
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
        f"n_sobol={n_sobol},n_bayes={n_bayes}",
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

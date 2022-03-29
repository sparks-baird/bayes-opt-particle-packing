from pathlib import Path
from boppf.utils.particle_packing import particle_packing_simulation
from ax.service.ax_client import AxClient
import numpy as np
from os.path import join, dirname
from uuid import uuid4

from boppf.utils.data import mean_names, std_names, frac_names, target_name

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models


def optimize_ppf(
    X_train,
    y_train,
    particles=int(1.5e6),
    n_sobol=None,
    n_bayes=100,
    savepath=join("results", "experiment.json"),
    max_parallel=4,
):
    n_train = X_train.shape[0]

    type = "range"
    mean_bnd = [1.0, 1000.0]
    std_bnd = [0.01, 5000.0]
    frac_bnd = [0.0, 1.0]
    mean_pars = [{"name": nm, "type": type, "bounds": mean_bnd} for nm in mean_names]
    std_pars = [{"name": nm, "type": type, "bounds": std_bnd} for nm in std_names]

    subfrac_names = frac_names[0:2]
    frac_pars = [{"name": nm, "type": type, "bounds": frac_bnd} for nm in frac_names]
    parameters = mean_pars + std_pars + frac_pars[:-1]

    if n_sobol is None:
        n_sobol = 2 * len(parameters)

    n_trials = n_sobol + n_bayes

    comp_constraint = [f"{subfrac_names[0]} + {subfrac_names[1]} <= 1.0"]

    # TODO: deal with inconsistency of Sobol sampling and compositional constraint
    gs = GenerationStrategy(
        steps=[
            # 1. Initialization step (does not require pre-existing data and is well-suited for
            # initial sampling of the search space)
            GenerationStep(
                model=Models.SOBOL,
                num_trials=n_sobol,
                min_trials_observed=n_sobol,  # How many trials need to be completed to move to next model
                max_parallelism=max_parallel,  # Max parallelism for this step
                model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
                model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
            ),
            # 2. Bayesian optimization step (requires data obtained from previous phase and learns
            # from all data available at the time of each new candidate generation call)
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                max_parallelism=max_parallel,  # Parallelism limit for this step, often lower than for Sobol
                # More on parallelism vs. required samples in BayesOpt:
                # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=False)

    ax_client.create_experiment(
        name="particle_packing",
        parameters=parameters,
        objective_name=target_name,
        minimize=False,  # Optional, defaults to False.
        parameter_constraints=comp_constraint,  # compositional constraint
    )

    for i in range(n_train):
        ax_client.attach_trial(X_train.iloc[i].to_dict())
        ax_client.complete_trial(trial_index=i, raw_data=y_train[i])

    def evaluate(parameters):
        means = np.array([parameters.get(name) for name in mean_names])
        stds = np.array([parameters.get(name) for name in std_names])
        fractions = np.array([parameters.get(name) for name in subfrac_names])
        uid = str(uuid4())[0:8]
        vol_frac = particle_packing_simulation(uid, particles, means, stds, fractions)
        return {target_name: (vol_frac, None)}

    for i in range(n_trials):
        parameterizations, is_complete = ax_client.get_next_trials(max_parallel)
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

    best_parameters, values = ax_client.get_best_parameters()

    mean, covariance = values

    # For custom filepath, pass `filepath` argument.
    ax_client.save_to_json_file(filepath=savepath)
    # restored_ax_client = AxClient.load_from_json_file()  # For custom filepath, pass
    # `filepath` argument.

    df = ax_client.get_trials_data_frame().tail(n_trials)

    # add `comp3` back in
    df[frac_names[-1]] = 1 - df[subfrac_names].sum(axis=1)

    # runtime
    trials = list(ax_client.generation_strategy.experiment.trials.values())
    trials = trials[n_train:]

    def get_runtime(trial):
        dt = (trial.time_completed - trial.time_run_started).total_seconds()
        return dt

    df["runtime"] = [get_runtime(trial) for trial in trials]

    pred = list(ax_client.get_model_predictions().values())
    pred = pred[n_train - 1 :]
    df["vol_frac_pred"] = [p[target_name][0] for p in pred]
    df["vol_frac_sigma"] = [p[target_name][1] for p in pred]

    Path("results").mkdir(exist_ok=True, parents=True)
    result_path = join(dirname(savepath), "results.csv")
    df.to_csv(result_path)

    return ax_client, best_parameters, mean, covariance


# %% code graveyard

# # parameter DataFrame
# trials_as_df = ax_client.generation_strategy.trials_as_df
# arms = trials_as_df["Arm Parameterizations"].values
# parameters = [list(arm.values())[0] for arm in arms]
# par_df = DataFrame(parameters)
# par_df[frac_names[-1]] = 1 - par_df[subfrac_names].sum(axis=1)
# df = trials_as_df.drop(columns=["Arm Parameterizations"])
# df = concat((df, par_df), axis=1)


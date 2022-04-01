import logging
from os import getcwd
from pathlib import Path
import torch
from tqdm import tqdm

from psutil import cpu_count
from boppf.utils.particle_packing import particle_packing_simulation
from ax.service.ax_client import AxClient
import numpy as np
from os.path import join, dirname
from uuid import uuid4

from ray import tune
from ray.tune import report
from ray.tune.suggest.ax import AxSearch

from boppf.utils.data import mean_names, std_names, frac_names, target_name

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models

from botorch.acquisition import qExpectedImprovement

logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.


def optimize_ppf(
    X_train,
    y_train,
    particles=int(1.5e6),
    n_sobol=None,
    n_bayes=100,
    savepath=join("results", "experiment.json"),
    max_parallel=cpu_count(logical=False),
    torch_device=torch.device("cuda"),
    use_saas=False,
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

    if use_saas:
        bayes_model = Models.FULLYBAYESIAN
    else:
        bayes_model = Models.BOTORCH_MODULAR
    # TODO: deal with inconsistency of Sobol sampling and compositional constraint
    gs = GenerationStrategy(
        steps=[
            # 1. Initialization step (does not require pre-existing data and is well-suited for
            # initial sampling of the search space)
            GenerationStep(
                model=Models.SOBOL,
                num_trials=n_sobol,
                # min_trials_observed=n_sobol,  # How many trials need to be completed to move to next model
                max_parallelism=max_parallel,  # Max parallelism for this step
                model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
                model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
            ),
            # 2. Bayesian optimization step (requires data obtained from previous phase and learns
            # from all data available at the time of each new candidate generation call)
            GenerationStep(
                model=bayes_model,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                model_kwargs={
                    "fit_out_of_design": True,
                    "torch_device": torch_device,
                    "botorch_acqf_class": qExpectedImprovement,
                    "acquisition_options": {
                        "optimizer_options": {"options": {"batch_limit": 1}}
                    },
                },
                # model_gen_kwargs={"num_restarts": 5, "raw_samples": 128},
                max_parallelism=max_parallel,  # Parallelism limit for this step, often lower than for Sobol
                # More on parallelism vs. required samples in BayesOpt:
                # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
            ),
        ]
    )

    ax_client = AxClient(
        generation_strategy=gs,
        enforce_sequential_optimization=False,
        verbose_logging=False,
    )

    ax_client.create_experiment(
        name="particle_packing",
        parameters=parameters,
        objective_name=target_name,
        minimize=False,  # Optional, defaults to False.
        parameter_constraints=comp_constraint,  # compositional constraint
    )

    for i in tqdm(range(n_train)):
        ax_client.attach_trial(X_train.iloc[i].to_dict())
        ax_client.complete_trial(trial_index=i, raw_data=y_train[i])

    def evaluate(parameters):
        means = np.array([parameters.get(name) for name in mean_names])
        stds = np.array([parameters.get(name) for name in std_names])
        fractions = np.array([parameters.get(name) for name in subfrac_names])
        uid = str(uuid4())[0:8]
        vol_frac = particle_packing_simulation(uid, particles, means, stds, fractions)
        d = {target_name: vol_frac}  # can't specify SEM perhaps?
        report(**d)

    # sequential
    # for i in range(n_trials):
    #     parameters, trial_index = ax_client.get_next_trial(max_parallel)
    #     ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

    # Set up AxSearcher in RayTune
    algo = AxSearch(ax_client=ax_client)
    # Wrap AxSearcher in a concurrently limiter, to ensure that Bayesian optimization
    # receives the data for completed trials before creating more trials
    algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=max_parallel)
    tune.run(
        evaluate,
        fail_fast=False,
        num_samples=n_trials,
        search_alg=algo,
        verbose=2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        local_dir=getcwd(),
        # To use GPU, specify: resources_per_trial={"gpu": 1}.
    )

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

    # REVIEW: even with v0.2.4, not getting all predictions
    # perhaps requires main branch as of 2022-03-30
    # https://github.com/facebook/Ax/issues/771#issuecomment-1067118102
    # pred = list(ax_client.get_model_predictions().values())
    # pred = pred[n_train - 1 :]
    # df["vol_frac_pred"] = [p[target_name][0] for p in pred]
    # df["vol_frac_sigma"] = [p[target_name][1] for p in pred]

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

# "acquisition_options": {
#     "optimizer_options": {"num_restarts": 10, "raw_samples": 256}
# },


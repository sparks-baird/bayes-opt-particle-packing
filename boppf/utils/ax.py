from itertools import permutations
import logging
from os import getcwd, path
from pathlib import Path
import pandas as pd
import ray
import torch
from tqdm import tqdm

from psutil import cpu_count
from boppf.utils.particle_packing import particle_packing_simulation
from ax.service.ax_client import AxClient
import numpy as np
from uuid import uuid4

from ray import tune
from ray.tune import report
from ray.tune.suggest.ax import AxSearch

from boppf.utils.data import SPLIT, get_parameters, frac_names, target_name

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

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
    save_dir="results",
    max_parallel=cpu_count(logical=False),
    torch_device=torch.device("cuda"),
    use_saas=False,
    seed=10,
    data_augmentation=False,
    remove_composition_degeneracy=True,
    remove_scaling_degeneracy=False,
    use_order_constraint=False,
):
    n_train = X_train.shape[0]

    (
        subfrac_names,
        parameters,
        generous_parameters,
        mean_names,
        std_names,
        orig_mean_names,
        orig_std_names,
    ) = get_parameters(
        remove_composition_degeneracy=remove_composition_degeneracy,
        remove_scaling_degeneracy=remove_scaling_degeneracy,
    )

    # TODO: make compatible with additional irreducible constraints

    if n_sobol is None:
        n_sobol = 2 * len(parameters)

    n_trials = n_sobol + n_bayes

    if remove_composition_degeneracy:
        comp_constraints = [f"{subfrac_names[0]} + {subfrac_names[1]} <= 1.0"]
    else:
        comp_constraints = []

    if use_order_constraint:
        n = len(std_names)
        order_constraints = [
            f"{std_names[i]} <= {std_names[j]}"
            for i, j in zip(range(n - 1), range(1, n))
        ]
    else:
        order_constraints = []

    parameter_constraints = comp_constraints + order_constraints

    if parameter_constraints == []:
        parameter_constraints = None

    if use_saas:
        bayes_model = Models.FULLYBAYESIAN
        kwargs = {}
    elif n_train + n_sobol + n_bayes > 2000:
        bayes_model = Models.BOTORCH_MODULAR
        kwargs = {
            "botorch_acqf_class": qExpectedImprovement,
            "acquisition_options": {
                "optimizer_options": {"options": {"batch_limit": 1}}
            },
        }
    else:
        bayes_model = Models.GPEI
        kwargs = {}
    # TODO: deal with inconsistency of Sobol sampling and compositional constraint
    sobol_step = GenerationStep(
        model=Models.SOBOL,
        num_trials=n_sobol,
        min_trials_observed=n_sobol,  # How many trials need to be completed to move to next model
        max_parallelism=max_parallel,  # Max parallelism for this step
        model_kwargs={"seed": seed},  # Any kwargs you want passed into the model
        model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
    )
    bayes_step = GenerationStep(
        model=bayes_model,
        num_trials=-1,  # No limitation on how many trials should be produced from this step
        model_kwargs={
            "fit_out_of_design": True,
            "torch_device": torch_device,
            "torch_dtype": torch.double,
            **kwargs,
        },
        # model_gen_kwargs={"num_restarts": 5, "raw_samples": 128},
        max_parallelism=max_parallel,  # Parallelism limit for this step, often lower than for Sobol
        # More on parallelism vs. required samples in BayesOpt:
        # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
    )
    if n_sobol == 0:
        steps = [bayes_step]
    elif n_bayes == 0:
        steps = [sobol_step]
    else:
        steps = [sobol_step, bayes_step]

    if n_sobol == 0 and n_bayes == 0:
        raise ValueError("n_sobol or n_bayes should be a positive integer.")

    gs = GenerationStrategy(steps=steps)

    ax_client = AxClient(
        generation_strategy=gs,
        enforce_sequential_optimization=False,
        verbose_logging=False,
    )

    if remove_scaling_degeneracy:
        final_params = generous_parameters
    else:
        final_params = parameters

    ax_client.create_experiment(
        name="particle_packing",
        parameters=final_params,
        objectives={target_name: ObjectiveProperties(minimize=False)},
        parameter_constraints=parameter_constraints,
        immutable_search_space_and_opt_config=False,
    )

    # search_space = deepcopy(ax_client.experiment.search_space)

    # if remove_scaling_degeneracy:  # and data_augmentation
    #     generous_search = ax_client.make_search_space(
    #         parameters=generous_parameters, parameter_constraints=parameter_constraints
    #     )
    #     ax_client.experiment.search_space = generous_search

    k = 0
    for i in tqdm(range(n_train)):
        x = X_train.iloc[i]
        y = y_train[i]
        combs = get_combs(data_augmentation, std_names)

        if remove_composition_degeneracy:
            last_component = frac_names[-1]
            x[last_component] = 1 - x[subfrac_names].sum()

        for comb in combs:
            x = reparameterize(
                remove_composition_degeneracy,
                remove_scaling_degeneracy,
                mean_names,
                std_names,
                orig_mean_names,
                orig_std_names,
                x,
                last_component,
                comb,
            )

            ax_client.attach_trial(x.to_dict())
            ax_client.complete_trial(trial_index=k, raw_data=y)
        k = k + 1

    def evaluate(parameters):
        # data augmentation non-functional
        # https://discuss.ray.io/t/how-to-perform-data-augmentation-with-raytune-and-axsearch/5829

        if not remove_composition_degeneracy:
            last_component = frac_names[-1]
            parameters.pop(last_component)

        if remove_scaling_degeneracy:
            names = mean_names + std_names
            for name in names:
                orig_name = name.replace("_div_mu3", "")
                parameters[orig_name] = parameters[name] * 10.0  # NOTE: hardcoded
                parameters["mu3"] = 10.0

        means = np.array([float(parameters.get(name)) for name in orig_mean_names])
        stds = np.array([float(parameters.get(name)) for name in orig_std_names])
        fractions = np.array([float(parameters.get(name)) for name in frac_names[:-1]])
        uid = str(uuid4())[0:8]  # 0:8 to shorten the long hash ID, 8 is arbitrary
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
        verbose=3,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        local_dir=getcwd(),
        # To use GPU, specify: resources_per_trial={"gpu": 1}.
    )

    # to allow breakpoints after this, might cause a worker.py file to display
    ray.shutdown()

    best_parameters, values = ax_client.get_best_parameters()

    mean, covariance = values

    ax_client.save_to_json_file(filepath=path.join(save_dir, "experiment.json"))
    # restored_ax_client = AxClient.load_from_json_file(filepath=...)

    df = ax_client.get_trials_data_frame().tail(n_trials)
    trials = list(ax_client.experiment.trials.values())
    # df = pd.DataFrame([trial.arm.parameters for trial in trials])

    # add `comp3` back in
    if remove_composition_degeneracy:
        df[frac_names[-1]] = 1 - df[subfrac_names].sum(axis=1)

    # runtime
    # trials = trials[n_train:]

    def get_runtime(trial):
        if trial.time_completed is not None:
            dt = (trial.time_completed - trial.time_run_started).total_seconds()
        else:
            dt = None

        return dt

    df["runtime"] = [get_runtime(trial) for trial in trials]

    # REVIEW: v0.2.5 should support when released, for now use stable as of 2022-04-16
    # https://github.com/facebook/Ax/issues/771#issuecomment-1067118102
    pred = list(ax_client.get_model_predictions().values())
    # pred = pred[n_train - 1 :]
    df["vol_frac_pred"] = [p[target_name][0] for p in pred]
    df["vol_frac_sigma"] = [p[target_name][1] for p in pred]

    Path("results").mkdir(exist_ok=True, parents=True)
    result_path = path.join(save_dir, "results.csv")
    df.to_csv(result_path)

    return ax_client, best_parameters, mean, covariance


def reparameterize(
    remove_composition_degeneracy,
    remove_scaling_degeneracy,
    mean_names,
    std_names,
    orig_mean_names,
    orig_std_names,
    x,
    last_component,
    comb,
):
    ordered_mean_names = [orig_mean_names[c] for c in comb]
    ordered_std_names = [orig_std_names[c] for c in comb]
    ordered_frac_names = [frac_names[c] for c in comb]
    std_mapper = {v: k for v, k in zip(orig_std_names, ordered_std_names)}
    mean_mapper = {v: k for v, k in zip(orig_mean_names, ordered_mean_names)}
    frac_mapper = {v: k for v, k in zip(frac_names, ordered_frac_names)}
    mapper = {**mean_mapper, **std_mapper, **frac_mapper}

    # https://stackoverflow.com/questions/57446160/swap-or-exchange-column-names-in-pandas-dataframe-with-multiple-columns
    x.index = [mapper.get(x, x) for x in x.to_frame().index]
    if remove_scaling_degeneracy:
        last_mean = mean_names[-1]
        scl = x[last_mean] / 10.0  # NOTE: hardcoded
        x = x / scl

        for name in mean_names + std_names:
            n1, n2 = name.split(SPLIT)
            x[name] = x[n1] / x[n2]
            # remove the original names
        [x.pop(n) for n in orig_mean_names + orig_std_names]

    if remove_composition_degeneracy:
        x.pop(last_component)
    return x


def get_combs(data_augmentation, std_names):
    n_components = len(std_names)
    vals = list(range(n_components))

    if data_augmentation:
        combs = list(permutations(vals, n_components))
    else:
        combs = [vals]
    return combs


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

# x.rename(mean_mapper)
# x.reset_index()["index"].map(mean_mapper)
# x.to_frame().rename(
#     index={**mean_mapper, **{v: k for k, v in mean_mapper.items()}},
#     inplace=True,
# )


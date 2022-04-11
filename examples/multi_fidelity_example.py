from ax.service.ax_client import AxClient
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from botorch.acquisition import qExpectedImprovement
import torch

problem = AugmentedHartmann(negate=True)


def objective(parameters):
    # x7 is the fidelity
    x = torch.tensor([parameters.get(f"x{i+1}") for i in range(7)])
    return {"f": (problem(x).item(), 0.0)}


n_sobol = 16
max_parallel = 5
device_str = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device_str)
gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=n_sobol,
            max_parallelism=max_parallel,
            model_kwargs={"seed": 999},
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.GPKG,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            model_kwargs={
                "fit_out_of_design": True,
                "torch_device": torch_device,
                "torch_dtype": torch.double,
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


ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    name="hartmann_mf_experiment",
    parameters=[
        {"name": "x1", "type": "range", "bounds": [0.0, 1.0],},
        {"name": "x2", "type": "range", "bounds": [0.0, 1.0],},
        {"name": "x3", "type": "range", "bounds": [0.0, 1.0],},
        {"name": "x4", "type": "range", "bounds": [0.0, 1.0],},
        {"name": "x5", "type": "range", "bounds": [0.0, 1.0],},
        {"name": "x6", "type": "range", "bounds": [0.0, 1.0],},
        {
            "name": "x7",
            "type": "range",
            "bounds": [0.0, 1.0],
            "is_fidelity": True,
            "target_value": 1.0,
        },
    ],
    objective_name="f",
)
# Initial sobol samples
for i in range(16):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=objective(parameters))

# KGBO
for i in range(6):
    q_p, q_t = [], []
    # Simulate batches
    for q in range(4):
        parameters, trial_index = ax_client.get_next_trial()
        q_p.append(parameters)
        q_t.append(trial_index)
    for q in range(4):
        pi = q_p[q]
        ti = q_t[q]
        ax_client.complete_trial(trial_index=ti, raw_data=objective(pi))

1 + 1


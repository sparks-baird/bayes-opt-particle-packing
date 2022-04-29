from copy import copy
from itertools import permutations, product
import pandas as pd
import numpy as np
from os.path import join

# rename the columns
mean_mapper = {f"Mean_Particle_Size_#{i}": f"mu{i}" for i in range(1, 4)}
std_mapper = {f"SD_of_Particle_#{i}": f"std{i}" for i in range(1, 4)}
frac_mapper = {f"Particle_#{i}_Mass_Fraction": f"comp{i}" for i in range(1, 4)}
mean_names = list(mean_mapper.values())
std_names = list(std_mapper.values())
frac_names = list(frac_mapper.values())
var_names = mean_names + std_names + frac_names
target_name = "vol_frac"
mapper = {**mean_mapper, **std_mapper, **frac_mapper, "Packing_Fraction": target_name}

SPLIT = "_div_"

SEEDS = list(range(13, 15))
DUMMY_SEEDS = SEEDS[0:2]

MU3 = 1.0

combs = list(product([True, False], repeat=3))

keys = [
    "remove_scaling_degeneracy",
    "remove_composition_degeneracy",
    "use_order_constraint",
]

COMBS_KWARGS = []
for comb in combs:
    kwargs = {k: v for k, v in zip(keys, comb)}
    COMBS_KWARGS.append(kwargs)


def load_data(fname="packing-fraction.csv", folder="data"):
    df = pd.read_csv(join(folder, fname))
    df = df.rename(columns=mapper)

    X_train = df[var_names]
    y_train = df[target_name]

    X_train = X_train.drop(columns=[frac_names[-1]])

    return X_train, y_train


def get_parameters(remove_composition_degeneracy=True, remove_scaling_degeneracy=False):
    # TODO: add kwargs for the other two irreducible search spaces
    type = "range"
    mean_bnd = [1.0, 5.0]
    std_bnd = [0.1, 1.0]
    frac_bnd = [0.0, 1.0]
    orig_mean_names = copy(mean_names)
    orig_std_names = copy(std_names)
    if remove_scaling_degeneracy:
        mean_low, mean_upp = mean_bnd
        generous_mean_bnd = [mean_low / mean_upp, mean_upp / mean_low]
        mean_bnd = [lim / MU3 for lim in mean_bnd]
        mean_names_out = [f"mu1{SPLIT}mu3", f"mu2{SPLIT}mu3"]

        # NOTE: sigma should stay same to preserve log-normal distribution shape
        generous_std_bnd = std_bnd
        std_names_out = std_names
    else:
        mean_names_out = mean_names
        std_names_out = std_names
        generous_mean_bnd = mean_bnd
        generous_std_bnd = std_bnd
    subfrac_names, parameters = _get_parameters(
        type,
        mean_bnd,
        std_bnd,
        frac_bnd,
        mean_names_out,
        std_names_out,
        remove_composition_degeneracy=remove_composition_degeneracy,
    )

    _, generous_parameters = _get_parameters(
        type,
        generous_mean_bnd,
        generous_std_bnd,
        frac_bnd,
        mean_names_out,
        std_names_out,
        remove_composition_degeneracy=remove_composition_degeneracy,
    )
    return (
        subfrac_names,
        parameters,
        generous_parameters,
        mean_names_out,
        std_names_out,
        orig_mean_names,
        orig_std_names,
    )


def _get_parameters(
    type,
    mean_bnd,
    std_bnd,
    frac_bnd,
    mean_names_out,
    std_names_out,
    remove_composition_degeneracy=True,
):
    mean_pars = [
        {"name": nm, "type": type, "bounds": mean_bnd, "value_type": "float"}
        for nm in mean_names_out
    ]
    std_pars = [
        {"name": nm, "type": type, "bounds": std_bnd, "value_type": "float"}
        for nm in std_names_out
    ]

    subfrac_names = frac_names[:-1]
    frac_pars = [
        {"name": nm, "type": type, "bounds": frac_bnd, "value_type": "float"}
        for nm in frac_names
    ]

    if remove_composition_degeneracy:
        parameters = mean_pars + std_pars + frac_pars[:-1]
    else:
        parameters = mean_pars + std_pars + frac_pars
    return subfrac_names, parameters


def gen_symmetric_trials(data, component_slot_names, composition_slot_names):
    nslots = len(data) // 2

    vals = list(data.values())
    pairs = list(zip(vals[:nslots], vals[nslots:]))
    combs = list(permutations(pairs, 5))

    comb_data = []
    for comb in combs:
        subcomponents, subcompositions = zip(*comb)
        component_dict = {
            component_slot_name: component
            for (component_slot_name, component) in zip(
                component_slot_names, subcomponents
            )
        }
        composition_dict = {
            composition_slot_name: component
            for (composition_slot_name, component) in zip(
                composition_slot_names, subcompositions
            )
        }
        comb_data.append({**component_dict, **composition_dict})

    return comb_data


# %% Code Graveyard
# std_low, std_upp = std_bnd
# generous_std_bnd = [std_low / mean_upp, std_upp / mean_low]
# std_bnd = [lim / mu3 for lim in std_bnd]
# std_names_out = [f"std1{SPLIT}mu3", f"std2{SPLIT}mu3", f"std3{SPLIT}mu3"]


from random import randint
from uuid import uuid4
from os import path
import cloudpickle as pickle
from math import ceil, pi
import os
import shutil
from subprocess import run
import numpy as np
import pandas as pd
from pathlib import Path
import stat
from scipy.stats import lognorm

slurm_savepath = path.join("results", "log_boppf", "packing-generation-results.csv")


def get_diameters(means, stds, comps, num_particles=100, seed=None):
    samples = []
    for mu, sigma, comp in zip(means, stds, comps):
        samples.append(
            lognorm.rvs(
                sigma, scale=mu, size=ceil(comp * num_particles), random_state=seed
            )
        )

    samples = np.concatenate(samples)[:num_particles]

    if len(samples) != num_particles:
        raise ValueError(
            f"Number of samples ({len(samples)}) does not match requested number of particles {num_particles}. Ensure `sum(comps)==1` (sum({comps})=={sum(comps)})"
        )

    return samples


def write_diameters(X, data_dir=".", uid=str(uuid4())):
    save_dir = path.join(data_dir, uid)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    fpath = path.join(save_dir, "diameters.txt")
    np.savetxt(fpath, X)


def get_box_length(X, safety_factor=2.0):
    particles_vol = sum(4 / 3 * pi * (X / 2) ** 3)
    return (particles_vol * safety_factor) ** (1 / 3)


def run_simulation(flag, util_dir=".", data_dir=".", uid=str(uuid4())):
    exe_name = "PackingGeneration.exe"
    simpath = path.join(util_dir, exe_name)
    new_dir = path.join(data_dir, uid)
    newpath = path.join(new_dir, exe_name)
    shutil.copyfile(simpath, newpath)
    f = Path(newpath)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    cwd = os.getcwd()
    os.chdir(new_dir)
    result = run(
        [f"./{exe_name}", flag], capture_output=True, text=True
    )  # stdout=PIPE, stderr=STDOUT
    # print(result.stdout)
    # print(result.stderr)
    os.chdir(cwd)


def particle_packing_simulation(
    means,
    stds,
    comps,
    num_particles=100,
    contraction_rate=1e-3,
    seed=None,
    data_dir=".",
    util_dir=".",
    uid=None,
    safety_factor=2.0,
):
    if seed is None:
        seed = randint(0, 100000)
    if uid is None:
        uid = str(uuid4())
        print(uid)

    save_dir = path.join(data_dir, uid)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    generation_conf_fpath = path.join(save_dir, "generation.conf")
    packing_nfo_fpath = path.join(save_dir, "packing.nfo")
    packing_xyzd_fpath = path.join(save_dir, "packing.xyzd")

    # X = np.repeat(1.0, num_particles)
    X = get_diameters(means, stds, comps, num_particles=num_particles, seed=seed)

    write_diameters(X, data_dir=data_dir, uid=uid)
    box_length = get_box_length(X, safety_factor=safety_factor)

    # remove existing data files, if any
    names = [
        "packing.xyzd",
        "packing_init.xyzd",
        "packing_prev.xyzd",
        "contraction_energies.txt",
        "packing.nfo",
    ]
    [
        os.remove(path.join(data_dir, uid, name)) if path.exists(name) else None
        for name in names
    ]

    with open(generation_conf_fpath, "w") as f:
        lines = [
            f"Particles count: {num_particles}",
            f"Packing size: {box_length} {box_length} {box_length}",
            "Generation start: 1",
            f"Seed: {seed}",
            "Steps to write: 1000",
            "Boundaries mode: 1",
            f"Contraction rate: {contraction_rate}",
        ]
        f.writelines(lines)

    run_simulation("-fba", util_dir=util_dir, data_dir=data_dir, uid=uid)

    with open(generation_conf_fpath, "w") as f:
        lines = [
            f"Particles count: {num_particles}",
            f"Packing size: {box_length} {box_length} {box_length}",
            f"Generation start: 0",
            f"Seed: {seed}",
            "Steps to write: 1000",
            "Boundaries mode: 1",
            f"Contraction rate: {contraction_rate}",
        ]
        f.writelines(lines)

    try:
        os.remove(packing_nfo_fpath)
    except Exception as e:
        print(e)
    try:
        run_simulation("-ls", util_dir=util_dir, data_dir=data_dir, uid=uid)
    except Exception as e:
        print(e)

    try:
        os.remove(packing_nfo_fpath)
    except Exception as e:
        print(e)
    try:
        run_simulation("-lsgd", util_dir=util_dir, data_dir=data_dir, uid=uid)
    except Exception as e:
        print(e)

    """https://github.com/VasiliBaranov/packing-generation/issues/30#issue-1103925864"""

    try:
        packing = np.fromfile(path.join(data_dir, uid, "packing.xyzd")).reshape(-1, 4)
        with open(path.join(data_dir, uid, "packing.nfo"), "r+") as nfo:
            lines = nfo.readlines()
            Theoretical_Porosity = float(lines[2].split()[2])
            Final_Porosity = float(lines[3].split()[2])
            # print(Theoretical_Porosity, Final_Porosity)

            scaling_factor = ((1 - Final_Porosity) / (1 - Theoretical_Porosity)) ** (
                1 / 3
            )

            real_diameters = packing[:, 3] * scaling_factor
            actual_density = (
                sum((4 / 3) * pi * (np.array(real_diameters) / 2) ** 3)
                / box_length ** 3
            )
            packing[:, 3] = real_diameters
            packing.tofile(
                packing_xyzd_fpath
            )  # updating the packing: this line will modifies diameters in the packing.xyzd

            # update packing.nfo and set TheoreticalPorosity to FinalPorosity to avoid scaling the packing once again the next time running this script.
            lines[3] = lines[3].replace(str(Final_Porosity), str(Theoretical_Porosity))
            nfo.seek(0)
            nfo.writelines(lines)

            return actual_density
    except Exception:
        return np.nan


# def evaluate(parameters):
#     packing_fraction = particle_packing_simulation(**parameters)
#     return {**parameters, "packing_fraction": packing_fraction}


def evaluate(parameters):
    mu3 = 3.0
    # print("current working directory: ", os.getcwd())
    means = [parameters[name] * mu3 for name in ["mu1_div_mu3", "mu2_div_mu3"]]
    means.append(mu3)
    stds = [parameters[name] for name in ["std1", "std2", "std3"]]
    comps = [parameters[name] for name in ["comp1", "comp2"]]
    comps.append(1 - sum(comps))
    num_particles = parameters["num_particles"]
    try:
        result = particle_packing_simulation(
            means, stds, comps, num_particles=num_particles
        )
    except Exception as e:
        print(e)
        result = np.nan
    return {**parameters, "packing_fraction": result}


def evaluate_batch(parameter_sets):
    return [evaluate(parameters) for parameters in parameter_sets]


def collect_results():
    with open("jobs.pkl", "rb") as f:
        jobs = pickle.load(f)

    results = [job.result() for job in jobs]
    pd.DataFrame(results).to_csv(slurm_savepath, index=False)

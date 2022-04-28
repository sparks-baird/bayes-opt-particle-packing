"""Call the appropriate MATLAB scripts and executable."""
from os import getcwd
import os
from pathlib import Path
from subprocess import DEVNULL, STDOUT, Popen, PIPE, run
from os.path import join, abspath
from sys import executable
from typing import List
import numpy as np
from sklearn.preprocessing import normalize
from numpy.random import lognormal
from scipy.stats import lognorm

# conda activate boppf
# cd C:\Program Files\MATLAB\R2021a\extern\engines\python
# python setup.py install
from matlab import engine, double

from boppf.utils.proprietary import write_proprietary_input_file


def particle_packing_simulation(
    uid: str = "tmp",
    particles: int = int(1.5e6),
    means: List[float] = [120.0, 120.0, 120.0],
    stds: List[float] = [10.0, 10.0, 10.0],
    fractions: List[float] = [0.33, 0.33],
):
    """Perform particle packing simulation.py

    Parameters
    ----------
    uid : str, optional
        The prefix to the input filename, by default 0
    particles : int, optional
        The number of particles to drop in the simulation, by default 1500000
    means : List[float], optional
        The log-normal mean radius of the 3 particles, by default double([120, 120, 120])
    stds : List[float], optional
        The log-normal standard deviations of the 3 particles, by default double([10, 10, 10])
    fractions : List[float], optional
        The mass fractions of the first two particles, by default double([0.33, 0.33])

    Returns
    -------
    vol_frac : float
        Volumetric packing fraction of the lump of dropped particles.
    """
    cwd, eng, util_dir, data_dir = write_input_file(
        uid, particles, means, stds, fractions
    )

    run_simulation(uid, util_dir, data_dir)

    vol_frac = read_vol_frac(uid, cwd, eng, data_dir)

    return vol_frac


def write_input_file(uid, particles, means, stds, fractions):
    fractions[fractions < 1e-6] = 0.0
    fractions = normalize(fractions.reshape(1, -1), norm="l1")

    # sample points and their probabilities from log-normal
    for mean, std in zip(means, stds):
        # lognormal(mean=mean, sigma=std, size=100)
        s = std
        scale = np.exp(mean)
        samples = lognorm.rvs(s, scale=scale)

        alphas = np.linspace(0, 1, 102)
        # remove first and last (avoid 0 or near-zero for log-normal)
        del alphas[0]
        del alphas[-1]
        samples = lognorm.ppf(alphas, s, scale=scale)

        probs = lognorm.pdf(samples, s, scale=scale)

        # REVIEW: not sure if it's better to use lognorm.ppf+np.linspace or lognorm.rvs
        # or just use more samples so it matters less
        # dist=lognorm([std],loc=mean)

    # TODO: don't include a mode if the fraction is close to 0, same for submode

    # working directory and path finagling
    cwd = os.getcwd()
    os.chdir(join("..", ".."))
    data_dir = join("boppf", "data")
    Path(data_dir).mkdir(exist_ok=True, parents=True)

    write_proprietary_input_file(
        uid, particles, s_radii, c_radii, m_fracs, data_dir=data_dir
    )

    return cwd, util_dir, data_dir


def run_simulation(uid, util_dir, data_dir):
    fpath = join(util_dir, "particle_packing_sim.exe")
    input = join(data_dir, f"{uid}.inp")
    run([fpath], input=input, text=True, stdout=PIPE, stderr=STDOUT)


def read_vol_frac(uid, cwd, eng, data_dir):
    eng = engine.start_matlab()
    eng.addpath(join("boppf", "utils"))
    vol_frac = eng.read_vol_frac(uid, data_dir)
    print("vol_frac: ", vol_frac)
    eng.quit()

    os.chdir(cwd)
    return vol_frac


# %% Code Graveyard
# with Popen([executable, fpath], stdin=PIPE, stdout=PIPE, stderr=STDOUT) as p:
# out, err = p.communicate(input=input)
# print(out, err)
# print(getcwd())

# def write_input_file(uid, particles, means, stds, fractions):
#     cwd = os.getcwd()
#     os.chdir(join("..", ".."))
#     eng = engine.start_matlab()
#     eng.addpath(join("boppf", "utils"))

#     means = double(list(means))
#     stds = double(list(stds))
#     fractions = np.append(fractions, 1 - np.sum(fractions))
#     fractions[fractions < 1e-6] = 0.0
#     fractions = normalize(fractions.reshape(1, -1), norm="l1")
#     fractions = double([fractions.tolist()])

#     util_dir = join("boppf", "utils")
#     data_dir = join("boppf", "data")

#     Path(data_dir).mkdir(exist_ok=True, parents=True)
#     eng.write_input_file(uid, means, stds, fractions, particles, data_dir, nargout=0)
#     eng.quit()
#     return cwd, eng, util_dir, data_dir

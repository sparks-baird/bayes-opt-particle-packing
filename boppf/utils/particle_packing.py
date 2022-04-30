"""Call the appropriate MATLAB scripts and executable."""
from os import getcwd, path
import os
from pathlib import Path
from subprocess import DEVNULL, STDOUT, Popen, PIPE, run
from os.path import join, abspath
from sys import executable
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from numpy.random import lognormal
from scipy.stats import lognorm
from plotly import offline
import plotly.express as px
from random import choices

# conda activate boppf
# cd C:\Program Files\MATLAB\R2021a\extern\engines\python
# python setup.py install
from matlab import engine, double

from boppf.utils.proprietary import SECTION_KEY, write_proprietary_input_file, LINE_KEY


def particle_packing_simulation(
    uid: str = "tmp",
    particles: int = int(1.5e6),
    means: List[float] = [120.0, 120.0, 120.0],
    stds: List[float] = [1.0, 1.0, 1.0],
    fractions: List[float] = [0.33, 0.33],
    max_submodes_per_mode: int = 33,
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
    max_n_submodes_per_mode : int
        Maximum number of submodes for a given node. May be less than this because of
        filtering based on max allowed size ratios. Might be restricted to fewer than
        100 total submodes across all modes.

    Returns
    -------
    vol_frac : float
        Volumetric packing fraction of the lump of dropped particles.
    """
    cwd, util_dir, data_dir = write_input_file(
        uid, particles, means, stds, fractions, size=max_submodes_per_mode
    )

    run_simulation(uid, util_dir, data_dir)

    vol_frac = read_vol_frac(uid, cwd, data_dir)

    return vol_frac


def normalize_row_l1(x):
    normed_row = x / sum(x)
    return normed_row


def write_input_file(
    uid,
    particles,
    means,
    stds,
    fractions,
    tol=1e-6,
    random_state=None,
    size=33,  # max ~200 across all submodes (docs say 100)
    alpha=0.99,
):
    fractions = np.array(fractions)
    fractions[fractions < tol] = 0.0
    fractions = normalize_row_l1(fractions)

    # sample points and their probabilities from log-normal
    s_radii = []
    c_radii = []
    m_fracs = []
    for mu, sigma, frac in zip(means, stds, fractions):
        # lognormal(mean=mean, sigma=std, size=100)
        if frac >= tol:

            s = sigma
            # scale: such that np.mean(samples) = mu (approx.)
            # scale = mu / np.sqrt(np.exp(1))

            scale = mu

            # s_mode_radii = lognorm.rvs(
            #     s, scale=scale, size=size, random_state=random_state
            # )

            # alphas = np.linspace(0, 1, size + 2)
            # # remove first and last (avoid 0 or near-zero for log-normal)
            # alphas = alphas[1:-1]
            # s_mode_radii = lognorm.ppf(alphas, s, scale=scale)

            running_size = size
            n_radii = 0
            s_mode_radii = None
            while n_radii <= size:
                s_mode_previous = s_mode_radii
                alphas = [1 / (running_size), (running_size - 1) / running_size]
                s_mode_low, s_mode_upp = lognorm.ppf(alphas, s, scale=scale)
                s_mode_radii = np.linspace(s_mode_low, s_mode_upp, running_size)

                # cutoff = lognorm.ppf(alpha, s, scale=scale)
                # s_mode_radii = s_mode_radii[s_mode_radii < cutoff]

                # by choosing the median rather than the mean after applying the scaling
                # then the max ratio between any two particles in a system of 3
                # distributions isn't a hard constraint

                # median = lognorm.median(s, scale=scale)

                # make it relative to mu so I know the exact max ratios
                max_ratio = 16
                upp = np.sqrt(max_ratio)  # e.g. 4
                low = 1 / upp  # e.g. 0.25
                s_mode_radii = s_mode_radii[
                    np.all(
                        [s_mode_radii > low * scale, s_mode_radii < upp * scale],
                        axis=0,
                    )
                ]
                n_radii = len(s_mode_radii)
                running_size += 1
            s_mode_radii = s_mode_previous

            probs = lognorm.pdf(s_mode_radii, s, scale=scale)

            normed_probs = normalize_row_l1(probs)
            m_mode_fracs = normed_probs * frac

            # # plot radii and sampled histogram from the new (discrete) distribution
            # check_samples = choices(s_mode_radii, weights=normed_probs, k=100000)
            # df = pd.DataFrame(
            #     dict(s_mode_radii=s_mode_radii, normed_probs=normed_probs)
            # )
            # fig = px.scatter(df, x="s_mode_radii", y="normed_probs")

            # fig.add_histogram(x=check_samples, histnorm="probability")
            # fig.add_annotation(
            #     xref="paper",
            #     yref="paper",
            #     x=0.9,
            #     y=0.9,
            #     text=f"scale={scale:.3f}, s={s:.3f}",
            #     showarrow=False,
            # )
            # fig.show()
            # print(
            #     f"scale={scale}, s={s}, mean={np.mean(check_samples)}, median={np.median(check_samples)}"
            # )

            # remove submodes close to zero
            # (might not have any effect with low enough max_ratio relative to tol)
            keep_ids = m_mode_fracs > tol

            probs = probs[keep_ids]
            normed_probs = normed_probs[keep_ids]
            m_mode_fracs = m_mode_fracs[keep_ids]
            s_mode_radii = s_mode_radii[keep_ids]

            c_mode_radii = 20 * s_mode_radii

            s_radii.append(s_mode_radii)
            c_radii.append(c_mode_radii)
            m_fracs.append(m_mode_fracs)

    # working directory and path finagling
    cwd = os.getcwd()
    os.chdir(join("..", ".."))
    util_dir = join("boppf", "utils")
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


def read_vol_frac(uid, cwd, data_dir):
    fpath = path.join(data_dir, f"{uid}.stat")
    with open(fpath, "r") as f:
        lines = f.readlines()
        passed_section = False
        for line in lines:
            if SECTION_KEY in line:
                passed_section = True
            if passed_section and LINE_KEY in line:
                vol_frac_line = line.replace("\n", "")
        vol_frac = vol_frac_line.split(", ")[1]
        vol_frac = float(vol_frac)
    print("vol_frac: ", vol_frac)

    os.chdir(cwd)
    return vol_frac


# def read_vol_frac(uid, cwd, eng, data_dir):
#     eng = engine.start_matlab()
#     eng.addpath(join("boppf", "utils"))
#     vol_frac = eng.read_vol_frac(uid, data_dir)
#     print("vol_frac: ", vol_frac)
#     eng.quit()

#     os.chdir(cwd)
#     return vol_frac


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

# alphas = np.linspace(0, 1, 1000 + 2)
# # remove first and last (avoid 0 or near-zero for log-normal)
# alphas = alphas[1:-1]

# s = np.log(std)
# scale = np.log(mean)
# samples = lognorm.rvs(s, scale=scale, size=100)

# probs = lognorm.pdf(samples, std, loc=mean)

# s_mode_radii = normalize_row_l1(s_mode_radii)

# maybe use something like MDL histogram density estimation

# if len(x) > 1:
#     normed_row = normalize(x.reshape(1, -1), norm="l1")[0]
# else:
#     normed_row = [1.0]
# vol_frac_line = [l if LINE_KEY in l else "" for l in lines]
# vol_frac_line = "".join(vol_frac_line)  # blank strings go away


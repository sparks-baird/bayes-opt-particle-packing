from os import chdir, getcwd
from uuid import uuid4

import numpy as np
from boppf.utils.particle_packing import particle_packing_simulation

uid = str(uuid4())[0:8]
particles = int(1e5)
# particles = 10


def degree_of_freedom_test():

    cwd = getcwd()
    chdir("boppf/utils")

    def simulation(mu0):
        means = [mu0, 2 * mu0, 3 * mu0]
        stds = [5 * mu0, 10 * mu0, 15 * mu0]
        fractions = [0.5, 0.25]
        vol_frac = particle_packing_simulation(uid, particles, means, stds, fractions)
        return vol_frac

    vol_fracs0 = [simulation(10) for _ in range(5)]
    vol_fracs1 = [simulation(100) for _ in range(5)]

    vf_mean0 = np.mean(vol_fracs0)
    vf_mean1 = np.mean(vol_fracs1)

    vf_std0 = np.std(vol_fracs0)
    vf_std1 = np.std(vol_fracs1)

    chdir(cwd)

    if abs(vf_mean0 - vf_mean1) >= 0.005:
        raise ValueError("means are different by more than 0.5%")

    return vf_mean0, vf_mean1, vf_std0, vf_std1


def symmetry_test():

    cwd = getcwd()
    chdir("boppf/utils")

    def swapFirstTwo(list):
        """https://www.geeksforgeeks.org/python-program-to-swap-two-elements-in-a-list/"""
        pos1 = 0
        pos2 = 1
        list[pos1], list[pos2] = list[pos2], list[pos1]
        return list

    def simulation(swap: bool):
        mu0 = 10.0
        means = [mu0, 2.0 * mu0, 3.0 * mu0]
        stds = [5.0 * mu0, 10.0 * mu0, 15.0 * mu0]
        fractions = [0.5, 0.25]
        if swap:
            means = swapFirstTwo(means)
            stds = swapFirstTwo(stds)
            fractions = swapFirstTwo(fractions)
        vol_frac = particle_packing_simulation(uid, particles, means, stds, fractions)
        return vol_frac

    vol_fracs0 = [simulation(False) for _ in range(5)]
    vol_fracs1 = [simulation(True) for _ in range(5)]

    vf_mean0 = np.mean(vol_fracs0)
    vf_mean1 = np.mean(vol_fracs1)

    vf_std0 = np.std(vol_fracs0)
    vf_std1 = np.std(vol_fracs1)

    chdir(cwd)

    if abs(vf_mean0 - vf_mean1) >= 0.005:
        raise ValueError("means are different by more than 0.5%")

    return vf_mean0, vf_mean1, vf_std0, vf_std1


if __name__ == "__main__":
    vf_mean0, vf_mean1, vf_std0, vf_std1 = symmetry_test()
    vf_mean0, vf_mean1, vf_std0, vf_std1 = degree_of_freedom_test()
    1 + 1


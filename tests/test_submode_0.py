from os import chdir, getcwd
from boppf.utils.particle_packing import particle_packing_simulation, write_input_file


uid = "test"
particles = 10000
means = [3.0, 4.0, 5.0]
stds = [0.5, 1.0, 0.75]
fractions = [0.5, 0.2, 0.3]


def test_submode_0():
    cwd = getcwd()
    chdir("boppf/utils")
    vol_frac = particle_packing_simulation(
        uid=uid,
        particles=particles,
        means=means,
        stds=stds,
        fractions=fractions,
        max_submodes_per_mode=33,
    )
    return vol_frac


if __name__ == "__main__":
    vol_frac = test_submode_0()
    1 + 1


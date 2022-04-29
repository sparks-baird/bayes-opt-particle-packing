from boppf.utils.particle_packing import write_input_file


uid = "test"
particles = 1000
means = [120, 120, 120]
stds = [1, 1, 1]
fractions = [0.33, 0.33, 0.34]


def test_write_input_file():
    write_input_file(uid, particles, means, stds, fractions)

if __name__ == "__main__":
    test_write_input_file()


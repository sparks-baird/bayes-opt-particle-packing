import pandas as pd
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


def load_data(fname="packing-fraction.csv", folder="data"):
    df = pd.read_csv(join(folder, fname))
    df = df.rename(columns=mapper)

    X_train = df[var_names]
    y_train = df[target_name]

    X_train = X_train.drop(columns=[frac_names[-1]])

    return X_train, y_train

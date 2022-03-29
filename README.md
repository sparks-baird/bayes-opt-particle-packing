# Bayesian Optimization of Particle Packing Fractions (BOPPF)

Bayesian optimization of particle packing fractions for nuclear fuels. The objective function (not released here) is based on proprietary code from Northrop Grumman Innovation Systems (NGIS).

To reproduce, this requires a proprietary executable (renamed to
`particle_packing_sim.exe`), a MATLAB file
for writing the input files (renamed to `write_input_file.m`), and a MATLAB file for reading the volume fraction from
the output files (renamed to `read_vol_frac.m`). These files should be placed into the [boppf/utils](boppf/utils).

## Installation

A local installation can be performed via:
```bash
conda create -n packing python==3.9.*
conda activate packing
git clone https://github.com/sparks-baird/bayes-opt-particle-packing.git
cd bayes-opt-particle-packing
conda install flit
flit install --pth-file
```

## Usage
The following is based on [boppf_example.py](examples/boppf_example.py)

First, take care of imports.
```python
from boppf.boppf import BOPPF
from boppf.utils.data import load_data
```
Load `X_train` and `y_train` from [packing-fraction.csv](data/packing-fraction.csv)

| Run 	| ID 	| Mean_Particle_Size_#1 	| SD_of_Particle_#1 	| Mean_Particle_Size_#2 	| SD_of_Particle_#2 	| Mean_Particle_Size_#3 	| SD_of_Particle_#3 	| Particle_#1_Mass_Fraction 	| Particle_#2_Mass_Fraction 	| Particle_#3_Mass_Fraction 	| Packing_Fraction 	|
|-----	|----	|-----------------------	|-------------------	|-----------------------	|-------------------	|-----------------------	|-------------------	|---------------------------	|---------------------------	|---------------------------	|------------------	|
| 1   	| 0  	| 20                    	| 1                 	| 40                    	| 2.8284            	| 60                    	| 5.1962            	| 0.2239                    	| 0.597                     	| 0.1791                    	| 0.74             	|
| 2   	| 1  	| 20                    	| 1                 	| 40                    	| 2.8284            	| 60                    	| 779.4229          	| 0.2239                    	| 0.597                     	| 0.1791                    	| 0.737            	|
| .   	| .  	| .                     	| .                 	| .                     	| .                 	| .                     	| .                 	| .                         	| .                         	| .                         	| .                	|

```python
data_dir = "data"
fname = "packing-fraction.csv"
X_train, y_train = load_data(fname="packing-fraction.csv", folder="data")

n_sobol = 16
n_bayes = 1000 - 16
particles = int(1.5e6)

boppf = BOPPF(n_sobol=n_sobol, n_bayes=n_bayes, particles=particles)
best_parameters, means, covariances, ax_client = boppf.optimize(
    X_train, y_train, return_ax_client=True
)
```

The Ax `experiment` object and a tabular summary are saved to the `results` directory.

# Bayesian Optimization of Particle Packing Fractions (BOPPF)

Bayesian optimization of particle packing fractions for solid rocket propellant fuels. The objective function (not released here) is based on proprietary code from Northrop Grumman Innovation Systems (NGIS).

To reproduce, this requires a proprietary Windows executable (renamed to
`particle_packing_sim.exe`) and Python functions contained in an unreleased module named `proprietary.m` for writing the input files and reading the volume fraction from the output files. These files should be placed into the [boppf/utils](boppf/utils) directory.

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
<!-- To be able to run the two MATLAB scripts (which again, are not released here) requires an active MATLAB subscription and installation and the Statistics and Machine Learning Toolbox. The MATLAB version must be [compatible](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf) with the Python version that you're using. For example, `R2022a` supports Python `3.8` and `3.9`. Additionally, you will need to run a `setup.py` script **as an administrator** within the MATLAB installation directory per MATLAB's [instructions](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html). Make sure that your `conda` environment is activated when you do this.

You must also add the [boppf/utils](boppf/utils) folder to the MATLAB path [in a persistent way](https://www.mathworks.com/matlabcentral/answers/97990-how-do-i-make-permanent-changes-to-the-matlab-path-using-command-line-functions) to accomodate the use of Ray Tune within Ax to perform asynchronous hyperparameter optimization scheduling. -->

<!-- Replace `<matlabroot>` with the appropriate path to the MATLAB installation directory. For example, `C:\Program Files\MATLAB\R2021a`, and run the following commands:
```bash
cd "<matlabroot>\extern\engines\python"
python setup.py build --build-base="C:\Temp" install
```
`--build-base="C:\Temp"` circumvents "access denied" issues that can crop up even when running in an Administrator shell by building in a temporary directory (Windows/Anaconda/MATLAB issue). -->

<!-- For troubleshooting issues with Windows/Anaconda/MATLAB installation, see also [1](https://www.mathworks.com/matlabcentral/answers/346068-how-do-i-properly-install-matlab-engine-using-the-anaconda-package-manager-for-python), [2](https://stackoverflow.com/questions/33357739/problems-installing-matlab-engine-for-python-with-anaconda), [3](https://stackoverflow.com/questions/50488997/anaconda-python-modulenotfounderror-no-module-named-matlab). -->

## Usage
The following is based on [boppf_example.py](examples/boppf_example.py), which can be run via `python examples/boppf_example.py`

First, take care of imports.
```python
from boppf.boppf import BOPPF
from boppf.utils.data import load_data
```
<!-- Load `X_train` and `y_train` from [packing-fraction.csv](data/packing-fraction.csv)

| Run 	| ID 	| Mean_Particle_Size_#1 	| SD_of_Particle_#1 	| Mean_Particle_Size_#2 	| SD_of_Particle_#2 	| Mean_Particle_Size_#3 	| SD_of_Particle_#3 	| Particle_#1_Mass_Fraction 	| Particle_#2_Mass_Fraction 	| Particle_#3_Mass_Fraction 	| Packing_Fraction 	|
|-----	|----	|-----------------------	|-------------------	|-----------------------	|-------------------	|-----------------------	|-------------------	|---------------------------	|---------------------------	|---------------------------	|------------------	|
| 1   	| 0  	| 20                    	| 1                 	| 40                    	| 2.8284            	| 60                    	| 5.1962            	| 0.2239                    	| 0.597                     	| 0.1791                    	| 0.74             	|
| 2   	| 1  	| 20                    	| 1                 	| 40                    	| 2.8284            	| 60                    	| 779.4229          	| 0.2239                    	| 0.597                     	| 0.1791                    	| 0.737            	|
| .   	| .  	| .                     	| .                 	| .                     	| .                 	| .                     	| .                 	| .                         	| .                         	| .                         	| .                	|
 -->
<!-- 
```python
data_dir = "data"
fname = "packing-fraction.csv"
X_train, y_train = load_data(fname="packing-fraction.csv", folder="data")
```
 -->
Define how many pseudo-random initial Sobol points to generate (`n_sobol`, typical is twice the number of parameters), the number of Bayesian optimization iterations `n_bayes`, and the number of particles to drop in each simulation (`particles`).
```python
n_sobol = 10
n_bayes = 40
particles = int(2.5e4)
```

Instantiate the `BOPPF` class, and call the `optimize` method.
```python
boppf = BOPPF(n_sobol=n_sobol, n_bayes=n_bayes, particles=particles)
best_parameters, means, covariances, ax_client = boppf.optimize(np.array([]), np.array([]), return_ax_client=True)
```

The Ax `experiment` object and a tabular summary are saved to the `results` directory.

Setup
-----

 - Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for Python 3.7
 - Run `conda env create -f environment.yml`
 - Run `conda activate tfp`
 - Run `jupyter notebook`

After setup, only the last two commands need to be run to re-initialize the environment.
To update the environment in case `environment.yml` changes, run `conda env update environment.yml`.

The pythia compilation flags can be found by running `pythia8-config --libs --cflags`

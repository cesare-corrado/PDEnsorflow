# PDEnsorflow 1.2.2

**PDEnsorflow**  is a library developed under `TensorFlow 2.X` to solve Partial dfferential equations.
Since version 1.2, it implements finite differences and finite element solvers.


## Pre-requisites
Install `TensorFlow 2.X` using anaconda. First create an environment and activate it; e.g.: 

```
conda create --name PDEnsorflow python=3.9
conda activate PDEnsorflow
```

If you want to just install *TensorFlow*, follows  [this link](https://www.tensorflow.org/install/pip). 

To install **PDEnsorflow**, proceed as follows:

Install `cudatoolkit` version 11.2 and `cudnn` version 8.1.0, as follows:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
Set up the `LD_LIBRARY_PATH` to the conda environment:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```
Finally, cd to `PDEnsorflow` main directory and install with `pip`:
```
cd PDEnsorflow
python -m pip install -e .
```


## Run the code


**Old version:** Once the environment is activated, source the `init.sh` file in the main directory of PDEnsorflow:

```
source init.sh
```

**New Version:** The paths are authomatically set within `__init__.py`.

Then, launch one of the examples; e.g.:

```
cd Tests/Fenton_atria
python fenton.py
```


**Note**: *This run **PDEnsorflow** under GPU, provided that libraries are correctly installed. Otherwise, it will run under standard CPU. 
In the examples, the console will show under wich device the code is executed.

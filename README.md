# PDEnsorflow 1.3.1

**PDEnsorflow**  is a library developed under `TensorFlow 2.X` to solve Partial dfferential equations.
Since version 1.2, it implements finite differences and finite element solvers.


## Pre-requisites
The only pre-requisite is `anaconda`/`conda`. First create an environment and activate it; e.g.:

```
conda create --name PDEnsorflow python=3.11
conda activate PDEnsorflow
```


## Install
From the repository root (the directory that contains `setup.py`, i.e. the top-level
`PDEnsorflow` folder of this repository, **not** the inner `PDEnsorflow/PDEnsorflow`
package folder), install with `pip`:

```
python -m pip install -e .
```

This single command installs **PDEnsorflow** together with all of its dependencies,
including the latest `TensorFlow`. On Linux it installs `tensorflow[and-cuda]`, so the
CUDA runtime is pulled in automatically and **no manual `cudatoolkit`/`cudnn`/`cuda-nvcc` installation is required**. 
The CUDA library paths are configured automatically at import time by `gpuSolve/__init__.py`, 
so you do **not** need to set `LD_LIBRARY_PATH` by hand either.

If you want to just install *TensorFlow* manually, follow [this link](https://www.tensorflow.org/install/pip).


## Run the code

Activate the environment 
```
conda activate PDEnsorflow
```


then, launch one of the examples; e.g.:

```
cd PDEnsorflow/Tests/FEM/Fenton
python fenton.py
```


**Note**: *This run **PDEnsorflow** under GPU, provided that libraries are correctly installed. Otherwise, it will run under standard CPU. 
In the examples, the console will show under wich device the code is executed.

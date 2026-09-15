# PDEnsorflow 1.5.0

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
The CUDA library paths are configured automatically, so you do **not** need to set
`LD_LIBRARY_PATH` by hand either: when installed into a conda environment the install also
writes a conda activation hook, so after `conda activate` **any** program (even a bare
`import tensorflow`) finds the GPU; importing `gpuSolve` sets the same paths at import time
as a fallback.

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

## Command-line interface

Since version 1.4, installing the package also installs a `PDEnsorflow` command.
It runs a finite-element monodomain (or pure diffusion) simulation described by a
parameter file and a list of flags, so no Python script is needed:

```
PDEnsorflow +F parameters.par
PDEnsorflow +F parameters.par -meshname atrium -tend 500
PDEnsorflow -dt 25 +F parameters.par +Save resolved.par
```

The parameter file is the `.par` format used by openCARP, and the mesh is read
from the `.pts` / `.elem` / `.lon` triple named by `meshname`, with **node
coordinates in micrometres** and **conductivities in S/m**, as that format
specifies. The conversion to the units the solver works in happens while the
parameters are read.

Three points are worth knowing before writing a parameter file:

* **Options are applied strictly from left to right, and the last definition of
  a key wins.** A flag placed *before* `+F` is therefore overridden by the file,
  and one placed *after* it is not. This is the rule the format defines; it is
  not "the command line beats the file".
* **An unknown key stops the run**, naming the key, so a misspelled parameter is
  never silently inert. Keys that are understood but describe something this
  solver does differently (`mass_lumping`, `parab_solve`, `bidomain`) are
  accepted and reported in the run banner.
* **A key that is absent takes the documented default of that format**, not a
  gpuSolve default, so a file means the same thing here as it does there. The
  one exception is a cell parameter that no `im_param` names: it keeps the
  default of the gpuSolve cell model, so that a parameter file and a
  hand-written script agree.

`PDEnsorflow --help` lists every key that is understood, with its type and
default, and the cell models available for `imp_region[].im`.

`PDEnsorflow +Save resolved.par` writes the fully resolved parameter set back
out, which is the quickest way to see what a command line actually asked for.

A worked example, with the same simulation expressed both as a parameter file
and as a Python script so the two can be compared, is in
`PDEnsorflow/Tests/FEM/mMS_carp_compatibility`.

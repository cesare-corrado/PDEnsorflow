# PDEnsorflow 1.9.2

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
* **A cell parameter can be modified instead of assigned.** An `im_param` item
  is `name<op>value`, where `op` is one of `= + - / *`: `GNa=0.3` assigns,
  while `GNa*0.3`, `GNa/2`, `GNa+0.3` and `GNa-0.3` act on the cell model
  default. A trailing `%` makes the operand that percentage of the default, so
  `GNa-10%` is `GNa - 0.1 GNa`. The base is the **gpuSolve** default of the
  parameter, which is the same rule as for an unnamed parameter above. A
  malformed modifier stops the run rather than silently keeping the default.
* **Ionic plugins are set per region.** `imp_region[].plugins` is a
  `:`-separated list of plugins added to the cell model of that region, and
  `imp_region[].plug_param` holds one parameter list per plugin, also
  `:`-separated, with the same syntax as `im_param`
  (`plug_param = "sigma*2:..."` tunes the first plugin, then the second). A
  plugin listed in some regions only adds its current there. A plugin listed
  twice in one region is refused: the reference implementation accepts it but
  tunes only the first copy.

* **A region can start from a single-cell state file.**
  `imp_region[].im_sv_init` names a `.sv` file, written by
  `singlecell --save-ini-file` or by the reference's own single-cell tool, and
  its state is copied onto every node of that region before the run starts, as
  the reference does. The file sets the state variables and the potential, not
  the parameters: a parameter stored in it is reported and ignored, so
  `im_param` stays in charge (the same rule as `singlecell --read-ini-file`). A
  file written for another cell model stops the run. When prepacing is switched
  on as well, the prepacing train departs from this state.

`PDEnsorflow --help` lists every key that is understood, with its type and
default, and the cell models and plugins available for `imp_region[].im` and
`imp_region[].plugins`.

`PDEnsorflow +Save resolved.par` writes the fully resolved parameter set back
out, which is the quickest way to see what a command line actually asked for.

A worked example, with the same simulation expressed both as a parameter file
and as a Python script so the two can be compared, is in
`PDEnsorflow/Tests/FEM/mMS_carp_compatibility`. `PDEnsorflow/Tests/FEM/StateInit`
shows `imp_region[].im_sv_init` against the single-cell trajectory it comes
from.

## Single-cell interface

Installing the package also installs a `singlecell` command. It runs one cell
of a gpuSolve ionic model with the options of openCARP's single-cell tool,
`bench`, spelled and parsed the same way:

```
singlecell --imp Courtemanche --imp-par "GKr*1.6" --stim-curr 20 --numstim 4 --bcl 1000
singlecell --imp Tomek --numstim 50 --bcl 1000 -F paced.sv -S 49000
singlecell --imp Tomek --read-ini-file paced.sv --duration 1000 -v
```

It implements regular and irregular pacing (`--numstim`, `--bcl`,
`--stim-start`, `--stim-times`, `--DIA`, `--stim-curr`, `--stim-dur`), the
run control (`--duration`, `--past-stim`, `--dt`), the models and plugins
(`--imp`, `--imp-par`, `--plug-in`, `--plug-par`, with the same names and
modifiers as the parameter file), the information modes (`--list-imps`,
`--imp-info`, `--plugin-outputs`), the output (`--dt-out`, `--start-out`,
`--fout`, `-B`, `-u`, `-v`) and the single-cell state files
(`--save-ini-file`, `--save-ini-time`, `--read-ini-file`). The output files and
the state files have bench's formats and names, so the tools that read bench
output read these too, and a state file written by either program can be read
by the other. `singlecell --help` lists the options.

Points that differ from `bench`, on purpose:

* **A bench option that is not implemented yet stops the run** (restitution,
  voltage clamp, light, strain, several cells, ...), rather than running a
  different experiment. So does a word that is not an option, which `bench`
  drops silently.
* **The default model is `Courtemanche`** (bench's `DrouhardRoberge` is not a
  gpuSolve model), and the models use gpuSolve's defaults and integration
  scheme, so a single cell and a tissue run of the same model agree.
  `--reference-scheme` selects the reference's scheme where gpuSolve differs
  (Tomek's gates and IKr chain, `Defib_AshiharaTrayanova`'s current).
  gpuSolve's `MitchellSchaeffer` works in mV with `tau_out = 6`;
  `--imp-par "V_min=0,V_max=1,tau_out=5"` gives bench's.
* **A state file sets states, not parameters.** bench also takes the
  parameters stored in the file (for example `GKr` of tenTusscherPanfilov),
  over `--imp-par`; `singlecell` keeps the defaults and `--imp-par`, and prints
  a warning at the start and at the end of the run for each file parameter
  that differs. The files it writes hold the parameters the run used, at full
  precision (bench writes 6 digits).
* **One cell runs on the CPU, on one thread**, which is faster than a GPU for
  so little work. `--target mlir-cuda` selects the GPU.
* The ionic model trace (`Trace_0.dat`) is not written yet.

`PDEnsorflow/Tests/DEVTESTS/ionic/singlecell_vs_bench.py` runs both programs on
every shared model and compares them.

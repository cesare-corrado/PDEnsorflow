# Physics

This sub-module of `gpuSolve` defines the Physics of the problems as classes.   
These classes are designed to handle data and matrix assembly of a specific problem and to advance one time step.   
It contains the following modules:

* `HeatSolver`: Solves the heat equation (parabolic solver)
* `MonodomainSolver`: Solves the heat monodomain Problem. In inherints from HeatSolver for the solution of the diffusion.

Both solvers can save and resume their state:

* `checkpoint()` returns `{'ionic_model', 'time', 'num_nodes', 'Vm', 'state_variables'}` at `ctime()`, every nodal array in the user's node order, so it does not depend on the renumbering setting.
* `restore_checkpoint(checkpoint)` sets the potential, the cell-model state and `ctime()`, before or after `finalize_for_run()`. It refuses a checkpoint from another mesh or cell model. The CG warm-start history is not restored, so a resumed run agrees with an uninterrupted one to within the CG tolerance.

It also contains two monitoring / conditioning classes:

* `LatDetector`: records local activation times during a run, the equivalent of the reference simulator's LAT detection (user guide 22.1/22.2/22.8). The caller feeds it `check(vm, t)` once per step; the parameter-file front end builds one per `lats[]` entry and does that itself (`num_LATs`, `LAT_ID`, `lats[].method/.threshold/.mode/.all/.start/.ID`). Only Vm is measured: `measurand = 1` (Phie) needs an extracellular solve this library does not perform, so such an entry is reported and dropped. Two execution knobs exist, both off by default. `set_use_graph_check(True)` runs one step as a single compiled graph, traced once in `init()`: it writes a byte-identical activation file and roughly halves the cost of a step on an idle RTX 5090 at every size from 63k to 4M nodes (e.g. at 1M nodes, `all = 0`: 1418 us against 625 us). Switch it on for a run of any length; it is off by default only because a caller that steps a handful of times never amortises the trace. `set_use_deferred_count(True)` keeps the per-step activation count on the device, so `check()` answers with a scalar tensor and no host wait: it measured **slower** in all sixteen configurations tried (by 100 to 220 us per step, 63k to 4M nodes), because the read-back also paces the eager queue, so it is for a caller that must never block rather than for speed. `Tests/DEVTESTS/LAT/lat_detector_scaling.py` reproduces both measurements, and needs an idle GPU to mean anything.
* `Prepacer`: conditions the cell models *before* a run, the equivalent of the reference simulator's `prepacing_*` parameters (user guide 22.3&ndash;22.7). It paces single cells for `beats` beats at `bcl` and hands every node the state of its cell at the moment matching the node's place in the activation sequence, read from a per-node activation-time file (the one `LatDetector.write()` writes with `all = 0`). It must run after the initial condition and before `finalize_for_run()`, and it writes its result through `restore_checkpoint()`.

`Prepacer` paces **one cell per mesh region** by default, which is what the reference does and costs a single cell on a homogeneous mesh. `set_group_by_region(False)` paces one cell per node instead: the same ODEs with the conductivity switched off, needed once a cell parameter varies *within* a region. The two agree to the bit when the parameters are uniform. Left unset, the choice is automatic: one cell per node as soon as a cell parameter was registered as a per-node (`'nodal'`) property, one per region otherwise.

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

* `LatDetector`: records local activation times during a run, the equivalent of the reference simulator's LAT detection (user guide 22.2/22.8).
* `Prepacer`: conditions the cell models *before* a run, the equivalent of the reference simulator's `prepacing_*` parameters (user guide 22.3&ndash;22.7). It paces single cells for `beats` beats at `bcl` and hands every node the state of its cell at the moment matching the node's place in the activation sequence, read from a per-node activation-time file (the one `LatDetector.write()` writes with `all = 0`). It must run after the initial condition and before `finalize_for_run()`, and it writes its result through `restore_checkpoint()`.

`Prepacer` paces **one cell per mesh region** by default, which is what the reference does and costs a single cell on a homogeneous mesh. `set_group_by_region(False)` paces one cell per node instead: the same ODEs with the conductivity switched off, needed once a cell parameter varies *within* a region. The two agree to the bit when the parameters are uniform. Left unset, the choice is automatic: one cell per node as soon as a cell parameter was registered as a per-node (`'nodal'`) property, one per region otherwise.

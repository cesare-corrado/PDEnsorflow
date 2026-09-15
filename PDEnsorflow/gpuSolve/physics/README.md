# Physics

This sub-module of `gpuSolve` defines the Physics of the problems as classes.   
These classes are designed to handle data and matrix assembly of a specific problem and to advance one time step.   
It contains the following modules:

* `HeatSolver`: Solves the heat equation (parabolic solver)
* `MonodomainSolver`: Solves the heat monodomain Problem. In inherints from HeatSolver for the solution of the diffusion.

Both solvers can save and resume their state:

* `checkpoint()` returns `{'ionic_model', 'time', 'num_nodes', 'Vm', 'state_variables'}` at `ctime()`, every nodal array in the user's node order, so it does not depend on the renumbering setting.
* `restore_checkpoint(checkpoint)` sets the potential, the cell-model state and `ctime()`, before or after `finalize_for_run()`. It refuses a checkpoint from another mesh or cell model. The CG warm-start history is not restored, so a resumed run agrees with an uninterrupted one to within the CG tolerance.

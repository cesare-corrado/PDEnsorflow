# Physics

This sub-module of `gpuSolve` defines the Physics of the problems as classes.   
These classes are designed to handle data and matrix assembly of a specific problem and to advance one time step.   
It contains the following modules:

* `HeatSolver`: Solves the heat equation (parabolic solver)
* `MonodomainSolver`: Solves the heat monodomain Problem. In inherints from HeatSolver for the solution of the diffusion.

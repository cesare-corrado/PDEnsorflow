# writers

This is the sub-directory of I/O contains functions for outputting data.

* `VedoPlotter`: a class to plot data in *vedo* format
* `ResultWriter`: a class to write results into a *numpy* format


## VedoPlotter

This generates a *vedo* object (through the *vedo* library).



## ResultWriter

This class store the output in a *numpy* array format and writes it to a file.

### IGBWriter

This class writes the output in *igb* format (meshalyzer).

## StateWriter

This class writes a checkpoint, the dict returned by `HeatSolver.checkpoint()` /
`MonodomainSolver.checkpoint()`, as a pickle file (see `readers/StateReader`).
`write(checkpoint, fname)` stores plain values only, and writes to
`<fname>.partial` before renaming it, so a run killed while saving keeps the
previous file intact.

## CarpMeshWriter

This class writes a mesh in carp format (*.pts*, *.elem* and *.lon* files)
Memeber methods:

* `assignMesh(msh)`: assigns the mesh *msh* to the writer
* `Mesh()`: returns the mesh
* `writeMesh(fprefix)`: writes the mesh in CARP format, using *fprefix* as the prtefix. The prefix must contain the path.



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

## RoeWriter
Writes a binary ionic state file (`.roe`) byte for byte as the reference
simulator does. `write(content, fname)` takes the dict `RoeReader.read()`
returns, so a file read and written back is identical; `format` (default 1, the
release-19.0 format), `revision`, `created` and `index_bytes` fall back to the
writer's settings when the content has none. `encode_records(state, fields,
packed)` turns per-node field values into records (gates cast to float32,
padding zero). The module functions name the files as the reference does:
`tsav_file_name(write_statef, tsav_label(t))` (`state.7.3.roe`),
`checkpoint_file_name(t)` (`checkpoint.10.0.roe`) and `exit_file_name(t)`
(`exit.save.12.500.roe`). A plugin entry without `stored_size` gets its true
size in the region table: release 19.0 restores a plugin only from such a file,
although it writes the model's size there itself.

## CarpMeshWriter

This class writes a mesh in carp format (*.pts*, *.elem* and *.lon* files)
Memeber methods:

* `assignMesh(msh)`: assigns the mesh *msh* to the writer
* `Mesh()`: returns the mesh
* `writeMesh(fprefix)`: writes the mesh in CARP format, using *fprefix* as the prtefix. The prefix must contain the path.



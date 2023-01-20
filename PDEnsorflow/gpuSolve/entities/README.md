# Entities

This sub-module of `gpuSolve` defines some general entities as classes that collect data into the same object

* `Domain3D`
* `Triangulation`

## Domain3D

This class implements a 3D domain; the domain gathers information on:

*  the voxel sizes
*  the anatomy
*  the conductivity(ies)
*  the fibres, if any

### Specify the anatomy

It is possible to load an anatomy from a file (`.png`, figures, `NifTI` images and `numpy` files) using the function `load_geometry_file`. When no file is specified, the anatomy is a 3D cube with dimensions (width, height, depth).
Otherwise, it is possible to specify the (binary) tensor directly with the function `assign_geometry`. 

To access the geometry, use the get function `geometry()`.


To load the conductivity, use the function `load_conductivity` (same inputs format of the anatomy), or specify the tensor with `assign_conductivity`. Both the functions check if the conductivity is isotropic or not; for anisotropic conductivity, both functions require transverse (t) and longitudinal (l) conductivities as the input and post-process the second channel to the difference of l-t.

To access the conductivity, use the get function `conductivity()`.
It is possible to determine if a conductivity is isotropic or not with get function `anisotropic()`.


To load fibers, use the function `load_fiber_direction`. The function internally build the 6 entries `a_i a_j` of the tensor that are used in the Laplace operator.

To access the tensor of `a_i a_j`, use the get function `fibtensor()`.

## Triangulation
This class implements the handler for triangulations (Finite Elements/Volumes Meshes). It gathers information on:

* The coordinates of the mesh vertices
* The Element types, region IDs and connectivity
* The fibres directions

### Data
A python dict contains all the element (one key per type) that belong to the mesh 
(It can handle hybrid meshes). Data are stored in numpy format (CPU) as meshes are used 
to only assemble matrices.


### Member functions
* `Pts()`: returns the point coordinates (numpy array)
* `Fibres()`: returns the fiber directions (numpy array)
* `Elems()`: returns a python dict (one entry for every element type of the mesh) with element connectivities and region ID. Element connectivities are numpy arrays of type *np.int32*
* `readMesh(filename)`: determines the mesh format and reads in the mesh.
The existing formats are:
  * Carp mesh format
  * Binary format (pkl file) with data as numpy arrays
* `saveMesh(fileName)`: saves the mesh in a *.pkl* format.
* `mesh_connectivity(storeConn=False)`: returns the mesh connectivity. When `storeConn=True`, it keeps the connectivity as an internal variable, avoiding recomputing in subsequent calls.
* `contravariant_basis(storeCbas=False)`: returns the contravariant basis evaluated on each element. For non-linear elements, it is evaluated at Gauss Points (NOT implemented yet!). When `storeCbas=True`, it keeps a copy of the contravariant_basis as an internal variable, avoiding recomputing in subsequent calls.


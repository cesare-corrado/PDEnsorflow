# Entities

This sub-module of `gpuSolve` defines some general entities as classes that collect data into the same object

* `domain3D`


## Domain3D

This class implements a 3D domain; the domain gathers information on:
* the voxel sizes
* the anatomy
* the conductivity(ies)
* the fibres, if any

### Specify the anatomy

It is possible to load an anatomy from a file (`.png`, figures, `NifTI` images and `numpy` files) using the function `load_geometry_file`. When no file is specified, the anatomy is a 3D cube with dimensions (width, height, depth).
Otherwise, it is possible to specify the (binary) tensor directly with the function `assign_geometry`. 

To access the geometry, use the get function `geometry()`.


To load the conductivity, use the function `load_conductivity` (same inputs format of the anatomy), or specify the tensor with `assign_conductivity`. Both the functions check if the conductivity is isotropic or not; for anisotropic conductivity, both functions require transverse (t) and longitudinal (l) conductivities as the input and post-process the second channel to the difference of l-t.

To access the conductivity, use the get function `conductivity()`.
It is possible to determine if a conductivity is isotropic or not with get function `anisotropic()`.


To load fibers, use the function `load_fiber_direction`. The function internally build the 6 entries `a_i a_j` of the tensor that are used in the Laplace operator.

To access the tensor of `a_i a_j`, use the get function `fibtensor()`.



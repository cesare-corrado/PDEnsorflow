# Entities

This sub-module of `gpuSolve` defines some general entities as classes that collect data into the same object

* `Domain3D`
* `Triangulation`
* `MaterialProperties`


These objects do not require Tensorflow.

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
Data are stored on the CPU (numpy and native python formats) as meshes are used to only assemble matrices.
Attributes are:
* `_Pts`: a numpy array with node coordinates
* `_Elems`: a python dict contains all the element (one key per type) that belong to the mesh  (can handle hybrid meshes). Each entry is a numpy array of integer of shape *nv*+1 where *nv* is the nb of vertices of the element.
* `_Fibres`: a numpy array with the fiber tensor
* `_connectivity`: a python dict that contains the mesh connectivity. **NOT** evaluated by default.
* `_contravbasis`: (deprecated?) a python dict with of the same format of `_Elems` but with the contravariant basis and the element measures. **NOT** evaluated by default.
* `_pointRegIDs`: a numpy array with the region *ID* for each node. **NOT** evaluated by default.


### Member functions
* `Pts()`: returns the point coordinates (numpy array)
* `Fibres()`: returns the fiber directions (numpy array)
* `Elems()`: returns a python dict (one entry for every element type of the mesh) with element connectivities and region ID. Element connectivities are numpy arrays of type *np.int32*
* `readMesh(filename)`: determines the mesh format and reads in the mesh.
The existing formats are:
  * Carp mesh format
  * Binary format (pkl file) with data as numpy arrays
* `saveMesh(fileName)`: saves the mesh in a *.pkl* format.
* `exportCarpFormat(foutSuffix)`: exports the mesh in carp format (*.pts*, *.elem*, *.lon*), with suffix `foutSuffix`.
* `mesh_connectivity(storeConn=False)`: returns the mesh connectivity. When `storeConn=True`, it keeps the connectivity as an internal variable, avoiding recomputing in subsequent calls.
* `contravariant_basis(storeCbas=False)`: returns the contravariant basis evaluated on each element. For non-linear elements, it is evaluated at Gauss Points (NOT implemented yet!). When `storeCbas=True`, it keeps a copy of the contravariant_basis as an internal variable, avoiding recomputing in subsequent calls.
* `point_region_ids(storeIDs=False)`: returns the region *ID* associated to each vertex. When `storeIDs=True`, it keeps a copy of the point IDs as an internal variable, avoiding recomputing in subsequent calls.
* `element_contravariant_basis(elemtype,elemID,localcoords=[])`: computes the contravariant basis at coordinates localcoords for the element elemID of type elemType and returns a python dict with the contravariant bais vectors ( v{1,2,3}) and the element measure
* `release_contravariant_basis()`: deletes the contravariant basis dictionary and releases the memory 
* `release_connectivity()`: deletes the connectivity dictionary and releases the memory
* `release_point_region_ids()`: deletes the point region IDs array and releases the memory


## `MaterialProperties`

This class collects all the material properties associated to nodes or elements and implements some proxy functions to access the values.

### Data
Material properties are associated to elements (e.g. *diffusivity*) or to vertices (e.g *gNa* in cardiac sdimulations). Two python dicts collect material propertiesa associated to elements and nodes:

* `_element_properties`: material properties associated to the elements.
* `_nodal_properties`: material properties associated to the nodes/vertices.

dict keys are the property names (e.g. *diffusivity*, *gNa*). Each property is characterised by a *type*: {*region*, *elem*} for the elements and {*region*, *nodal*} for the nodes that specify if assigned by region *IDs* or by node/element *IDs*. Finally, the *idmap* maps the *ID* to the numerical value of the property.

Other attributes:

* `_ud_functions`: a python dict that points to user defined functions. note that in these function one can not reference the class itself



### Member functions

* `add_element_property(pname,ptype,pmap)`:  adds the element material property `pname` of type `ptype` (*uniform*, *region* or *elem*) with the mapping `pmap`.
* `add_nodal_property(pname,ptype,pmap)`: adds the nodal material property `pname` of type `ptype` (*uniform*, *region* or *nodal*) with the mapping `pmap`.
* `remove_element_property(pname)`: removes the property `pname` from the `_element_properties` dict
* `remove_nodal_property(pname)`: removes the property `pname` from the `_nodal_properties` dict
* `remove_all_element_properties()`: deletes all the material properties associated to elements and releases memory
* `remove_all_nodal_properties()`: : deletes all the material properties associated to nodes and releases memory
* `ElementProperty(pname,elemtype,elemID,regionID)`: returns the nodal property `pname` at element `elemID` of type `elemtype` or at region `regionID`
* `NodalProperty(pname,pointID,regionID)`: returns the nodal property `pname` at node `pointID` or at region `regionID`
* `add_ud_function(fname,fdef)`: assigns to `_ud_functions` dict the user-defined function `fdef` with the key `fname`
* `execute_ud_func(fname,*kwargs)`: executes the user defined function with key `fname`, passing the arguments `*kwargs`
* `remove_ud_function(fname)`: removes the function `fname` from the `_ud_functions` dict
* `element_property_names()`: returns the names (keys) of the element material properties; `None` if no properties are defined
* `nodal_property_names()`: returns the names (keys) of the nodal material properties; `None` if no properties are defined
* `element_property_type(pname)`: returns the type of the element material property `pname`; `None` if `pname` does not exist
* `nodal_property_type(pname)`: returns the type of the nodal material property `pname`; `None` if `pname` does not exist


# Matrices

This sub-module of `gpuSolve` defines some local matrices for Finite Elements and functions to assemble global sparse matrices.

* `localMass`
* `localStiffness`
* `globalMatrices`

## localMass

This module implements the functions that compute the local mass matrices; 
the function `localMass(elemtype, elemData,props=None)` takes as a input the type of element and 
the python dict `elemData` containing the contravariant basis and the element measure and
calls one of the following:

* `linear_triangular_local_Mass(elemData,props=None)`: local mass for linear triangular elements

the argument `props=[]` is a dummy argument, used to keep the same signature on local matrices and to
allow extra arguments in the future.


## localStiffness

This module implements the functions that compute the local stiffness matrices; 
the function `localStiffness(elemtype, elemData, Sigma=np.identity(3))` takes as a input the type of element,the python dict `elemData` containing the contravariant basisand the element measure, and the diffusion tensor `Sigma` and calls one of the following:

* `linear_triangular_local_Stiffness(elemData, Sigma=np.identity(3) )`: local stiffness for linear triangular elements (diffusion can be anisotropic)


## globalMatrices

This module implements functions to assemble global finite element matrices in sparse format. Sparse formats 
are characterised by a number *nnzero* of non-zero entries proportional to the matrix shape *npt*. 

###  compute_coo_pattern(connectivity)

Given the mesh connectivity (input argument), this function computes three arrays to assemble a sparse matrix in coo form: 

* I: an array of shape *nnzero* with the row IDs
* J: an array of shape *nnzero* with the column IDs
* StartIndex: an array of shape *npt+1* with the index on I and J that corresponds to the begin of each row. 

E.g., entries of row k start at index StartIndex[k] and terminate at index StartIndex[k+1]-1.


### assemble_mass_matrix(matrix_pattern,domain,connectivity=None)

This function computes the sparse mass matrix using the domain connectivity and the matrix pattern. The output is in TensorFlow sparse tensor format. Input arguments are:

* matrix_pattern: the sparsity pattern of the matrix
* domain:         the domain object
* connectivity:   the domain connectivity (if None, it is computed and kept in memory)

### assemble_stiffness_matrix(matrix_pattern,domain,matprops,stif_pname='Sigma',connectivity=None)

This function computes the sparse stiffness matrix using the domain connectivity and the matrix pattern. The output is in TensorFlow sparse tensor format. Input arguments are:

* matrix_pattern: the sparsity pattern of the matrix
* domain:         the domain object
* matprops:       a MaterialProperties object that implements functions to provide local properties 
* stif_pname:     the name of the function that evaluates the matertial properties
* connectivity:   the domain connectivity (if None, it is computed and kept in memory)

### assemble_matrices_dict(local_matrices_dict,matrix_pattern,domain,matprops,connectivity=None)

Given a python dict of functions to compute local matrices, this function computes all the global the sparse matrices using the domain connectivity and the matrix pattern. The output is a dict with elements in TensorFlow sparse tensor format. Input arguments are:

* local\_matrices\_dict: a python dict of functions to compute the local matrices
* matrix_pattern: the sparsity pattern of the matrix
* domain:         the domain object
* matprops:       a MaterialProperties object that implements functions to provide local properties
* connectivity:   the domain connectivity (if None, it is computed and kept in memory)



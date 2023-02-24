import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

import sys
from time import time
from gpuSolve.matrices.localMass import localMass 
from gpuSolve.matrices.localStiffness import localStiffness


def compute_coo_pattern(connectivity: dict) -> dict:
    """ function compute_coo_pattern(connectivity)
    computes the patterns of a COO matrix from
    the mesh connectivity.
    Input: 
        connectivity: the mesh connectivity
    Output (dictionary):
        I: an array of shape nnzero with the row IDs
        J: an array of shape nnzero with the column IDs
        StartIndex: an array of shape npt+1 with the index on
                    I and J that corresponds to the begin of 
                    each row. E.g., entries of row k are in
                    the range StartIndex[k]:(StartIndex[k+1]-1)
    Note: this function also provides the CSR pattern, as StartIndex provides
          the starting/ending indices of each row
    """
    npt     = len(connectivity)
    nzero   = 0
    print('Computing sparsity pattern for coo type matrices',flush=True)
    t0 = time()
    for vertices,loc_conn in connectivity.items():
        nzero += (loc_conn.shape[0])
    I             = np.zeros(shape=(nzero),dtype=int)
    J             = np.zeros(shape=(nzero),dtype=int)
    StartIndex    = np.zeros(shape=(npt+1),dtype=int)
    StartIndex[0] = 0
    k             = -1
    for jpt in range(npt):
        loc_con = connectivity[jpt]
        StartIndex[jpt+1] = StartIndex[jpt]+loc_con.shape[0]
        for jloc in range(loc_con.shape[0]):
            k = k+1
            I[k] = jpt
            J[k] = loc_con[jloc]
    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)
    
    return({'I': I.astype(np.int32), 'J': J.astype(np.int32), 'StartIndex':StartIndex.astype(np.int32)})

def compute_reverse_cuthill_mckee_indexing(matrix_pattern : dict, sym_mat : bool=True) -> dict:
    """ function compute_reverse_cuthill_mckee_indexing(matrix_pattern, sym_mat = True)
    computes the reverse_cuthill_mckee indexing to concentrate the entries aroud the diagonal.
    This function evaluates the permutaion and the inverse permutation to map back to the original indexing.
    Permutations are numpy arrays of integers.
    Input:
        matrix_pattern:         the sparsity pattern of the matrix
        sym_mat (default True): a boolean flag that tells if the original matrix is symmetric or not
    Output:
        indexmap:  a dict with direct ('perm') and inverse ('iperm') permutations
    """
    indptr    = matrix_pattern['StartIndex']
    indices   = matrix_pattern['J']
    data      = np.ones(shape = indices.shape,dtype=np.float32)
    npt       = indptr.shape[0] - 1
    A0        = csr_matrix((data, indices, indptr), shape=(npt, npt))
    perm_rcm  = reverse_cuthill_mckee(A0,symmetric_mode=sym_mat).astype(np.int32)
    iperm_rcm = np.zeros(shape=npt,dtype=np.int32)
    for js,jt in enumerate(perm_rcm):
        iperm_rcm[jt] = js
    return({'perm': perm_rcm, 'iperm': iperm_rcm })    


def assemble_mass_matrix(matrix_pattern : dict,domain,connectivity: dict = None):
    """ function assemble_mass_matrix(matrix_pattern,domain,connectivity=None)
    computes the sparse mass matrix using the domain connectivity and the matrix pattern.
    It returs a TensorFlow sparse tensor.
    Input:
        matrix_pattern: the sparsity pattern of the matrix
        domain:         the domain object
        connectivity:   the domain connectivity (if None, it is computed and kept in memory)
    Output:
        MASS:  a TensorFlow sparse tensor storing the mass matrix.
    """
    npt = domain.Pts().shape[0]
    I   = matrix_pattern['I']
    J   = matrix_pattern['J']
    VM  = np.zeros(shape=I.shape)
    k0  = matrix_pattern['StartIndex']
    print('Assembly sparse mass matrix',flush=True)    
    if connectivity is None:
        connectivity = domain.mesh_connectivity(True)
    t0 = time()
    for elemtype, Elements in domain.Elems().items():
        for iElem,Elem in enumerate(Elements):
            Elem     = Elem[:-1]
            elemData = domain.element_contravariant_basis(elemtype,iElem)
            lmass    = localMass(elemtype,elemData)
            for iEntry,irow in enumerate(Elem):
                for jEntry,jcol in enumerate(Elem):
                    indexEntry      = k0[irow]+np.where(connectivity[irow]==jcol)[0]
                    VM[indexEntry] += lmass[iEntry,jEntry]
    indices   = np.hstack([I[:,np.newaxis], J[:,np.newaxis]])
    MASS      = tf.sparse.SparseTensor(indices=indices, values=VM.astype(np.float32), dense_shape=[npt, npt])
    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)    
    return(MASS)

def assemble_stiffness_matrix(matrix_pattern: dict ,domain,matprops,stif_pname: str = 'Sigma',connectivity :dict =None):
    """ function assemble_stiffness_matrix(matrix_pattern,domain,matprops,connectivity=None)
    computes the sparse stiffness matrix using the domain connectivity and the matrix pattern.
    It returs a TensorFlow sparse tensor.
    Input:
        matrix_pattern: the sparsity pattern of the matrix
        domain:         the domain object
        matprops:       a MaterialProperties object that implements functions to provide local properties 
        stif_pname:     the name of the function that evaluates the matertial properties
        connectivity:   the domain connectivity (if None, it is computed and kept in memory)
    Output:
        STIFFNESS:  a TensorFlow sparse tensor storing the mass matrix.
    """
    npt = domain.Pts().shape[0]
    I   = matrix_pattern['I']
    J   = matrix_pattern['J']
    VM  = np.zeros(shape=I.shape)
    k0  = matrix_pattern['StartIndex']
    print('Assembly sparse stiffness matrix',flush=True)    
    if connectivity is None:
        connectivity = domain.mesh_connectivity(True)
    t0 = time()
    for elemtype, Elements in domain.Elems().items():
        for iElem,Elem in enumerate(Elements):
            Elem     = Elem[:-1]
            elemData = domain.element_contravariant_basis(elemtype,iElem)
            local_props = matprops.execute_ud_func(stif_pname ,elemtype, iElem,domain,matprops)
            lstiffness  = localStiffness(elemtype,elemData,local_props)
            for iEntry,irow in enumerate(Elem):
                for jEntry,jcol in enumerate(Elem):
                    indexEntry      = k0[irow]+np.where(connectivity[irow]==jcol)[0]
                    VM[indexEntry] += lstiffness[iEntry,jEntry]
    indices   = np.hstack([I[:,np.newaxis], J[:,np.newaxis]])
    STIFFNESS = tf.sparse.SparseTensor(indices=indices, values=VM.astype(np.float32), dense_shape=[npt, npt])
    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)    
    return(STIFFNESS)


def assemble_vectmat_dict(local_matrices_dict,matrix_pattern,domain,matprops,connectivity: dict = None):
    """ function assemble_vectmat_dict(local_matrices_dict,matrix_pattern,domain,matprops,connectivity=None)
    Given a python dict of functions to compute local matrices, this function computes all the 
    vectors of the entries of the global sparse matrices using the domain connectivity and the matrix pattern.
    Input:
        local_matrices_dict: a python dict of functions to compute the local matrices
        matrix_pattern:      the sparsity pattern of the matrix
        domain:              the domain object
        matprops:            a MaterialProperties object that implements functions to provide local properties
        connectivity:        the domain connectivity (if None, it is computed and kept in memory)
    Output:
        VM:  a python dict with the numpy tensors of entries.
    """
    npt = domain.Pts().shape[0]
    k0  = matrix_pattern['StartIndex']    
    VM  = {}
    print('Assembly the following sparse matrices:'.format(len(local_matrices_dict.keys())),flush=True)    
    for matr_name in local_matrices_dict.keys():
        print('{}'.format(matr_name),flush=True)
        VM[matr_name]  = np.zeros(shape=matrix_pattern['I'].shape)
    if connectivity is None:
        connectivity = domain.mesh_connectivity(True)
    lmat = {}
    t0 = time()
    for elemtype, Elements in domain.Elems().items():
        for iElem,Elem in enumerate(Elements):
            Elem       = Elem[:-1]
            elemData   = domain.element_contravariant_basis(elemtype,iElem)            
            lmat.clear()
            # Compute the local matrices
            for matr_name,lmateval in local_matrices_dict.items():
                local_props = matprops.execute_ud_func(matr_name,elemtype, iElem,domain,matprops)
                lmat[matr_name]   = lmateval(elemtype,elemData,local_props)
            # Assembling
            for iEntry,irow in enumerate(Elem):
                for jEntry,jcol in enumerate(Elem):
                    indexEntry      = k0[irow]+np.where(connectivity[irow]==jcol)[0]
                    for matr_name,local_matrix in lmat.items():
                        VM[matr_name][indexEntry] += local_matrix[iEntry,jEntry]
    # Now generate a dict of tf.sparse objects
    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)    
    return(VM)




def assemble_matrices_dict(local_matrices_dict : dict ,matrix_pattern: dict,domain,matprops,connectivity : dict = None) -> dict[tf.sparse.SparseTensor]:
    """ function assemble_matrices_dict(local_matrices_dict,matrix_pattern,domain,matprops,connectivity=None)
    Given a python dict of functions to compute local matrices, this function computes all the 
    global the sparse matrices using the domain connectivity and the matrix pattern.
    It returs a dict of TensorFlow sparse tensors.
    Input:
        local_matrices_dict: a python dict of functions to compute the local matrices
        matrix_pattern:      the sparsity pattern of the matrix
        domain:              the domain object
        matprops:            a MaterialProperties object that implements functions to provide local properties
        connectivity:        the domain connectivity (if None, it is computed and kept in memory)
    Output:
        MATRICES:  a python dict with the TensorFlow sparse tensors storing the matrices.
    """
    npt = domain.Pts().shape[0]
    I   = matrix_pattern['I']
    J   = matrix_pattern['J']
    VM  = assemble_vectmat_dict(local_matrices_dict,matrix_pattern,domain,matprops,connectivity)
    indices  = np.hstack([I[:,np.newaxis], J[:,np.newaxis]])
    MATRICES = {}
    for matr_name,VMAT in VM.items():
        MATRICES[matr_name] = tf.sparse.SparseTensor(indices=indices,values=VMAT.astype(np.float32), dense_shape=[npt,npt])
    return(MATRICES)


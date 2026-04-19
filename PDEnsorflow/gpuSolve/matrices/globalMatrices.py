import numpy as np
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness


####################################################################################
#############                                                          #############
#############         functions that DO NOT require TensorFlow         #############
#############                                                          #############
####################################################################################

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


# Vectorized contravariant basis computation for all elements
# of a given type at once. Replaces per-element calls to
# element_contravariant_basis with batch numpy operations.
# Returns arrays of shape (nElems, 3) for basis vectors and (nElems,1) for measures.

def _batch_contravariant_basis_Edges(Pts, Elems_noids):
    """Vectorized contravariant basis for all edge elements."""
    V0 = Pts[Elems_noids[:, 0], :]  # (nElems, 3)
    V1 = Pts[Elems_noids[:, 1], :]  # (nElems, 3)
    v10 = V1 - V0                    # (nElems, 3)
    E_len = np.linalg.norm(v10, axis=1, keepdims=True)  # (nElems, 1)
    # Build covariant basis and invert for each element
    nElems = Elems_noids.shape[0]
    v1c = np.zeros((nElems, 3), dtype=np.float64)
    v2c = np.zeros((nElems, 3), dtype=np.float64)
    v3c = np.zeros((nElems, 3), dtype=np.float64)
    for e in range(nElems):
        u, s, v = np.linalg.svd(v10[e, :, np.newaxis])
        v20 = u[:, 1]
        v30 = u[:, 2]
        covbT = np.zeros((3, 3), dtype=np.float64)
        covbT[0, :] = v10[e, :]
        covbT[1, :] = v20
        covbT[2, :] = v30
        contrb = np.linalg.inv(covbT)
        v1c[e, :] = contrb[:, 0]
        v2c[e, :] = contrb[:, 1]
        v3c[e, :] = contrb[:, 2]
    return {'v1': v1c, 'v2': v2c, 'v3': v3c, 'meas': E_len}


def _batch_contravariant_basis_Trias(Pts, Elems_noids):
    """Vectorized contravariant basis for all triangular elements."""
    V0 = Pts[Elems_noids[:, 0], :]  # (nElems, 3)
    V1 = Pts[Elems_noids[:, 1], :]  # (nElems, 3)
    V2 = Pts[Elems_noids[:, 2], :]  # (nElems, 3)
    v10 = V1 - V0                    # (nElems, 3)
    v20 = V2 - V0                    # (nElems, 3)
    # Normal vector
    N12 = np.cross(v10, v20)         # (nElems, 3)
    area = np.linalg.norm(N12, axis=1, keepdims=True)  # (nElems, 1)
    v12 = N12 / area                 # (nElems, 3)
    # Covariant basis: rows are v10, v20, v12 for each element
    covbT = np.stack([v10, v20, v12], axis=1)  # (nElems, 3, 3)
    # Contravariant basis: batch matrix inverse
    contrb = np.linalg.inv(covbT)    # (nElems, 3, 3)
    v1c = contrb[:, :, 0]            # (nElems, 3)
    v2c = contrb[:, :, 1]            # (nElems, 3)
    v3c = contrb[:, :, 2]            # (nElems, 3)
    T_area = 0.5 * area              # (nElems, 1)
    return {'v1': v1c, 'v2': v2c, 'v3': v3c, 'meas': T_area}


def _batch_contravariant_basis_Tetras(Pts, Elems_noids):
    """Vectorized contravariant basis for all tetrahedral elements."""
    V0 = Pts[Elems_noids[:, 0], :]  # (nElems, 3)
    V1 = Pts[Elems_noids[:, 1], :]  # (nElems, 3)
    V2 = Pts[Elems_noids[:, 2], :]  # (nElems, 3)
    V3 = Pts[Elems_noids[:, 3], :]  # (nElems, 3)
    v10 = V1 - V0
    v20 = V2 - V0
    v30 = V3 - V0
    vprod = np.cross(v10, v20)
    # Volume = |det(v10, v20, v30)| / 6
    T_vol = np.abs(np.sum(vprod * v30, axis=1, keepdims=True)) / 6.0  # (nElems, 1)
    covbT = np.stack([v10, v20, v30], axis=1)  # (nElems, 3, 3)
    contrb = np.linalg.inv(covbT)              # (nElems, 3, 3)
    v1c = contrb[:, :, 0]
    v2c = contrb[:, :, 1]
    v3c = contrb[:, :, 2]
    return {'v1': v1c, 'v2': v2c, 'v3': v3c, 'meas': T_vol}


# Dispatch table for vectorized contravariant basis functions.
# Element types not listed here will fall back to per-element computation.
_batch_contrabasis_dispatch = {
    'Edges': _batch_contravariant_basis_Edges,
    'Trias': _batch_contravariant_basis_Trias,
    'Tetras': _batch_contravariant_basis_Tetras,
}


# Vectorized local mass matrix computation for all elements of one type.
# Produces (nElems, nV, nV) mass matrices from element measures.
# Template matrices match the original localMass functions exactly.
_mass_templates = {
    'Edges': {
        'nV': 2,
        'scale': 1.0,
        'template': np.array([[1.0/3.0, 1.0/6.0],
                              [1.0/6.0, 1.0/3.0]])
    },
    'Trias': {
        'nV': 3,
        'scale': 2.0,
        'template': np.array([[1.0/12.0, 1.0/24.0, 1.0/24.0],
                              [1.0/24.0, 1.0/12.0, 1.0/24.0],
                              [1.0/24.0, 1.0/24.0, 1.0/12.0]])
    },
    'Tetras': {
        'nV': 4,
        'scale': 6.0,
        'template': np.array([[1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0],
                              [1.0/120.0, 1.0/60.0, 1.0/120.0, 1.0/120.0],
                              [1.0/120.0, 1.0/120.0, 1.0/60.0, 1.0/120.0],
                              [1.0/120.0, 1.0/120.0, 1.0/120.0, 1.0/60.0]])
    },
}

# Local gradient templates for stiffness computation.
# These match the localgrad arrays in localStiffness.py.
_stiffness_localgrad = {
    'Edges': np.array([[-1.0, 1.0]]),                                         # (1, 2)
    'Trias': np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]),                # (2, 3)
    'Tetras': np.array([[-1.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 1.0]]),                              # (3, 4)
}

# Number of contravariant basis vectors used for stiffness per element type
_stiffness_nbasis = {
    'Edges': 1,
    'Trias': 2,
    'Tetras': 3,
}


def _batch_local_mass_vectorized(elemtype, measures):
    """Vectorized local mass matrix computation.
    Input:
        elemtype: element type string
        measures: (nElems, 1) array of element measures
    Output:
        (nElems, nV, nV) array of local mass matrices
    """
    tpl = _mass_templates[elemtype]
    return tpl['scale'] * measures[:, :, np.newaxis] * tpl['template'][np.newaxis, :, :]


def _batch_local_stiffness_vectorized(elemtype, contrabasis, measures, Sigma_batch):
    """Vectorized local stiffness matrix computation.
    Input:
        elemtype:     element type string
        contrabasis:  dict with 'v1', 'v2', 'v3' arrays of shape (nElems, 3)
        measures:     (nElems, 1) array of element measures
        Sigma_batch:  (nElems, 3, 3) array of diffusion tensors
    Output:
        (nElems, nV, nV) array of local stiffness matrices
    """
    nbas     = _stiffness_nbasis[elemtype]
    localgrad = _stiffness_localgrad[elemtype]     # (nbas, nV)
    # Build contravariant basis matrix: (nElems, 3, nbas)
    basis_keys = ['v1', 'v2', 'v3'][:nbas]
    contra_bas = np.stack([contrabasis[k] for k in basis_keys], axis=2)  # (nElems, 3, nbas)
    # grad = contra_bas @ localgrad  → (nElems, 3, nV)
    grad = np.einsum('eij,jk->eik', contra_bas, localgrad)
    # flux = Sigma_batch @ grad  → (nElems, 3, nV)
    flux = np.einsum('eij,ejk->eik', Sigma_batch, grad)
    # lstiff = measures * grad^T @ flux → (nElems, nV, nV)
    lstiff = measures[:, :, np.newaxis] * np.einsum('eji,ejk->eik', grad, flux)
    return lstiff


# Vectorized Sigma (diffusion tensor) pre-computation for all elements.
# Bypasses the per-element execute_ud_func call when material properties follow
# the standard region-based pattern with sigma_l, sigma_t and fibre directions:
#   Sigma = sigma_t * I + (sigma_l - sigma_t) * fib ⊗ fib
# Falls back to per-element computation when this pattern does not apply.
def _try_batch_sigma(matr_name, elemtype, Elements, domain, matprops):
    """Try to vectorize Sigma computation. Returns (Sigma_batch, success).
    Input:
        matr_name:  the name of the matrix (used to find the ud_function)
        elemtype:   element type string
        Elements:   numpy array of element connectivity (nElems x (nV+1))
        domain:     the domain object
        matprops:   MaterialProperties object
    Output:
        Sigma_batch: (nElems, 3, 3) array of diffusion tensors, or None
        success:     True if vectorized, False if fallback needed
    """
    nElems = Elements.shape[0]
    # Check if the standard sigma_l, sigma_t properties exist
    prop_names = matprops.element_property_names()
    if prop_names is None or 'sigma_l' not in prop_names or 'sigma_t' not in prop_names:
        return None, False
    # Check if fibre data is available
    fibs = domain.Fibres()
    if fibs is None:
        return None, False
    # Check if properties are region-based (vectorizable)
    if matprops.element_property_type('sigma_l') != 'region':
        return None, False
    if matprops.element_property_type('sigma_t') != 'region':
        return None, False

    # Vectorized: gather region IDs and fibre directions for all elements
    regionIDs = Elements[:, -1]                                # (nElems,)
    fib_batch = fibs[:nElems, :]                               # (nElems, 3)
    # Build sigma_l and sigma_t arrays from region mapping
    sigma_l_map = matprops._element_properties['sigma_l']['idmap']
    sigma_t_map = matprops._element_properties['sigma_t']['idmap']
    sigma_l_arr = np.array([sigma_l_map[int(r)] for r in regionIDs], dtype=np.float64)  # (nElems,)
    sigma_t_arr = np.array([sigma_t_map[int(r)] for r in regionIDs], dtype=np.float64)  # (nElems,)
    # Sigma = sigma_t * I + (sigma_l - sigma_t) * fib ⊗ fib
    eye3 = np.eye(3, dtype=np.float64)
    diff_sigma = sigma_l_arr - sigma_t_arr                     # (nElems,)
    # fib_outer = fib ⊗ fib: (nElems, 3, 3)
    fib_outer = fib_batch[:, :, np.newaxis] * fib_batch[:, np.newaxis, :]
    Sigma_batch = (sigma_t_arr[:, np.newaxis, np.newaxis] * eye3[np.newaxis, :, :] +
                   diff_sigma[:, np.newaxis, np.newaxis] * fib_outer)
    return Sigma_batch, True


# Batched local matrix computation for all elements of one type.
# Uses vectorized numpy operations for known element types (Edges, Trias, Tetras)
# with the standard localMass/localStiffness functions.
# Falls back to per-element computation for unknown types or custom local functions,
# allowing future support for non-linear elements.
def _batch_local_matrices(elemtype, Elements, domain, local_matrices_dict, matprops):
    """Compute all local matrices for all elements of a given type.
    Input:
        elemtype:            the element type string (e.g. 'Trias', 'Tetras', 'Edges')
        Elements:            numpy array of element connectivity (nElems x (nV+1))
        domain:              the domain object
        local_matrices_dict: dict of {name: local_matrix_function}
        matprops:            MaterialProperties object
    Output:
        batch_lmat: dict of {name: numpy array of shape (nElems, nV, nV)}
        Elems_noids: numpy array of shape (nElems, nV) with node IDs (no region ID)
    """
    nElems = Elements.shape[0]
    nV     = Elements.shape[1] - 1   # last column is region ID
    Elems_noids = Elements[:, :-1]   # (nElems, nV)

    batch_lmat = {}
    remaining  = dict(local_matrices_dict)

    # Try vectorized path for known element types and standard local functions.
    if elemtype in _batch_contrabasis_dispatch:
        contrabasis = _batch_contrabasis_dispatch[elemtype](domain.Pts(), Elems_noids)
        measures    = contrabasis['meas']  # (nElems, 1) or (nElems,)
        if measures.ndim == 1:
            measures = measures[:, np.newaxis]

        # Vectorized mass matrix
        for matr_name in list(remaining.keys()):
            if remaining[matr_name] is localMass and elemtype in _mass_templates:
                batch_lmat[matr_name] = _batch_local_mass_vectorized(elemtype, measures)
                del remaining[matr_name]

        # Vectorized stiffness matrix
        for matr_name in list(remaining.keys()):
            if remaining[matr_name] is localStiffness and elemtype in _stiffness_localgrad:
                # Try fully vectorized Sigma computation first.
                # Falls back to per-element loop if properties are not region-based.
                Sigma_batch, sigma_ok = _try_batch_sigma(
                    matr_name, elemtype, Elements, domain, matprops)
                if not sigma_ok:
                    # Pre-compute all Sigma tensors per-element (fallback)
                    Sigma_batch = np.zeros((nElems, 3, 3), dtype=np.float64)
                    for iElem in range(nElems):
                        local_props = matprops.execute_ud_func(matr_name, elemtype, iElem, domain, matprops)
                        if local_props is None:
                            Sigma_batch[iElem, :, :] = np.identity(3)
                        else:
                            Sigma_batch[iElem, :, :] = local_props
                batch_lmat[matr_name] = _batch_local_stiffness_vectorized(
                    elemtype, contrabasis, measures, Sigma_batch)
                del remaining[matr_name]

    # Fallback: per-element computation for remaining (custom) local matrix functions.
    # This path supports future non-linear elements and user-defined local matrices.
    if remaining:
        for matr_name in remaining.keys():
            batch_lmat[matr_name] = np.zeros(shape=(nElems, nV, nV), dtype=np.float64)
        for iElem in range(nElems):
            elemData = domain.element_contravariant_basis(elemtype, iElem)
            for matr_name, lmateval in remaining.items():
                local_props = matprops.execute_ud_func(matr_name, elemtype, iElem, domain, matprops)
                batch_lmat[matr_name][iElem, :, :] = lmateval(elemtype, elemData, local_props)

    return batch_lmat, Elems_noids


def assemble_vectmat_dict(local_matrices_dict,matrix_pattern,domain,matprops,connectivity: dict = None) -> dict[np.ndarray]:
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
    t0 = time()
    for elemtype, Elements in domain.Elems().items():
        batch_lmat, Elems_noids = _batch_local_matrices(
            elemtype, Elements, domain, local_matrices_dict, matprops)
        nElems = Elems_noids.shape[0]
        nV     = Elems_noids.shape[1]
        scatter_indices = np.zeros(shape=(nV, nV, nElems), dtype=np.int64)
        for iEntry in range(nV):
            irows = Elems_noids[:, iEntry]     # (nElems,) global row indices
            # Vectorized row start offsets for all elements
            row_starts = k0[irows]             # (nElems,)
            for jEntry in range(nV):
                jcols = Elems_noids[:, jEntry]  # (nElems,) global col indices
                # Vectorized index lookup using searchsorted.
                # For each element, find the position of jcol in the sorted
                # connectivity of irow, offset by the row start in the COO array.
                for e in range(nElems):
                    row_conn = connectivity[irows[e]]
                    loc = np.searchsorted(row_conn, jcols[e])
                    scatter_indices[iEntry, jEntry, e] = row_starts[e] + loc

        for matr_name in local_matrices_dict.keys():
            lmat_batch = batch_lmat[matr_name]   # (nElems, nV, nV)
            for iEntry in range(nV):
                for jEntry in range(nV):
                    idx = scatter_indices[iEntry, jEntry, :]   # (nElems,)
                    vals = lmat_batch[:, iEntry, jEntry]        # (nElems,)
                    np.add.at(VM[matr_name], idx, vals)

    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)
    return(VM)


####################################################################################
#############                                                          #############
#############            functions that REQUIRE TensorFlow             #############
#############                                                          #############
####################################################################################
import tensorflow as tf
# global matrices are now returned as CSRSparseMatrix wrappers
# (cuSPARSE-backed). Per-iteration SpMV inside the solvers goes through
# tf.raw_ops.SparseMatrixMatMul, which is ~1.6x/2.2x faster than
# tf.sparse.sparse_dense_matmul on the coarse/fine meshes in
# Tests/DEVTESTS/ConjugateGradients. csr_axpby is the CSR-native replacement
# for `tf.sparse.add(a, tf.sparse.map_values(tf.multiply, b, scalar))`.
from tensorflow.python.ops.linalg.sparse.sparse_csr_matrix_ops import CSRSparseMatrix


def _wrap_sparse_tensor_as_csr(sp: tf.sparse.SparseTensor) -> CSRSparseMatrix:
    """Reorder (required: unordered COO produces a broken CSR on GPU) and wrap
    a SparseTensor as a CSRSparseMatrix whose handle_data is set so the
    variant tensor survives @tf.function boundaries."""
    sp = tf.sparse.reorder(sp)
    return CSRSparseMatrix(sp)


def csr_axpby(A: CSRSparseMatrix, alpha, B: CSRSparseMatrix = None, beta=None) -> CSRSparseMatrix:
    """Compute alpha*A + beta*B (or alpha*A if B is None) as a CSRSparseMatrix.
    Replaces `tf.sparse.add(a, tf.sparse.map_values(tf.multiply, b, s))` for
    CSR matrices.
    """
    dtype = A.dtype
    a_alpha = tf.constant(alpha, dtype=dtype)
    if B is None:
        # alpha*A via SparseMatrixAdd with a zero matrix is overkill; instead
        # rebuild via SparseTensor scaling (one-time cost at assembly).
        sp = A.to_sparse_tensor()
        sp = tf.sparse.SparseTensor(sp.indices, sp.values * a_alpha, sp.dense_shape)
        return _wrap_sparse_tensor_as_csr(sp)
    b_beta  = tf.constant(beta,  dtype=dtype)
    result_variant = tf.raw_ops.SparseMatrixAdd(
        a=A._matrix, b=B._matrix, alpha=a_alpha, beta=b_beta)
    return A._from_matrix(result_variant, handle_data=A._matrix._handle_data)


def csr_to_sparse_tensor(A: CSRSparseMatrix) -> tf.sparse.SparseTensor:
    """Extract COO indices/values/shape from a CSRSparseMatrix (for callers
    such as JacobiPrecond.build_preconditioner that need the COO form)."""
    return A.to_sparse_tensor()

def assemble_mass_matrix(matrix_pattern : dict,domain,connectivity: dict = None, renumbering: dict = None) -> tf.sparse.SparseTensor:
    """ function assemble_mass_matrix(matrix_pattern,domain,connectivity=None)
    computes the sparse mass matrix using the domain connectivity and the matrix pattern.
    It returs a TensorFlow sparse tensor.
    Input:
        matrix_pattern: the sparsity pattern of the matrix
        domain:         the domain object
        connectivity:   the domain connectivity (if None, it is computed and kept in memory)
        renumbering:    the node renumbering to reduce breadwidth (default = None)
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
    from gpuSolve.entities.materialproperties import MaterialProperties
    dummy_matprops = MaterialProperties()
    dummy_matprops.add_ud_function('mass', lambda *args: None)
    local_matrices_dict = {'mass': localMass}
    for elemtype, Elements in domain.Elems().items():
        batch_lmat, Elems_noids = _batch_local_matrices(
            elemtype, Elements, domain, local_matrices_dict, dummy_matprops)
        nElems = Elems_noids.shape[0]
        nV     = Elems_noids.shape[1]
        lmat_batch = batch_lmat['mass']   # (nElems, nV, nV)
        for iEntry in range(nV):
            irows = Elems_noids[:, iEntry]
            row_starts = k0[irows]
            for jEntry in range(nV):
                jcols = Elems_noids[:, jEntry]
                idx = np.empty(nElems, dtype=np.int64)
                for e in range(nElems):
                    row_conn = connectivity[irows[e]]
                    loc = np.searchsorted(row_conn, jcols[e])
                    idx[e] = row_starts[e] + loc
                np.add.at(VM, idx, lmat_batch[:, iEntry, jEntry])

    if renumbering is None:
        indices   = np.hstack([I[:,np.newaxis], J[:,np.newaxis]])
    else:
        iperm   = renumbering['iperm']
        I_rnmb  = iperm[I].astype(I.dtype)
        J_rnmb  = iperm[J].astype(J.dtype)
        indices = np.hstack([I_rnmb[:,np.newaxis], J_rnmb[:,np.newaxis]])
    MASS      = tf.sparse.SparseTensor(indices=indices, values=VM.astype(np.float32), dense_shape=[npt, npt])
    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)
    return _wrap_sparse_tensor_as_csr(MASS)


def assemble_stiffness_matrix(matrix_pattern: dict ,domain,matprops,stif_pname: str = 'Sigma',connectivity : dict =None, renumbering : dict = None) -> tf.sparse.SparseTensor:
    """ function assemble_stiffness_matrix(matrix_pattern,domain,matprops,connectivity=None)
    computes the sparse stiffness matrix using the domain connectivity and the matrix pattern.
    It returs a TensorFlow sparse tensor.
    Input:
        matrix_pattern: the sparsity pattern of the matrix
        domain:         the domain object
        matprops:       a MaterialProperties object that implements functions to provide local properties
        stif_pname:     the name of the function that evaluates the matertial properties
        connectivity:   the domain connectivity (if None, it is computed and kept in memory)
        renumbering:    the node renumbering to reduce breadwidth (default = None)
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
    local_matrices_dict = {stif_pname: localStiffness}
    for elemtype, Elements in domain.Elems().items():
        batch_lmat, Elems_noids = _batch_local_matrices(
            elemtype, Elements, domain, local_matrices_dict, matprops)
        nElems = Elems_noids.shape[0]
        nV     = Elems_noids.shape[1]
        lmat_batch = batch_lmat[stif_pname]   # (nElems, nV, nV)
        for iEntry in range(nV):
            irows = Elems_noids[:, iEntry]
            row_starts = k0[irows]
            for jEntry in range(nV):
                jcols = Elems_noids[:, jEntry]
                idx = np.empty(nElems, dtype=np.int64)
                for e in range(nElems):
                    row_conn = connectivity[irows[e]]
                    loc = np.searchsorted(row_conn, jcols[e])
                    idx[e] = row_starts[e] + loc
                np.add.at(VM, idx, lmat_batch[:, iEntry, jEntry])

    if renumbering is None:
        indices   = np.hstack([I[:,np.newaxis], J[:,np.newaxis]])
    else:
        iperm   = renumbering['iperm']
        I_rnmb  = iperm[I].astype(I.dtype)
        J_rnmb  = iperm[J].astype(J.dtype)
        indices = np.hstack([I_rnmb[:,np.newaxis], J_rnmb[:,np.newaxis]])
    STIFFNESS = tf.sparse.SparseTensor(indices=indices, values=VM.astype(np.float32), dense_shape=[npt, npt])
    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)
    return _wrap_sparse_tensor_as_csr(STIFFNESS)



def assemble_matrices_dict(local_matrices_dict : dict ,matrix_pattern: dict,domain,matprops,connectivity : dict = None, renumbering : dict = None) -> dict[tf.sparse.SparseTensor]:
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
        renumbering:         the node renumbering to reduce breadwidth (default = None)
    Output:
        MATRICES:  a python dict with the TensorFlow sparse tensors storing the matrices.
    """
    npt = domain.Pts().shape[0]
    print('Assembly the following sparse matrices:'.format(len(local_matrices_dict.keys())),flush=True)
    for matr_name in local_matrices_dict.keys():
        print('{}'.format(matr_name),flush=True)

    t0 = time()
    all_rows = []
    all_cols = []
    all_vals = {name: [] for name in local_matrices_dict.keys()}

    for elemtype, Elements in domain.Elems().items():
        batch_lmat, Elems_noids = _batch_local_matrices(
            elemtype, Elements, domain, local_matrices_dict, matprops)
        nV = Elems_noids.shape[1]

        for iEntry in range(nV):
            for jEntry in range(nV):
                all_rows.append(Elems_noids[:, iEntry])        # (nElems,)
                all_cols.append(Elems_noids[:, jEntry])        # (nElems,)
                for matr_name in local_matrices_dict.keys():
                    all_vals[matr_name].append(
                        batch_lmat[matr_name][:, iEntry, jEntry])  # (nElems,)

    rows = np.concatenate(all_rows).astype(np.int64)
    cols = np.concatenate(all_cols).astype(np.int64)

    if renumbering is not None:
        iperm = renumbering['iperm']
        rows  = iperm[rows].astype(np.int64)
        cols  = iperm[cols].astype(np.int64)

    dense_shape = [npt, npt]
    linearized  = tf.constant(rows * npt + cols, dtype=tf.int64)
    y, idx      = tf.unique(linearized)
    unique_rows = tf.cast(y // npt, dtype=tf.int64)
    unique_cols = tf.cast(y % npt, dtype=tf.int64)
    indices     = tf.stack([unique_rows, unique_cols], axis=1)

    MATRICES = {}
    for matr_name in local_matrices_dict.keys():
        vals   = tf.constant(np.concatenate(all_vals[matr_name]), dtype=tf.float32)
        values = tf.math.unsorted_segment_sum(vals, idx, tf.shape(y)[0])
        sp = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape)
        MATRICES[matr_name] = _wrap_sparse_tensor_as_csr(sp)

    elapsed = time() - t0
    print('done in {:3.2f} s'.format(elapsed),flush=True)
    return(MATRICES)

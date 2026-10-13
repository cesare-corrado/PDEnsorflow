#!/usr/bin/env python
"""
    Tier-1 unit tests for the matrix cache
    (gpuSolve.matrices.matrixcache.MatrixCache and the two CSR-array functions
    of gpuSolve.matrices.globalMatrices that feed it).

    The cache stores the assembled mass and stiffness matrices so that several
    runs on the same mesh with the same conductivities assemble them once. Two
    properties have to hold, and they pull in opposite directions:

      * what comes back must be what was assembled, to the BIT. The assembly
        sums the element entries on the host precisely so that it is
        reproducible (the device segment sum was not), so the round trip is
        compared exactly and not within a tolerance. The entries are also
        compared against assemble_matrices_dict(), the production path that
        does not use the cache: the cache writes them in CSR order, which for a
        renumbered mesh means an explicit sort that has to reproduce what
        tf.sparse.reorder does on the device.
      * a file that does not describe this problem must NEVER be used. A cache
        that silently supplied the wrong conductivities would run physics
        nobody asked for and report nothing, which is worse than being slow. So
        every way a file can fail to match is checked here: each field of the
        key, the format version, a mesh whose content changed under the same
        name, and a file that is not readable at all. All of them must lead to
        a reassembly, not to an exception and not to a stale matrix.

    Corrupt-but-accepted files are the one case that must raise instead: if the
    key says the file is for this problem, a column array that does not fit the
    stated shape is a damaged cache, and solving on a mangled operator would be
    silent nonsense.

    CPU-only and fast: a 1D edge mesh of a few elements, as in test_matrices.py.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle
import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.globalMatrices import compute_reverse_cuthill_mckee_indexing
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.matrices.globalMatrices import assemble_matrices_csr_arrays
from gpuSolve.matrices.globalMatrices import csr_matrices_from_arrays
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.matrixcache import MatrixCache
from gpuSolve.matrices.matrixcache import MATRIX_CACHE_FORMAT_VERSION

SIGMA = 1.0                                      # isotropic conductivity
NELEM = 8                                        # elements of the 1D test mesh
LENGTH = 2.0                                     # length of the 1D test mesh
MATRIX_NAMES = ('mass', 'stiffness')


def _dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """Empty mass-property callback (mass is material-independent)."""
    return(None)


def _sigma_iso(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties) -> np.ndarray:
    """Isotropic diffusion tensor SIGMA*I for the stiffness matrix."""
    return(SIGMA * np.eye(3))


def _write_1d_mesh(fname: str, nelem: int = NELEM, length: float = LENGTH):
    """Write a uniform 1D edge mesh over [0, length] as a .pkl readable by
    Triangulation.readMesh."""
    npt         = nelem + 1
    Pts         = np.zeros(shape=(npt, 3), dtype=np.float64)
    Pts[:, 0]   = np.linspace(0.0, length, npt)
    Edges       = np.zeros(shape=(nelem, 3), dtype=np.int32)
    Edges[:, 0] = np.arange(0, nelem)
    Edges[:, 1] = np.arange(1, nelem + 1)
    Edges[:, 2] = 1
    mesh = {'Pts': Pts, 'Elems': {'Edges': Edges}, 'Fibres': None}
    with open(fname, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)


def _setup(mesh_file: str, renumber: bool):
    """Read the mesh and build everything the assembly needs. Returns the
    arguments of the two assembly functions plus the renumbering."""
    domain = Triangulation()
    domain.readMesh(mesh_file)
    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    renumbering  = compute_reverse_cuthill_mckee_indexing(pattern) if renumber else None
    materials    = MaterialProperties()
    materials.add_ud_function('mass', _dfmass)
    materials.add_ud_function('stiffness', _sigma_iso)
    lmatr        = {'mass': localMass, 'stiffness': localStiffness}
    return(lmatr, pattern, domain, materials, connectivity, renumbering)


def _assemble_arrays(mesh_file: str, renumber: bool) -> tuple:
    """Assemble the host CSR arrays and the renumbering for this mesh."""
    lmatr, pattern, domain, materials, connectivity, renumbering = _setup(mesh_file, renumber)
    arrays = assemble_matrices_csr_arrays(lmatr, pattern, domain, materials, connectivity,
                                          renumbering=renumbering)
    return(arrays, renumbering)


def _cache(fname: str, key: dict = None) -> MatrixCache:
    """A MatrixCache on fname, with a default key that the tests perturb."""
    if key is None:
        key = {'mesh': 'line', 'sigma_l': {1: 1.0}, 'renumbering': 1}
    cache = MatrixCache({'fname': fname})
    cache.set_key(key)
    return(cache)


def _coo(matrix) -> tuple:
    """The (indices, values) of a device matrix, in CSR order."""
    st = matrix.to_sparse_tensor()
    return(st.indices.numpy(), st.values.numpy())


# ---- the round trip --------------------------------------------------------

@pytest.mark.parametrize('renumber', [True, False])
def test_roundtrip_is_bit_exact(tmp_path, renumber):
    """What load() returns must be bit for bit what save() was given, with and
    without the node renumbering."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, renumbering = _assemble_arrays(mesh_file, renumber)

    cache = _cache(str(tmp_path / 'mats.npz'))
    cache.save(arrays, renumbering=renumbering)
    stored = cache.load()

    assert stored is not None
    assert stored['npt'] == arrays['npt']
    assert stored['nnz'] == arrays['nnz']
    assert np.array_equal(stored['indptr'], arrays['indptr'])
    assert np.array_equal(stored['col'], arrays['col'])
    for name in MATRIX_NAMES:
        # exact, not approximate: the host assembly is deterministic by design
        assert np.array_equal(stored['values'][name], arrays['values'][name])
    if renumbering is None:
        assert stored['renumbering'] is None
    else:
        assert np.array_equal(stored['renumbering']['perm'], renumbering['perm'])
        assert np.array_equal(stored['renumbering']['iperm'], renumbering['iperm'])


@pytest.mark.parametrize('renumber', [True, False])
def test_reloaded_matrices_equal_the_production_assembly(tmp_path, renumber):
    """The matrices rebuilt from a cache file must have the same indices and the
    same values as assemble_matrices_dict() gives for the same problem.

    This is what makes the cache safe to substitute for the assembly. With the
    renumbering on it also checks the sort that assemble_matrices_csr_arrays()
    does on the host: the pattern is sorted in the original numbering and then
    permuted, so it is no longer sorted, and the stored order has to be the one
    tf.sparse.reorder() produces on the device.
    """
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)

    lmatr, pattern, domain, materials, connectivity, renumbering = _setup(mesh_file, renumber)
    reference = assemble_matrices_dict(lmatr, pattern, domain, materials, connectivity,
                                       renumbering=renumbering)
    arrays, _ = _assemble_arrays(mesh_file, renumber)
    cache = _cache(str(tmp_path / 'mats.npz'))
    cache.save(arrays, renumbering=renumbering)
    rebuilt = csr_matrices_from_arrays(cache.load())

    for name in MATRIX_NAMES:
        iref, vref = _coo(reference[name])
        inew, vnew = _coo(rebuilt[name])
        assert np.array_equal(iref, inew)
        assert np.array_equal(vref, vnew)


def test_stored_arrays_are_in_csr_order(tmp_path):
    """The stored entries must be sorted by (row, column) and the row pointer
    must agree with them, which is what lets the load path skip the device
    reordering."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, _ = _assemble_arrays(mesh_file, True)

    npt    = arrays['npt']
    indptr = arrays['indptr']
    col    = arrays['col']
    assert indptr.shape == (npt + 1,)
    assert int(indptr[0]) == 0
    assert int(indptr[-1]) == arrays['nnz']
    for row in range(npt):
        cols_of_row = col[indptr[row]:indptr[row + 1]]
        assert np.all(np.diff(cols_of_row) > 0)


# ---- refusing a file that does not match -----------------------------------

def test_no_file_named_is_inert(tmp_path):
    """A cache with no file name is inactive and loads nothing: a solver that
    was not asked for a cache must not look for one."""
    cache = MatrixCache()
    assert not cache.active()
    assert cache.load() is None


def test_missing_file_returns_none(tmp_path):
    """A named file that does not exist yet is a miss, not an error: it is the
    state of the first run of a study."""
    assert _cache(str(tmp_path / 'absent.npz')).load() is None


@pytest.mark.parametrize('perturbed', [
    {'mesh': 'other'},                           # another mesh
    {'sigma_l': {1: 2.0}},                       # a conductivity changed
    {'renumbering': 0},                          # the numbering changed
    {'gpuSolve_version': '0.0.0'},               # another library version
])
def test_key_mismatch_refuses_the_file(tmp_path, perturbed):
    """Any difference in the key must lead to a reassembly. The key is compared
    whole, so each field is perturbed on its own here."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, renumbering = _assemble_arrays(mesh_file, True)

    fname = str(tmp_path / 'mats.npz')
    _cache(fname).save(arrays, renumbering=renumbering)

    key = {'mesh': 'line', 'sigma_l': {1: 1.0}, 'renumbering': 1}
    key.update(perturbed)
    assert _cache(fname, key).load() is None


def test_key_order_does_not_matter(tmp_path):
    """Two keys that hold the same values must match whatever order their dicts
    were built in: the key is a description of the problem, not of the walk
    that produced it."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, renumbering = _assemble_arrays(mesh_file, True)

    fname = str(tmp_path / 'mats.npz')
    _cache(fname, {'a': 1, 'b': {'y': 2, 'x': 3}}).save(arrays, renumbering=renumbering)
    assert _cache(fname, {'b': {'x': 3, 'y': 2}, 'a': 1}).load() is not None


def test_file_content_fingerprint_sees_an_edit_in_place(tmp_path):
    """The fingerprint must follow the CONTENT of the mesh file: a mesh
    rewritten in place keeps its name, its path and (here) its size, and would
    otherwise be read from the cache as if it were the old one."""
    fname = str(tmp_path / 'mesh.bin')
    with open(fname, 'wb') as fout:
        fout.write(b'A' * 4096)
    cache = _cache(str(tmp_path / 'mats.npz'))
    before = cache.file_fingerprint(fname)
    with open(fname, 'wb') as fout:
        fout.write(b'A' * 4095 + b'B')
    after = cache.file_fingerprint(fname)

    assert before['bytes'] == after['bytes'] == 4096
    assert before['sha256'] != after['sha256']


def test_unreadable_file_returns_none(tmp_path):
    """A file that is not an npz at all (a truncated write, something else of
    the same name) is a stale cache, not a reason to stop the run."""
    fname = str(tmp_path / 'mats.npz')
    with open(fname, 'wb') as fout:
        fout.write(b'not an npz archive')
    assert _cache(fname).load() is None


def test_other_format_version_returns_none(tmp_path):
    """A file written by another layout of the cache must be refused: its
    arrays may mean something else."""
    fname = str(tmp_path / 'mats.npz')
    cache = _cache(fname)
    np.savez(fname,
             format_version=np.asarray(MATRIX_CACHE_FORMAT_VERSION + 1, dtype=np.int64),
             key=np.asarray(''), npt=np.asarray(1, dtype=np.int64),
             nnz=np.asarray(1, dtype=np.int64))
    assert cache.load() is None


def test_save_is_atomic_and_leaves_no_temporary(tmp_path):
    """save() must leave the directory holding the cache file and nothing else:
    the file is written beside its name and renamed, so that a second run of the
    same study never reads a half-written one."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, renumbering = _assemble_arrays(mesh_file, True)
    outdir = tmp_path / 'cachedir'
    outdir.mkdir()
    fname = str(outdir / 'mats.npz')
    _cache(fname).save(arrays, renumbering=renumbering)
    assert [path.name for path in outdir.iterdir()] == ['mats.npz']


def test_save_overwrites_a_stale_file(tmp_path):
    """A run whose key does not match the file must leave the file describing
    ITS problem, so the runs after it hit instead of reassembling in turn."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, renumbering = _assemble_arrays(mesh_file, True)

    fname = str(tmp_path / 'mats.npz')
    stale = _cache(fname, {'mesh': 'line', 'sigma_l': {1: 9.0}, 'renumbering': 1})
    stale.save(arrays, renumbering=renumbering)
    fresh = _cache(fname)
    assert fresh.load() is None
    fresh.save(arrays, renumbering=renumbering)
    assert fresh.load() is not None
    assert stale.load() is None


# ---- refusing to continue with a damaged file ------------------------------

@pytest.mark.parametrize('damage', ['indptr', 'col', 'values', 'perm', 'indptr_end'])
def test_inconsistent_shapes_raise(tmp_path, damage):
    """A file the key accepted but whose arrays do not describe one matrix of
    npt x npt with nnz entries must raise. Unlike a stale file this cannot be
    answered by reassembling: it means the cache is damaged, and a matrix built
    from arrays that do not fit would be wrong without being invalid."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    arrays, renumbering = _assemble_arrays(mesh_file, True)
    fname = str(tmp_path / 'mats.npz')
    cache = _cache(fname)
    cache.save(arrays, renumbering=renumbering)

    with np.load(fname, allow_pickle=False) as stored:
        content = {name: stored[name] for name in stored.files}
    if damage == 'indptr':
        content['indptr'] = content['indptr'][:-1]
    elif damage == 'col':
        content['col'] = content['col'][:-1]
    elif damage == 'values':
        content['values_stiffness'] = content['values_stiffness'][:-1]
    elif damage == 'perm':
        content['perm'] = content['perm'][:-1]
    elif damage == 'indptr_end':
        content['nnz'] = np.asarray(int(content['nnz']) + 1, dtype=np.int64)
        content['col'] = np.concatenate([content['col'], content['col'][-1:]])
        content['values_mass'] = np.concatenate([content['values_mass'], content['values_mass'][-1:]])
        content['values_stiffness'] = np.concatenate([content['values_stiffness'],
                                                      content['values_stiffness'][-1:]])
    np.savez(fname, **content)

    with pytest.raises(ValueError):
        cache.load()


# ---- the solver path -------------------------------------------------------

def test_solver_refuses_a_cache_without_a_key(tmp_path):
    """A file named through the configuration but never given a key must stop
    the run: a key of None would compare equal to the key of any other run that
    also set none, on any mesh."""
    from gpuSolve.physics.heatSolver import HeatSolver

    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    solver = HeatSolver({'mesh_file_name': mesh_file, 'dt': 0.1, 'Tend': 1.0,
                         'matrix_cache_file': str(tmp_path / 'mats.npz')})
    solver.add_element_material_property('sigma_l', 'region', {1: SIGMA})
    solver.add_material_function('mass', _dfmass)
    solver.add_material_function('stiffness', _sigma_iso)
    with pytest.raises(ValueError):
        solver.assemble_matrices()


def test_solver_refuses_a_cache_of_another_mesh(tmp_path):
    """A file whose key the caller made too narrow to see the mesh must still
    not be read onto a mesh of another size: the node count is checked where
    the mesh is."""
    from gpuSolve.physics.heatSolver import HeatSolver

    small = str(tmp_path / 'small.pkl')
    large = str(tmp_path / 'large.pkl')
    _write_1d_mesh(small, nelem=NELEM)
    _write_1d_mesh(large, nelem=NELEM + 4)
    fname = str(tmp_path / 'mats.npz')
    # a key that says nothing about the mesh, as a careless caller might write
    key   = {'sigma_l': {1: SIGMA}}

    for mesh_file, expect_raise in ((small, False), (large, True)):
        solver = HeatSolver({'mesh_file_name': mesh_file, 'dt': 0.1, 'Tend': 1.0})
        solver.add_element_material_property('sigma_l', 'region', {1: SIGMA})
        solver.add_element_material_property('sigma_t', 'region', {1: SIGMA})
        solver.add_material_function('mass', _dfmass)
        solver.add_material_function('stiffness', _sigma_iso)
        solver.set_matrix_cache(fname, key)
        if expect_raise:
            with pytest.raises(ValueError):
                solver.assemble_matrices()
        else:
            solver.assemble_matrices()


def test_heat_solver_uses_the_cache(tmp_path, capsys):
    """Two HeatSolver runs, the first assembling and writing the cache and the
    second reading it, must take the same step from the same initial condition,
    and the second must not assemble at all.

    The step is compared rather than the matrix itself because that is what the
    cache has to preserve: it exercises the system matrix, the right-hand side
    matrix, the preconditioner built from them and the node permutation, all of
    which the loading run rebuilds from the stored arrays.
    """
    from gpuSolve.physics.heatSolver import HeatSolver

    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file)
    fname  = str(tmp_path / 'mats.npz')
    key    = {'mesh': mesh_file, 'sigma_l': {1: SIGMA}}
    config = {'mesh_file_name': mesh_file, 'dt': 0.1, 'Tend': 1.0}
    # a non-uniform initial condition, so that the diffusion step moves it
    U0 = np.zeros(shape=(NELEM + 1, 1), dtype=np.float32)
    U0[:(NELEM // 2), 0] = 1.0

    solvers = []
    reports = []
    for _run in range(2):
        solver = HeatSolver(config)
        solver.add_element_material_property('sigma_l', 'region', {1: SIGMA})
        solver.add_element_material_property('sigma_t', 'region', {1: SIGMA})
        solver.add_material_function('mass', _dfmass)
        solver.add_material_function('stiffness', _sigma_iso)
        solver.set_matrix_cache(fname, key)
        capsys.readouterr()
        solver.assemble_matrices()
        reports.append(capsys.readouterr().out)
        solver.set_initial_condition(U0)
        solver.finalize_for_run()
        solver.step(solver.dt())
        solvers.append(solver)

    assert os.path.isfile(fname)
    np.testing.assert_array_equal(solvers[0].U().numpy(), solvers[1].U().numpy())
    # the first run assembled and wrote the file, the second read it and did
    # neither the connectivity nor the assembly
    assert 'Assembly the following sparse matrices' in reports[0]
    assert 'written' in reports[0]
    assert 'loaded' in reports[1]
    assert 'Assembly the following sparse matrices' not in reports[1]
    assert 'mesh connectivity' not in reports[1]
    # the permutation the whole run needs came back with the matrices
    assert solvers[1].renumbering() is not None

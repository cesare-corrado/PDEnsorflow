#!/usr/bin/env python
"""
    Tier-1 tests of where the membrane surface-to-volume ratio beta enters the
    monodomain system.

    The equation is

        beta (Cm dV/dt + I_ion) = div(sigma grad V)

    and beta can be moved out of sight by dividing through by it, which is what
    this library did from the start: the solver was handed sigma/beta and the
    membrane side carried no beta at all. That is exact while beta is uniform
    and WRONG as soon as it is not, because

        div((sigma/beta) grad V) != (1/beta) div(sigma grad V)

    the difference being a spurious term in grad(beta). A beta that differs
    between regions therefore has to stay on the membrane side, as the
    coefficient of the mass integral. Dividing by a CONSTANT is still allowed,
    so the conductivity is divided by one reference value, beta_ref, and the
    mass integral carries beta/beta_ref. That keeps the system the same size as
    before (the absolute tolerance of the CG solver sees the same residuals)
    and leaves a uniform-beta problem bit for bit where it was.

    What is checked, and why each check is the one that matters:

      * with a UNIFORM beta, registering beta_ref changes nothing at all. The
        matrices must agree to the BIT with the ones assembled the old way,
        because that is every existing run of this library;
      * with a VARYING beta, the mass matrix is the beta-weighted one, element
        by element, against the closed form of a 1D cable. beta is piecewise
        constant per element, so the weighted integral is exact and the
        interface node correctly receives one contribution from each side with
        ITS OWN beta;
      * with a varying beta, the stiffness is divided by beta_ref and NOT by
        the element's beta, which is the half of the change that is easy to
        leave out and would then double-count;
      * the coefficient reaches the matrix through both assembly paths. The
        vectorised path never called the material function at all, so a
        coefficient it silently ignored would pass every test written against
        the per-element path;
      * the parameter-file front end wires it up: two ionic regions with
        different cellSurfVolRatio must produce the weighted mass matrix, with
        the reference taken from imp_region[0].

    CPU-only, no solve: everything here is assembly.

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
from gpuSolve.matrices.globalMatrices import compute_coo_pattern, assemble_matrices_dict
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import mass_beta_weight
from gpuSolve.physics import no_mass_property
from gpuSolve.physics import HeatSolver
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner


_NELEM = 6                                       # even, so the two tags share a node
_LEN   = 3.0                                     # the cable length, in the mesh unit
_SIGMA = 2.0                                     # the conductivity of both regions
_BETA  = {1: 0.14, 2: 0.35}                      # the two regions' beta


def _write_cable(fname: str, beta_tags: dict, fibres: bool = True):
    """A uniform 1D cable of _NELEM edges whose first half carries tag 1 and
    second half tag 2, so one node sits on the interface.
    fibres=True writes a fibre field along the cable, which is what sends the
    assembly down the VECTORISED path: without one it falls back to the
    per-element material functions, and a defect in the vectorised arithmetic
    would never be seen. Every mesh of the reference's format carries fibres
    (its .lon file), so the vectorised path is the production one.
    """
    npt         = _NELEM + 1
    Pts         = np.zeros(shape=(npt, 3), dtype=np.float64)
    Pts[:, 0]   = np.linspace(0.0, _LEN, npt)
    Edges       = np.zeros(shape=(_NELEM, 3), dtype=np.int32)
    Edges[:, 0] = np.arange(0, _NELEM)
    Edges[:, 1] = np.arange(1, _NELEM + 1)
    tags        = sorted(beta_tags.keys())
    Edges[:, 2] = np.where(np.arange(_NELEM) < _NELEM // 2, tags[0], tags[-1])
    Fibres      = None
    if fibres:
        Fibres      = np.zeros(shape=(_NELEM, 3), dtype=np.float64)
        Fibres[:, 0] = 1.0                       # along the cable
    mesh = {'Pts': Pts, 'Elems': {'Edges': Edges}, 'Fibres': Fibres}
    with open(fname, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return(np.asarray(Edges))


def _reference_matrices(edges: np.ndarray, weights: dict, sigma_over: float) -> tuple:
    """(M, K) of the cable, summed element by element from the closed forms of
    a linear edge: h/6 [[2,1],[1,2]] for the mass and sigma/h [[1,-1],[-1,1]]
    for the stiffness. `weights` is the coefficient of the mass integral per
    tag, and `sigma_over` the single value the conductivity is divided by.
    This is the loop the vectorised assembly replaced, written out per element
    so that the interface node is visibly the sum of its two neighbours, each
    with its own weight.
    """
    npt = int(edges[:, :-1].max()) + 1
    h   = _LEN / _NELEM
    M   = np.zeros(shape=(npt, npt), dtype=np.float64)
    K   = np.zeros(shape=(npt, npt), dtype=np.float64)
    for edge in edges:
        ii, jj, tag = int(edge[0]), int(edge[1]), int(edge[2])
        w    = weights[tag]
        kval = (_SIGMA / sigma_over) / h
        M[ii, ii] += w * 2.0 * h / 6.0
        M[jj, jj] += w * 2.0 * h / 6.0
        M[ii, jj] += w * 1.0 * h / 6.0
        M[jj, ii] += w * 1.0 * h / 6.0
        K[ii, ii] += kval
        K[jj, jj] += kval
        K[ii, jj] += -kval
        K[jj, ii] += -kval
    return((M, K))


def _assemble(mesh_file: str, beta_tags: dict, beta_ref: float = None) -> tuple:
    """(M, K) assembled through the production path, dense.
    beta_ref=None registers beta alone, which is the formulation every caller
    written before this change gets: beta divides the conductivity and the mass
    carries nothing.
    """
    domain = Triangulation()
    domain.readMesh(mesh_file)
    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    materials    = MaterialProperties()
    materials.add_element_property('sigma_l', 'region', {tag: _SIGMA for tag in beta_tags})
    materials.add_element_property('sigma_t', 'region', {tag: _SIGMA for tag in beta_tags})
    materials.add_element_property('beta', 'region', dict(beta_tags))
    if beta_ref is None:
        materials.add_ud_function('mass', no_mass_property)
    else:
        materials.add_element_property('beta_ref', 'uniform', beta_ref)
        materials.add_ud_function('mass', mass_beta_weight)
    materials.add_ud_function('stiffness', conductivity_tensor)
    matrices = assemble_matrices_dict({'mass': localMass, 'stiffness': localStiffness},
                                      pattern, domain, materials, connectivity)
    dense = []
    for name in ('mass', 'stiffness'):
        st = tf.sparse.reorder(matrices[name].to_sparse_tensor())
        dense.append(tf.sparse.to_dense(st).numpy())
    return((dense[0], dense[1]))


# ---- a uniform beta must not move -------------------------------------------
def test_a_uniform_beta_assembles_the_same_matrices(tmp_path):
    """The whole point of dividing by a reference rather than by beta itself:
    where beta is uniform the coefficient is exactly 1, the conductivity is
    divided by exactly what it used to be divided by, and both matrices are
    unchanged. Bit equality, not a tolerance: an existing run must not move.
    """
    mesh_file = str(tmp_path / 'uniform.pkl')
    _write_cable(mesh_file, {1: 0.14, 2: 0.14})
    legacy = _assemble(mesh_file, {1: 0.14, 2: 0.14})
    weighted = _assemble(mesh_file, {1: 0.14, 2: 0.14}, beta_ref=0.14)
    assert np.array_equal(legacy[0], weighted[0]), 'the mass matrix moved'
    assert np.array_equal(legacy[1], weighted[1]), 'the stiffness matrix moved'


def test_without_beta_ref_the_old_formulation_is_untouched(tmp_path):
    """A caller that registers beta alone still gets sigma/beta and a bare mass
    matrix, varying beta included. The new form is selected by beta_ref and by
    nothing else, so no existing script changes under anyone's feet.
    """
    mesh_file = str(tmp_path / 'legacy.pkl')
    edges     = _write_cable(mesh_file, _BETA)
    M, K      = _assemble(mesh_file, _BETA)
    Mref      = _reference_matrices(edges, {tag: 1.0 for tag in _BETA}, 1.0)[0]
    np.testing.assert_allclose(M, Mref, rtol=1.0e-6, atol=1.0e-9)
    # the stiffness still carries the per-element division: the interface node
    # is the sum of sigma/beta_1 and sigma/beta_2
    h = _LEN / _NELEM
    mid = _NELEM // 2
    assert K[mid, mid] == pytest.approx((_SIGMA / _BETA[1] + _SIGMA / _BETA[2]) / h)


# ---- a varying beta ---------------------------------------------------------
def test_the_mass_matrix_carries_the_beta_of_each_element(tmp_path):
    """Element by element against the closed form, with the weight of the tag
    that element carries. The interface node is the check that matters: it must
    be the sum of the two half-elements, each weighted by its own beta, which
    is what resolving beta onto the nodes first could not produce.
    """
    mesh_file  = str(tmp_path / 'varying.pkl')
    edges      = _write_cable(mesh_file, _BETA)
    beta_ref   = _BETA[1]
    M, _K      = _assemble(mesh_file, _BETA, beta_ref=beta_ref)
    weights    = {tag: value / beta_ref for tag, value in _BETA.items()}
    Mref, _Kref = _reference_matrices(edges, weights, beta_ref)
    np.testing.assert_allclose(M, Mref, rtol=1.0e-6, atol=1.0e-9)
    # written out for the interface node, independently of the loop above
    h   = _LEN / _NELEM
    mid = _NELEM // 2
    assert M[mid, mid] == pytest.approx((weights[1] + weights[2]) * 2.0 * h / 6.0)
    # and the matrix stays symmetric, which is what keeps ConjGrad applicable
    np.testing.assert_allclose(M, M.T, rtol=1.0e-12, atol=1.0e-14)


@pytest.mark.parametrize('fibres', [True, False], ids=['vectorised', 'per element'])
def test_the_stiffness_is_divided_by_the_reference_only(tmp_path, fibres):
    """The other half of the change, and it has to be checked on BOTH paths.
    With beta_ref registered the conductivity is divided by that single value
    everywhere, so both halves of the cable have the same stiffness entries
    although their beta differ. Dividing by the element's beta here as well
    would count beta twice, on both sides of the equation, and the run would be
    wrong by the ratio of the two betas in every region but the reference one.

    The parametrisation is the point: the vectorised path builds the tensor
    itself, without calling conductivity_tensor, and it is taken only when the
    mesh carries fibres. It did divide by the element's beta while the
    per-element function divided by the reference, so this test passed on a
    cable with no fibre field while the production path was wrong.
    """
    mesh_file   = str(tmp_path / 'stiff.pkl')
    edges       = _write_cable(mesh_file, _BETA, fibres=fibres)
    beta_ref    = _BETA[1]
    _M, K       = _assemble(mesh_file, _BETA, beta_ref=beta_ref)
    _Mref, Kref = _reference_matrices(edges, {tag: 1.0 for tag in _BETA}, beta_ref)
    np.testing.assert_allclose(K, Kref, rtol=1.0e-6, atol=1.0e-9)
    h = _LEN / _NELEM
    assert K[1, 1] == pytest.approx(2.0 * (_SIGMA / beta_ref) / h)
    assert K[_NELEM - 1, _NELEM - 1] == pytest.approx(2.0 * (_SIGMA / beta_ref) / h)


@pytest.mark.parametrize('fibres', [True, False], ids=['vectorised', 'per element'])
def test_a_uniform_beta_is_bit_identical_on_both_paths(tmp_path, fibres):
    """The inertness promise, on the vectorised path as well: registering
    beta_ref where beta is uniform must leave both matrices exactly as they
    were, whichever path assembles them."""
    mesh_file = str(tmp_path / 'uniform_paths.pkl')
    _write_cable(mesh_file, {1: 0.14, 2: 0.14}, fibres=fibres)
    legacy   = _assemble(mesh_file, {1: 0.14, 2: 0.14})
    weighted = _assemble(mesh_file, {1: 0.14, 2: 0.14}, beta_ref=0.14)
    assert np.array_equal(legacy[0], weighted[0]), 'the mass matrix moved'
    assert np.array_equal(legacy[1], weighted[1]), 'the stiffness matrix moved'


def test_the_vectorised_mass_follows_the_registered_function(tmp_path):
    """The vectorised mass path must take its instruction from the registered
    material function, not from the presence of the properties: a problem that
    registers beta and beta_ref but asks for the geometric mass must get the
    geometric mass. Deciding from the property names gave it a weighted one and
    the two paths then disagreed on the same input.
    """
    mesh_file = str(tmp_path / 'declined.pkl')
    edges     = _write_cable(mesh_file, _BETA)
    domain    = Triangulation()
    domain.readMesh(mesh_file)
    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    materials    = MaterialProperties()
    materials.add_element_property('beta', 'region', dict(_BETA))
    materials.add_element_property('beta_ref', 'uniform', _BETA[1])
    materials.add_ud_function('mass', no_mass_property)
    matrices = assemble_matrices_dict({'mass': localMass}, pattern, domain,
                                      materials, connectivity)
    st   = tf.sparse.reorder(matrices['mass'].to_sparse_tensor())
    M    = tf.sparse.to_dense(st).numpy()
    Mref = _reference_matrices(edges, {tag: 1.0 for tag in _BETA}, 1.0)[0]
    np.testing.assert_allclose(M, Mref, rtol=1.0e-6, atol=1.0e-9)


def test_a_non_positive_beta_is_refused_by_the_assembly(tmp_path):
    """beta_ref divides the conductivity and weights the mass, so a
    non-positive value must stop the assembly rather than produce a mass matrix
    that is not positive definite, which would quietly invalidate ConjGrad. The
    two paths must refuse it alike.
    """
    mesh_file = str(tmp_path / 'negative.pkl')
    _write_cable(mesh_file, _BETA)
    with pytest.raises(ValueError, match='must be positive'):
        _assemble(mesh_file, _BETA, beta_ref=-_BETA[1])
    with pytest.raises(ValueError, match='must be positive'):
        _assemble(mesh_file, {1: _BETA[1], 2: -_BETA[2]}, beta_ref=_BETA[1])


def test_both_assembly_paths_apply_the_coefficient(tmp_path):
    """The vectorised path builds the mass matrix from the element measures and
    never called the material function, so it could ignore the coefficient
    while the per-element path applied it. A property that is not region-based
    is what sends the assembly down the per-element path, so the two are
    compared on the same problem.
    """
    mesh_file = str(tmp_path / 'paths.pkl')
    edges     = _write_cable(mesh_file, _BETA)
    beta_ref  = _BETA[1]
    vectorised, _K = _assemble(mesh_file, _BETA, beta_ref=beta_ref)

    domain = Triangulation()
    domain.readMesh(mesh_file)
    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    materials    = MaterialProperties()
    # 'elem' rather than 'region': the same values, addressed per element, which
    # the vectorised gather cannot use and the per-element function can
    per_elem = {'Edges': np.array([_BETA[int(edge[-1])] for edge in edges], dtype=np.float64)}
    materials.add_element_property('beta', 'elem', per_elem)
    materials.add_element_property('beta_ref', 'uniform', beta_ref)
    materials.add_ud_function('mass', mass_beta_weight)
    matrices = assemble_matrices_dict({'mass': localMass}, pattern, domain,
                                      materials, connectivity)
    st       = tf.sparse.reorder(matrices['mass'].to_sparse_tensor())
    per_element_path = tf.sparse.to_dense(st).numpy()
    np.testing.assert_allclose(per_element_path, vectorised, rtol=1.0e-6, atol=1.0e-9)


# ---- through the parameter-file front end -----------------------------------
def test_the_front_end_takes_the_reference_from_the_default_region():
    """reference_beta() is imp_region[0]'s cellSurfVolRatio * volFrac whenever
    that region governs a tag, so a file with one beta divides by the value it
    always did. A pure-diffusion file with no imp_region falls back to the
    documented default, which is what the beta map holds for every tag then.
    """
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-num_imp_regions', '2',
                                        '-imp_region[0].cellSurfVolRatio', '0.2',
                                        '-imp_region[0].volFrac', '0.5',
                                        '-imp_region[1].cellSurfVolRatio', '0.8',
                                        '-imp_region[1].volFrac', '1.0',
                                        '-imp_region[1].ID', '2']))
    maps = mapper.element_property_maps({1, 2})
    assert maps['beta'][1] == pytest.approx(0.1)          # the default region's
    assert maps['beta'][2] == pytest.approx(0.8)
    beta_ref = mapper.reference_beta(maps['beta'])
    assert beta_ref == pytest.approx(0.1)
    # the weights the mass integral then carries
    assert maps['beta'][1] / beta_ref == pytest.approx(1.0)
    assert maps['beta'][2] / beta_ref == pytest.approx(8.0)

    bare = ParameterMapper()
    bare.resolve(OptionReader().read(['-bidm_eqv_mono', '0']))
    bare_maps = bare.element_property_maps({1})
    assert bare.reference_beta(bare_maps['beta']) == pytest.approx(0.14)


def test_a_default_region_that_governs_nothing_does_not_rescale_the_system():
    """The default region's beta can belong to no element, when the other
    regions claim every tag of the mesh between them. Dividing by it would
    rescale both matrices of a problem whose beta is in fact uniform, and
    ConjGrad's default tolerance is ABSOLUTE, so the solve accuracy would move
    for no reason. The reference then comes from the map itself.
    """
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-num_imp_regions', '2',
                                        '-imp_region[0].cellSurfVolRatio', '0.5',
                                        '-imp_region[0].ID', '99',
                                        '-imp_region[1].cellSurfVolRatio', '0.14',
                                        '-imp_region[1].ID', '1,2']))
    maps = mapper.element_property_maps({1, 2})
    assert sorted(maps['beta'].values()) == pytest.approx([0.14, 0.14])
    assert mapper.reference_beta(maps['beta']) == pytest.approx(0.14)


def test_a_non_positive_reference_is_refused():
    """beta_ref divides the conductivity, so zero is not a degenerate case to
    be carried into the assembly."""
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-imp_region[0].volFrac', '0.0']))
    with pytest.raises(ValueError, match='must be positive'):
        mapper.element_property_maps({1})


def test_the_run_registers_both_beta_and_the_reference(tmp_path, monkeypatch):
    """The wiring: a parameter-file run must register beta as the per-region
    map AND beta_ref as the single reference, and ask for the weighted mass
    function. Everything above tests the arithmetic; this is what puts the two
    halves of it in front of the assembler.
    """
    folder = str(tmp_path)
    npt    = _NELEM + 1
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(npt))
        for ipt in range(npt):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * 100.0))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(iel, iel + 1,
                                              1 if iel < _NELEM // 2 else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')
    with open(os.path.join(folder, 'cable.par'), 'w') as fout:
        fout.write('meshname        = cable\n'
                   'simID           = OUT\n'
                   'tend            = 0.025\n'
                   'dt              = 25.0\n'
                   'bidm_eqv_mono   = 0\n'
                   'num_gregions    = 1\n'
                   'gregion[0].g_il = 0.01\n'
                   'gregion[0].g_it = 0.01\n'
                   'num_imp_regions = 2\n'
                   'imp_region[0].im               = "MitchellSchaeffer"\n'
                   'imp_region[0].cellSurfVolRatio = 0.14\n'
                   'imp_region[0].ID               = 1\n'
                   'imp_region[1].im               = "MitchellSchaeffer"\n'
                   'imp_region[1].cellSurfVolRatio = 0.56\n'
                   'imp_region[1].ID               = 2\n')
    # what the runner registers is recorded as it registers it: the solver
    # drops its element properties once the matrices are assembled, so they
    # cannot be read back afterwards
    properties : dict = {}
    functions : dict  = {}
    orig_prop = HeatSolver.add_element_material_property
    orig_func = HeatSolver.add_material_function

    def spy_property(self, pname, ptype, prop):
        properties[pname] = (ptype, prop)
        return(orig_prop(self, pname, ptype, prop))

    def spy_function(self, fname, fsign):
        functions[fname] = fsign
        return(orig_func(self, fname, fsign))

    monkeypatch.setattr(HeatSolver, 'add_element_material_property', spy_property)
    monkeypatch.setattr(HeatSolver, 'add_material_function', spy_function)
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        mapper = ParameterMapper()
        mapper.resolve(OptionReader().read(['+F', 'cable.par']))
        runner = SimulationRunner({'verbose': False})
        runner.set_mapper(mapper)
        runner.build()
    finally:
        os.chdir(cwd)
    assert properties['beta'][0] == 'region'
    assert properties['beta'][1] == {1: pytest.approx(0.14), 2: pytest.approx(0.56)}
    assert properties['beta_ref'] == ('uniform', pytest.approx(0.14))
    assert functions['mass'] is mass_beta_weight

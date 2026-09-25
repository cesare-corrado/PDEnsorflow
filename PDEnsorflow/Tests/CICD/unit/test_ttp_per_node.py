#!/usr/bin/env python
"""
    Tier-1 tests of the per-node parameters of the ten Tusscher-Panfilov model
    and of their path from a parameter file to differentiate():

      * a two-region run through the parameter-file front end, with GNa*0.0 in
        one region (a scar), builds and leaves the scar at rest;
      * per-node GNa, GKr and GKs columns reach differentiate() as (n,) vectors
        with the right values (a (n, 1) column must not broadcast to (n, n));
      * a model whose nodes have different cell types matches, node by node,
        three uniform single-type models;
      * the vectorised MonodomainSolver.assign_nodal_properties gives the same
        columns as the node-by-node loop it replaced.

    Small problems, short runs, CPU-friendly.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.carp_compatibility.main import main
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.IO.readers import IGBReader
from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov, CELL_TYPE_IDS
from gpuSolve.ionic.tomek import Tomek
from gpuSolve.physics.monodomainSolver import MonodomainSolver

_DT = 0.02          # ms


def _run_cells(model: TenTusscherPanfilov, nnodes: int, nsteps: int) -> np.ndarray:
    """Forward-Euler single cells, all nodes stimulated for the first ms;
    returns V (nsteps, nnodes)."""
    U = tf.Variable(tf.fill([nnodes, 1], tf.constant(-86.2, dtype=tf.float32)))
    model.initialize_state_variables(U)
    trace = np.zeros((nsteps, nnodes))
    for step in range(nsteps):
        dU = model.differentiate(U)
        stim = 52.0 if step * _DT < 1.0 else 0.0
        U.assign(U + _DT * (dU + stim))
        trace[step] = U.numpy()[:, 0]
    return(trace)


# ---- (b) per-node conductances reach differentiate() -----------------------
def test_per_node_conductances_reach_differentiate():
    """A node of a model with per-node GNa, GKr, GKs gives the same current as
    a uniform model holding that node's values, and dU keeps shape (n, 1)."""
    gna = np.array([[0.0], [14.838], [7.0]])
    gkr = np.array([[0.153], [0.2295], [0.05]])
    gks = np.array([[0.392], [0.588], [0.1]])
    model = TenTusscherPanfilov(dt=_DT)
    model.set_parameter('GNa', gna)
    model.set_parameter('GKr', gkr)
    model.set_parameter('GKs', gks)
    assert model.get_parameter('GNa').dtype == tf.float32
    assert tuple(model.get_parameter('GNa').shape) == (3,)
    np.testing.assert_allclose(model.get_parameter('GNa').numpy(), gna[:, 0], rtol=1e-6)
    U = tf.Variable(tf.constant([[-40.0], [-40.0], [-40.0]], dtype=tf.float32))
    model.initialize_state_variables(U)
    dU = model.differentiate(U)
    assert tuple(dU.shape) == (3, 1)
    for node in range(3):
        single = TenTusscherPanfilov(dt=_DT)
        single.set_parameter('GNa', gna[node, 0])
        single.set_parameter('GKr', gkr[node, 0])
        single.set_parameter('GKs', gks[node, 0])
        Us = tf.Variable(tf.constant([[-40.0]], dtype=tf.float32))
        single.initialize_state_variables(Us)
        np.testing.assert_allclose(dU.numpy()[node, 0], single.differentiate(Us).numpy()[0, 0],
                                   rtol=1e-5, atol=1e-6)


def test_gna_is_a_tensor_and_non_tunable_parameters_stay_uniform():
    """The build failed on GNa being a Python float; a table parameter refuses
    per-node values instead of silently misbehaving."""
    model = TenTusscherPanfilov(dt=_DT)
    assert model.get_parameter('GNa').numpy() == pytest.approx(14.838)
    with pytest.raises(ValueError, match='one value for all the nodes'):
        model.set_parameter('Ko', np.array([[5.4], [6.0]]))
    model.set_parameter('Ko', np.array([[6.0], [6.0]]))
    assert model.get_parameter('Ko') == pytest.approx(6.0)


# ---- (c) mixed cell types --------------------------------------------------
def test_mixed_cell_types_match_uniform_models():
    """Nodes typed EPI, MCELL, ENDO in one model follow the uniform models of
    their own type (GKs, Gto and the S gate all switch per node)."""
    nsteps = int(15.0 / _DT)
    mixed = TenTusscherPanfilov(dt=_DT)
    mixed.set_parameter('celltype', np.array([[CELL_TYPE_IDS['EPI']], [CELL_TYPE_IDS['MCELL']],
                                              [CELL_TYPE_IDS['ENDO']]]))
    np.testing.assert_allclose(mixed.get_parameter('GKs').numpy(), [0.392, 0.098, 0.392], rtol=1e-6)
    np.testing.assert_allclose(mixed.get_parameter('Gto').numpy(), [0.294, 0.294, 0.073], rtol=1e-6)
    trace = _run_cells(mixed, 3, nsteps)
    for node, ctype in enumerate(('EPI', 'MCELL', 'ENDO')):
        uniform = _run_cells(TenTusscherPanfilov(dt=_DT, cell_type=ctype), 1, nsteps)
        np.testing.assert_allclose(trace[:, node], uniform[:, 0], rtol=0.0, atol=1e-3)
    # the types really differ (ENDO has a smaller Ito, so a higher plateau)
    assert np.abs(trace[:, 0] - trace[:, 2]).max() > 1.0


def test_flags_per_region_scale_each_regions_own_default():
    """flags differ by region; GKs*1.5 scales the default of each region's
    own type, and celltype comes first in the map."""
    argv = ['-num_imp_regions', '2']
    for index, (flag, tag) in enumerate((('ENDO', 1), ('MCELL', 2))):
        argv += ['-imp_region[{}].im'.format(index), 'tenTusscherPanfilov',
                 '-imp_region[{}].im_param'.format(index), 'GKs*1.5,flags={}'.format(flag),
                 '-imp_region[{}].ID'.format(index), str(tag)]
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    assert mapper.ionic_model_options() == {'cell_type': 'EPI'}
    model = mapper.ionic_model_class()(dt=_DT, **mapper.ionic_model_options())
    maps  = mapper.ionic_parameter_maps(model, {1, 2})
    assert list(maps.keys())[0] == 'celltype'
    assert maps['celltype'] == {1: CELL_TYPE_IDS['ENDO'], 2: CELL_TYPE_IDS['MCELL']}
    assert maps['GKs'][1] == pytest.approx(1.5 * 0.392)
    assert maps['GKs'][2] == pytest.approx(1.5 * 0.098)


# ---- (a) end to end: a scar region with GNa*0.0 ----------------------------
_NELEM = 40
_DX    = 100.0      # micrometres


def _write_two_region_cable(folder: str, scar_gna: str):
    """A 4 mm cable: the first half tagged 1 (healthy), the second 2 (scar,
    GNa scaled by scar_gna)."""
    npt = _NELEM + 1
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(npt))
        for ipt in range(npt):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * _DX))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(iel, iel + 1, 1 if iel < _NELEM // 2 else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')
    with open(os.path.join(folder, 'cable.par'), 'w') as fout:
        fout.write('meshname = cable\n'
                   'simID    = OUT\n'
                   'tend     = 8.0\n'
                   'dt       = 20.0\n'
                   'spacedt  = 0.5\n'
                   'timedt   = 100.0\n'
                   'bidm_eqv_mono = 0\n'
                   'num_imp_regions = 2\n'
                   'imp_region[0].im       = "tenTusscherPanfilov"\n'
                   'imp_region[0].im_param = "GNa*1.00,GKr*1.50,GKs*1.50,flags=ENDO"\n'
                   'imp_region[0].num_IDs  = 1\n'
                   'imp_region[0].ID[0]    = 1\n'
                   'imp_region[1].im       = "tenTusscherPanfilov"\n'
                   'imp_region[1].im_param = "GNa*{}"\n'
                   'imp_region[1].num_IDs  = 1\n'
                   'imp_region[1].ID[0]    = 2\n'
                   'num_gregions  = 1\n'
                   'gregion[0].g_il = 0.174\n'
                   'gregion[0].g_it = 0.174\n'
                   'num_stim = 1\n'
                   'stim[0].pulse.strength = 60.0\n'
                   'stim[0].ptcl.start     = 0.0\n'
                   'stim[0].ptcl.duration  = 1.0\n'
                   'stim[0].elec.p0[0]     = 0.0\n'
                   'stim[0].elec.p1[0]     = 250.0\n'.format(scar_gna))


def _run_cable(folder: str, scar_gna: str) -> np.ndarray:
    """Runs the two-region cable through the front end; returns V (frames, nodes)."""
    os.makedirs(folder, exist_ok=True)
    _write_two_region_cable(folder, scar_gna)
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        status = main(['+F', 'cable.par'])
    finally:
        os.chdir(cwd)
    assert status == 0
    reader = IGBReader()
    reader.read(os.path.join(folder, 'OUT', 'vm.igb'))
    return(np.array(reader.data()).reshape(reader.header()['t'], reader.header()['x']))


def test_scar_region_with_gna_zero_builds_and_stays_at_rest(tmp_path):
    """The failing configuration in miniature: per-region GNa (a Python float
    before the fix), mixed cell types (ENDO healthy half, EPI scar), a paced
    healthy half. The build succeeds and the healthy half fires. The far
    millimetre of the scar stays below -40 mV (it only sees the electrotonic
    foot of the plateau next door), while in the control run, GNa*1.0, the
    same nodes fire: so it is GNa = 0 that keeps them at rest."""
    far = slice(_NELEM // 2 + 10, None)
    V = _run_cable(str(tmp_path / 'scar'), '0.0')
    assert np.all(np.isfinite(V))
    assert V[:, :_NELEM // 2 - 2].max(axis=0).min() > 0.0      # every healthy node fired
    assert V[:, far].max() < -40.0                             # no far scar node did
    control = _run_cable(str(tmp_path / 'control'), '1.0')
    assert control[:, far].max(axis=0).min() > 0.0


# ---- (d) vectorised assign_nodal_properties --------------------------------
class _Domain:
    def __init__(self, ids):
        self._ids = ids

    def point_region_ids(self):
        return(self._ids)


class _Recorder:
    """A cell model stand-in that records what set_parameter receives."""
    def __init__(self):
        self._set = {}

    def get_parameter(self, pname):
        return(tf.constant(1.5, dtype=tf.float32))

    def set_parameter(self, pname, pvalue):
        self._set[pname] = pvalue


def _loop_reference(materials, ionic, point_region_ids) -> dict:
    """The node-by-node loop that assign_nodal_properties used to run."""
    out = {}
    npt = point_region_ids.shape[0]
    for mat_prop in materials.nodal_property_names():
        prtype = materials.nodal_property_type(mat_prop)
        refval = ionic.get_parameter(mat_prop)
        if prtype == 'uniform':
            out[mat_prop] = materials.NodalProperty(mat_prop, -1, -1)
        else:
            pvals = np.full(shape=(npt, 1), fill_value=refval.numpy())
            for pointID, regionID in enumerate(point_region_ids):
                pvals[pointID] = materials.NodalProperty(mat_prop, pointID, regionID)
            out[mat_prop] = pvals
    return(out)


def test_vectorised_assign_nodal_properties_matches_the_loop():
    rng = np.random.default_rng(3)
    ids = rng.choice([1, 4, 7], size=200)
    nodal = rng.normal(size=(200, 1))

    def fill(materials):
        materials.add_nodal_property('A', 'region', {1: 0.5, 4: 0.0, 7: 2.25})
        materials.add_nodal_property('B', 'nodal', nodal)
        materials.add_nodal_property('C', 'nodal', {k: float(k) for k in range(200)})
        materials.add_nodal_property('D', 'uniform', 3.0)

    reference_materials = MaterialProperties()
    fill(reference_materials)
    expected = _loop_reference(reference_materials, _Recorder(), ids)

    solver = object.__new__(MonodomainSolver)
    solver._materials = MaterialProperties()
    fill(solver._materials)
    solver._Domain = _Domain(ids)
    solver._ionic_model = _Recorder()
    solver._use_renumbering = True
    # the solver is built without __init__ to isolate assign_nodal_properties,
    # so the two fields that method records what it saw in are set up by hand
    solver._nodal_cell_parameters = []
    solver._has_nodal_cell_parameters = False
    solver.assign_nodal_properties()
    got = solver._ionic_model._set
    assert set(got.keys()) == set(expected.keys())
    for pname, value in expected.items():
        np.testing.assert_array_equal(np.asarray(got[pname]), np.asarray(value))
        assert np.asarray(got[pname]).dtype == np.asarray(value).dtype


# ---- shared cell-type numbering --------------------------------------------
def test_cell_type_numbering_is_shared_with_tomek():
    """celltype = 0 is an ENDO cell in every model (Tomek's numbering), and
    each TT number selects that type's own GKs and Gto defaults."""
    assert CELL_TYPE_IDS == {'ENDO': 0, 'EPI': 1, 'MCELL': 2}
    assert float(Tomek(dt=_DT).get_parameter('celltype')) == CELL_TYPE_IDS['ENDO']
    for name, gks, gto in (('ENDO', 0.392, 0.073), ('EPI', 0.392, 0.294), ('MCELL', 0.098, 0.294)):
        model = TenTusscherPanfilov(dt=_DT, cell_type=name)
        assert float(model.get_parameter('celltype')) == CELL_TYPE_IDS[name]
        assert float(model.get_parameter('GKs')) == pytest.approx(gks)
        assert float(model.get_parameter('Gto')) == pytest.approx(gto)

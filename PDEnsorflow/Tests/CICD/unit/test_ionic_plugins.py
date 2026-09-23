#!/usr/bin/env python
"""
    Tier-1 tests of the ionic plugins: the base class IonicPlugin, the plugin
    ElectroporationDeBruinKrassowska98, the wrapper IonicModelWithPlugins, and
    the imp_region[].plugins / plug_param keys of the parameter-file front end.

    The plugin is checked against a line-by-line NumPy transcription of the
    reference's generated C step (current with the pore density at the start of
    the step, then forward Euler on n). Its pore conductance is 0/0 at V = 0 and
    at V = +-V_w (about 935 mV); there the limit must be finite and continuous
    with the formula just outside the 1e-6 mV band where it is used.

    The closed-loop agreement with the reference single-cell tool over whole
    protocols (a passive parent at rest and under a shock to +470 mV) is shown by
    Tests/DEVTESTS/ionic/electroporation_debruin_krassowska98.py, which needs
    the reference binary and is not run here.

    Small problems, CPU-only. Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.simulationrunner import SimulationRunner
from gpuSolve.IO.writers import StateWriter
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.ionic.plugins.electroporation_debruin_krassowska98 import ElectroporationDeBruinKrassowska98

EP      = ElectroporationDeBruinKrassowska98
EP_NAME = 'ElectroporationDeBruinKrassowska98'
PAR_EP  = 'Electroporation_DeBruinKrassowska98'
_DT     = 0.01                                    # ms

# the reference's constants and default parameters
_PI, _K, _T, _E = 3.14159, 1.38066e-20, 310.0, 1.6021765e-19
_DEFAULTS = {'alpha': 200.0, 'beta': 6.25e-5, 'q': 2.46, 'N0': 1.5e5,
             'sigma': 13.0, 'h': 5.0e-7, 'nn': 0.15, 'w0': 5.25}
_ND_F = _E / (_K * _T)
_V_W  = _DEFAULTS['w0'] / (_DEFAULTS['nn'] * _ND_F)   # mV, where w0 - nn V e/kT = 0

# the cable of the front-end tests: two element tags, one per half
_NELEM  = 20
_NPT    = _NELEM + 1
_DX     = 100.0                                   # micrometres
_DT_US  = 50.0                                    # microseconds


def _reference_gp(V: np.ndarray, p: dict = _DEFAULTS) -> np.ndarray:
    """The pore conductance exactly as the generated C writes it."""
    gp_f  = ((_PI * p['sigma']) * p['h']) * 0.25
    nd_vm = V * _ND_F
    nvm   = p['nn'] * nd_vm
    nvmm  = p['w0'] - nvm
    nvmp  = p['w0'] + nvm
    with np.errstate(all='ignore'):
        return((gp_f * (np.exp(nd_vm) - 1.0))
               / (((np.exp(nd_vm) * ((p['w0'] * np.exp(nvmm)) - nvm)) / nvmm)
                  - (((p['w0'] * np.exp(nvmp)) + nvm) / nvmp)))


def _reference_step(V: np.ndarray, n: np.ndarray, p: dict = _DEFAULTS) -> tuple:
    """(current, n after one step) exactly as the generated C computes them."""
    current = (_reference_gp(V, p) * n) * V
    dvm_2   = V * V
    dN1     = p['alpha'] * np.exp(p['beta'] * dvm_2)
    dN2     = ((-dN1) / p['N0']) * np.exp(((-p['q']) * p['beta']) * dvm_2)
    return((current, n + (dN1 + dN2 * n) * _DT))


def _plugin_at(V: list, n: list = None, dtype=tf.float64) -> tuple:
    """A plugin initialised at the potentials V (one node each), with its pore
    density overwritten by n when given. Returns (plugin, U)."""
    plugin = EP(dt=_DT)
    U = tf.Variable(np.reshape(np.array(V), (-1, 1)), dtype=dtype)
    plugin.initialize_state_variables(U)
    if n is not None:
        plugin.set_state_variables({'n': np.array(n)})
    return((plugin, U))


# ---- the plugin ----------------------------------------------------------------
def test_the_plugin_matches_the_reference_step():
    V = np.array([-300.0, -88.7638, -20.0, 5.0, 40.0, 300.0, 700.0, 1200.0])
    n = np.linspace(2.0e5, 3.0e7, V.size)
    plugin, U = _plugin_at(V.tolist(), n.tolist())
    current = np.reshape(plugin.compute_current(U).numpy(), (-1,))
    ref_current, ref_n = _reference_step(V, n)
    np.testing.assert_allclose(current, ref_current, rtol=1.0e-12)
    np.testing.assert_allclose(plugin.get_state_variables()['n'], ref_n, rtol=1.0e-12)


def test_the_initial_pore_density_is_the_steady_state():
    V = np.array([-88.7638, -20.0, 150.0])
    plugin, U = _plugin_at(V.tolist())
    n0 = plugin.get_state_variables()['n']
    np.testing.assert_allclose(n0, _DEFAULTS['N0'] * np.exp(_DEFAULTS['q'] * _DEFAULTS['beta'] * V * V),
                               rtol=1.0e-14)
    plugin.compute_current(U)
    np.testing.assert_allclose(plugin.get_state_variables()['n'], n0, rtol=1.0e-12)


@pytest.mark.parametrize('V_s', [0.0, _V_W, -_V_W], ids=['V=0', 'V=+Vw', 'V=-Vw'])
def test_the_pore_conductance_is_finite_and_continuous_at_its_singular_points(V_s):
    # the direct formula is 0/0 at the point itself; that is the case to handle
    if V_s == 0.0:
        assert np.isnan(_reference_gp(np.array([V_s])))[0]
    # the singular potential itself (not at 0, where the current V gp n is 0
    # whatever gp: that case is test_the_limit_at_zero_is_the_analytic_one)
    # and two potentials inside the band around it
    inside  = [V_s + d for d in (-5.0e-7, 5.0e-7)] + ([V_s] if V_s != 0.0 else [])
    plugin, U = _plugin_at(inside, [1.0] * len(inside))
    gp_inside = np.reshape(plugin.compute_current(U).numpy(), (-1,)) / np.array(inside)
    assert np.all(np.isfinite(gp_inside))
    # 1e-3 mV away the formula is accurate and gp is smooth: the limit must join it
    outside = np.mean(_reference_gp(np.array([V_s - 1.0e-3, V_s + 1.0e-3])))
    np.testing.assert_allclose(gp_inside, outside, rtol=1.0e-4)


def test_the_singular_points_follow_the_tunable_parameters():
    # +-V_w = +-w0/(nn e/kT) moves with w0 and nn, which plug_param can set per
    # region: each node sits exactly on the singular point of its own
    # parameters. nn = 0 removes the +-V_w points altogether (V_w is infinite).
    w0 = np.array([6.0, 4.0, 7.0, 5.25])
    nn = np.array([0.2, 0.1, 0.3, 0.0])
    with np.errstate(divide='ignore'):
        V_w = w0 / (nn * _ND_F)
    V = np.array([V_w[0], -V_w[1], V_w[2], 500.0])
    plugin = EP(dt=_DT)
    plugin.set_parameter('w0', np.reshape(w0, (-1, 1)))
    plugin.set_parameter('nn', np.reshape(nn, (-1, 1)))
    U = tf.Variable(np.reshape(V, (-1, 1)), dtype=tf.float64)
    plugin.initialize_state_variables(U)
    plugin.set_state_variables({'n': np.ones(V.size)})
    gp = np.reshape(plugin.compute_current(U).numpy(), (-1,)) / V
    assert np.all(np.isfinite(gp))
    for node in range(V.size):
        p = dict(_DEFAULTS, w0=w0[node], nn=nn[node])
        outside = np.mean(_reference_gp(np.array([V[node] - 1.0e-3, V[node] + 1.0e-3]), p))
        assert gp[node] == pytest.approx(outside, rel=1.0e-4), 'node {}'.format(node)


def test_the_limit_at_zero_is_the_analytic_one():
    p  = _DEFAULTS
    gp_f = ((_PI * p['sigma']) * p['h']) * 0.25
    gp_0 = gp_f / (np.exp(p['w0']) * (1.0 - 2.0 * p['nn'] + 2.0 * p['nn'] / p['w0'])
                   - 2.0 * p['nn'] / p['w0'])
    plugin, U = _plugin_at([1.0e-7], [1.0])
    np.testing.assert_allclose(plugin.compute_current(U).numpy().item() / 1.0e-7, gp_0, rtol=1.0e-12)
    # at V = 0 exactly: no current, finite state, and n at its steady state N0
    plugin, U = _plugin_at([0.0])
    assert plugin.compute_current(U).numpy().item() == 0.0
    assert plugin.get_state_variables()['n'][0] == pytest.approx(p['N0'], rel=1.0e-14)


def test_the_plugin_works_in_float64_whatever_the_potential():
    # 1e-4 mV: float32 arithmetic gives NaN here, float64 does not
    plugin, U = _plugin_at([1.0e-4, -88.0], dtype=tf.float32)
    current = plugin.compute_current(U)
    assert current.dtype == tf.float64
    assert plugin.state_variable('n').dtype == tf.float64
    assert np.all(np.isfinite(current.numpy()))


def test_a_plugin_cannot_run_on_its_own():
    plugin, U = _plugin_at([-80.0])
    with pytest.raises(NotImplementedError, match='plugin'):
        plugin.differentiate(U)


def test_the_plugin_accepts_only_its_own_parameters():
    plugin = EP(dt=_DT)
    plugin.set_parameter('sigma', 26.0)
    assert float(plugin.get_parameter('sigma')) == 26.0
    with pytest.raises(ValueError, match='not a tunable parameter'):
        plugin.set_parameter('GNa', 1.0)
    with pytest.raises(ValueError, match='w0'):
        plugin.set_parameter('w0', 0.0)


# ---- the wrapper ---------------------------------------------------------------
def _wrapped(nodes: int = 4) -> tuple:
    """mMS (dimensional, [-80, 20] mV) with the plugin, at four potentials."""
    model = IonicModelWithPlugins(dt=_DT)
    model.set_model(ModifiedMS2v(dt=_DT))
    model.add_plugin(EP(dt=_DT))
    U = tf.Variable(np.reshape(np.array([-80.0, -20.0, 10.0, 300.0][:nodes]), (-1, 1)),
                    dtype=tf.float32)
    model.initialize_state_variables(U)
    return((model, U))


def test_the_wrapper_subtracts_the_plugin_current_from_the_model_one():
    model, U = _wrapped()
    parent = ModifiedMS2v(dt=_DT)
    parent.initialize_state_variables(U)
    plugin, _U64 = _plugin_at([-80.0, -20.0, 10.0, 300.0])
    expected = parent.differentiate(U) - tf.cast(plugin.compute_current(U), tf.float32)
    dU = model.differentiate(U)
    assert dU.dtype == U.dtype
    np.testing.assert_allclose(dU.numpy(), expected.numpy(), rtol=1.0e-6)
    np.testing.assert_allclose(model.get_state_variables()['{}.n'.format(EP_NAME)],
                               plugin.get_state_variables()['n'], rtol=1.0e-14)
    np.testing.assert_allclose(model.get_state_variables()['H_state'],
                               parent.get_state_variables()['H_state'], rtol=1.0e-7)


def test_names_are_routed_to_the_model_or_to_the_plugin():
    model, _U = _wrapped()
    assert model.model_name() == 'ModifiedMS2v+{}'.format(EP_NAME)
    assert model.state_variable_names() == ('H_state', '{}.n'.format(EP_NAME))
    assert model.state_variable('H_state') is model.model().state_variable('H_state')
    assert model.state_variable('{}.n'.format(EP_NAME)) is model.plugins()[0].state_variable('n')
    assert float(model.get_parameter('tau_in')) == pytest.approx(0.1)
    assert float(model.get_parameter('{}.sigma'.format(EP_NAME))) == 13.0
    assert model.get_parameter('{}.GNa'.format(EP_NAME)) is None
    assert model.get_parameter('Other.sigma') is None
    model.set_parameter('{}.sigma'.format(EP_NAME), 26.0)
    assert float(model.plugins()[0].get_parameter('sigma')) == 26.0
    with pytest.raises(ValueError, match='not attached'):
        model.set_parameter('Other.sigma', 1.0)
    # the state round-trips under the prefixed names
    states = model.get_state_variables()
    states['{}.n'.format(EP_NAME)] = states['{}.n'.format(EP_NAME)] * 2.0
    model.set_state_variables(states)
    np.testing.assert_allclose(model.plugins()[0].get_state_variables()['n'],
                               states['{}.n'.format(EP_NAME)])


def test_a_plugin_is_attached_once_and_only_to_a_cell_model():
    model = IonicModelWithPlugins(dt=_DT)
    model.set_model(ModifiedMS2v(dt=_DT))
    model.add_plugin(EP(dt=_DT))
    with pytest.raises(ValueError, match='already attached'):
        model.add_plugin(EP(dt=_DT))
    with pytest.raises(ValueError, match='plugin, not a cell model'):
        IonicModelWithPlugins().set_model(EP())


def test_the_switch_removes_the_plugin_current_node_by_node():
    model, U = _wrapped()
    model.set_parameter('{}.active'.format(EP_NAME), np.array([[1.0], [0.0], [1.0], [0.0]]))
    parent = ModifiedMS2v(dt=_DT)
    parent.initialize_state_variables(U)
    plugin, _U64 = _plugin_at([-80.0, -20.0, 10.0, 300.0])
    dU_parent = parent.differentiate(U).numpy()
    current   = plugin.compute_current(U).numpy()
    dU = model.differentiate(U).numpy()
    np.testing.assert_allclose(dU[[1, 3]], dU_parent[[1, 3]], rtol=1.0e-7)
    np.testing.assert_allclose(dU[[0, 2]], (dU_parent - current.astype(np.float32))[[0, 2]], rtol=1.0e-6)
    # where the plugin is off its state is still advanced
    np.testing.assert_allclose(model.get_state_variables()['{}.n'.format(EP_NAME)],
                               plugin.get_state_variables()['n'], rtol=1.0e-14)
    with pytest.raises(ValueError, match='0 or 1'):
        model.set_parameter('{}.active'.format(EP_NAME), 0.5)


def test_the_time_step_is_handed_down_on_initialization():
    model = IonicModelWithPlugins()
    model.set_model(ModifiedMS2v())
    model.add_plugin(EP())
    model._dt = 0.02                  # what MonodomainSolver does to its cell model
    model.initialize_state_variables(tf.Variable([[-80.0]]))
    assert model.model().dt() == 0.02
    assert model.plugins()[0].dt() == 0.02


# ---- the parameter-file front end ----------------------------------------------
def _mapper(argv: list) -> ParameterMapper:
    """Resolve a command line straight into a mapper."""
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


def _two_regions(plugins0: str = PAR_EP, plug_param0: str = '', extra: list = None) -> list:
    """Two imp_regions on tags 1 and 2, the plugin listed in the first only."""
    argv = ['-num_imp_regions', '2',
            '-imp_region[0].im', 'mMS', '-imp_region[0].ID', '1',
            '-imp_region[0].plugins', plugins0,
            '-imp_region[1].im', 'mMS', '-imp_region[1].ID', '2']
    if len(plug_param0) > 0:
        argv += ['-imp_region[0].plug_param', plug_param0]
    return(argv + (extra if extra is not None else []))


def test_the_plugin_list_is_read_and_checked():
    mapper = _mapper(_two_regions())
    assert mapper.region_plugins(0) == [PAR_EP]
    assert mapper.region_plugins(1) == []
    assert mapper.ionic_plugin_classes() == [EP]
    with pytest.raises(ValueError, match='unknown plugin'):
        _mapper(_two_regions('NotAPlugin')).ionic_plugin_classes()
    with pytest.raises(ValueError, match='more than once.*first copy'):
        _mapper(_two_regions('{0}:{0}'.format(PAR_EP))).ionic_plugin_classes()


def test_a_plugin_needs_a_cell_model():
    mapper = _mapper(['-imp_region[0].plugins', PAR_EP])
    with pytest.raises(ValueError, match='cannot run on its own'):
        mapper.ionic_plugin_classes()


def test_plug_param_and_the_regional_switch_become_tag_maps():
    mapper = _mapper(_two_regions(plug_param0='sigma*2,w0=6'))
    model  = IonicModelWithPlugins()
    model.set_model(ModifiedMS2v())
    model.add_plugin(EP())
    maps = mapper.plugin_parameter_maps(model, {1, 2})
    assert maps['{}.active'.format(EP_NAME)] == {1: 1.0, 2: 0.0}
    assert maps['{}.sigma'.format(EP_NAME)] == {1: 26.0, 2: 13.0}
    assert maps['{}.w0'.format(EP_NAME)] == {1: 6.0, 2: 5.25}


def test_a_plugin_on_every_region_needs_no_switch():
    mapper = _mapper(['-imp_region[0].im', 'mMS', '-imp_region[0].plugins', PAR_EP])
    model  = IonicModelWithPlugins()
    model.set_model(ModifiedMS2v())
    model.add_plugin(EP())
    assert mapper.plugin_parameter_maps(model, {1, 2}) == {}


@pytest.mark.parametrize('plug_param,message', [
    ('sigma=1:sigma=2', 'entries'),
    ('GNa=1', 'no parameter'),
    ('active=0', 'no parameter'),
])
def test_a_malformed_plug_param_is_refused(plug_param, message):
    mapper = _mapper(_two_regions(plug_param0=plug_param))
    model  = IonicModelWithPlugins()
    model.set_model(ModifiedMS2v())
    model.add_plugin(EP())
    with pytest.raises(ValueError, match=message):
        mapper.plugin_parameter_maps(model, {1, 2})


def _write_cable(folder: str, shuffle: bool = False):
    """A 1D cable in the three-file format, tag 1 on the first half and tag 2
    on the second. With shuffle the node numbers are scrambled, so renumbering
    is far from the identity."""
    order = np.arange(_NPT)
    if shuffle:
        order = np.random.default_rng(3).permutation(_NPT)
    xcoord = np.zeros(_NPT)
    xcoord[order] = np.arange(_NPT) * _DX
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(xcoord[ipt]))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(order[iel], order[iel + 1], 1 if iel < _NELEM // 2 else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


def _build(argv: list) -> SimulationRunner:
    runner = SimulationRunner({'verbose': False})
    runner.set_mapper(_mapper(argv))
    runner.build()
    return(runner)


def _cable_argv(plugins: bool = True, extra: list = None) -> list:
    argv = ['-meshname', 'cable', '-simID', 'OUT', '-tend', '1.0', '-dt', str(_DT_US),
            '-num_imp_regions', '2',
            '-imp_region[0].im', 'mMS', '-imp_region[0].ID', '1',
            '-imp_region[1].im', 'mMS', '-imp_region[1].ID', '2']
    if plugins:
        argv += ['-imp_region[0].plugins', PAR_EP, '-imp_region[0].plug_param', 'sigma*2']
    return(argv + (extra if extra is not None else []))


def test_a_parameter_file_run_attaches_the_plugin_to_its_region(tmp_path, monkeypatch):
    _write_cable(str(tmp_path))
    monkeypatch.chdir(str(tmp_path))
    runner = _build(_cable_argv())
    ionic  = runner.model().ionic_model()
    assert isinstance(ionic, IonicModelWithPlugins)
    active = np.reshape(ionic.get_parameter('{}.active'.format(EP_NAME)).numpy(), (-1,))
    sigma  = np.reshape(ionic.get_parameter('{}.sigma'.format(EP_NAME)).numpy(), (-1,))
    # node i sits at x = i*dx: nodes of the first half carry the plugin
    assert np.all(active[:_NELEM // 2] == 1.0) and np.all(active[_NELEM // 2 + 1:] == 0.0)
    assert np.all(sigma[:_NELEM // 2] == 26.0) and np.all(sigma[_NELEM // 2 + 1:] == 13.0)
    runner.run()
    assert np.all(np.isfinite(runner.model().U().numpy()))
    # a run without plugins keeps the bare model
    plain = _build(_cable_argv(plugins=False, extra=['-simID', 'OUT_PLAIN']))
    assert isinstance(plain.model().ionic_model(), ModifiedMS2v)


def test_a_checkpoint_names_the_plugins_and_restores_only_with_them(tmp_path, monkeypatch):
    _write_cable(str(tmp_path))
    monkeypatch.chdir(str(tmp_path))
    written = _build(_cable_argv()).model().checkpoint()
    assert written['ionic_model'] == 'ModifiedMS2v+{}'.format(EP_NAME)
    assert '{}.n'.format(EP_NAME) in written['state_variables']
    StateWriter().write(written, 'with_plugin.pkl')
    with pytest.raises(ValueError, match='cell model'):
        _build(_cable_argv(plugins=False, extra=['-start_statef', 'with_plugin.pkl']))
    plain = _build(_cable_argv(plugins=False)).model().checkpoint()
    StateWriter().write(plain, 'without_plugin.pkl')
    with pytest.raises(ValueError, match='cell model'):
        _build(_cable_argv(extra=['-start_statef', 'without_plugin.pkl']))


def test_the_plugin_state_is_renumbered_with_the_rest(tmp_path, monkeypatch):
    _write_cable(str(tmp_path), shuffle=True)
    monkeypatch.chdir(str(tmp_path))
    model = _build(_cable_argv(extra=['-renumbering', '1'])).model()
    perm  = model.renumbering()['perm']
    assert not np.array_equal(perm, np.arange(_NPT))
    written = model.checkpoint()
    ramp = np.arange(_NPT, dtype=np.float64)
    for name in written['state_variables']:
        written['state_variables'][name] = written['state_variables'][name] * (1.0 + 1.0e-2 * ramp)
    StateWriter().write(written, 'saved.pkl')
    restored = _build(_cable_argv(extra=['-renumbering', '1', '-start_statef', 'saved.pkl'])).model()
    back = restored.checkpoint()
    solver_order = restored.ionic_model().get_state_variables()
    name = '{}.n'.format(EP_NAME)
    np.testing.assert_allclose(back['state_variables'][name], written['state_variables'][name], rtol=1.0e-12)
    np.testing.assert_allclose(solver_order[name], written['state_variables'][name][perm], rtol=1.0e-12)

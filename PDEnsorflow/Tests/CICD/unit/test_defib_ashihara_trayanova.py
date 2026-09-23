#!/usr/bin/env python
"""
    Tier-1 tests of the ionic plugin DefibAshiharaTrayanova, the outward
    current Ia activated by strong depolarization (Ashihara and Trayanova 2004,
    formulated by Cheng et al. 1999, Eq. 1).

    The plugin has two lower branches. The default is the one of Cheng et al.,
    exp(0.09 (V - 100)), checked against Eq. 1 and for continuity at VtakeOff.
    The reference form, exp(0.09 (V - VtakeOff)), is checked against a NumPy
    transcription of the reference's generated C to 1e-12. The plugin has no
    state variable: the reference's Ki never changes, because its conversion
    factor sl_i2c is 0 for a plugin.

    The closed-loop agreement with the reference single-cell tool (a passive
    parent and Tomek, at rest and under shocks) is shown by
    Tests/DEVTESTS/ionic/defib_ashihara_trayanova.py, which needs the reference
    binary and is not run here.

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
from gpuSolve.ionic.plugins.defib_ashihara_trayanova import DefibAshiharaTrayanova
from gpuSolve.ionic.plugins.electroporation_debruin_krassowska98 import ElectroporationDeBruinKrassowska98

IA      = DefibAshiharaTrayanova
IA_NAME = 'DefibAshiharaTrayanova'
PAR_IA  = 'Defib_AshiharaTrayanova'
PAR_EP  = 'Electroporation_DeBruinKrassowska98'
_DT     = 0.01                                    # ms

# potentials around rest, the plateau, both sides of VtakeOff and a strong shock
_V = np.array([-300.0, -88.7638, -20.0, 40.0, 100.0, 159.999, 160.0, 160.001, 250.0, 700.0])

# the cable of the front-end tests: two element tags, one per half
_NELEM  = 20
_NPT    = _NELEM + 1
_DX     = 100.0                                   # micrometres
_DT_US  = 50.0                                    # microseconds


def _reference_Ia(V: np.ndarray, VtakeOff: float = 160.0, slopeFac: float = 1.0) -> np.ndarray:
    """Ia exactly as the reference's generated C writes it."""
    with np.errstate(over='ignore'):
        return(np.where(V > VtakeOff,
                        np.exp(.09 * (VtakeOff - 100.0)) * (((0.09 * slopeFac) * (V - VtakeOff)) + 1.0),
                        np.exp(.09 * (V - VtakeOff))))


def _cheng_Ia(V: np.ndarray) -> np.ndarray:
    """Cheng et al. (1999), Eq. 1a and 1b, with their hard-coded 160 mV."""
    with np.errstate(over='ignore'):
        return(np.where(V <= 160.0, np.exp(0.09 * (V - 100.0)),
                        np.exp(0.09 * 60.0) * (0.09 * (V - 160.0) + 1.0)))


def _current(V: np.ndarray, reference_form: bool = False, params: dict = None,
             dtype=tf.float64) -> np.ndarray:
    """The plugin current at the potentials V, one node each."""
    plugin = IA(dt=_DT)
    plugin.set_use_reference_form(reference_form)
    for name, value in (params if params is not None else {}).items():
        plugin.set_parameter(name, value)
    U = tf.Variable(np.reshape(np.asarray(V), (-1, 1)), dtype=dtype)
    plugin.initialize_state_variables(U)
    current = plugin.compute_current(U)
    assert current.dtype == tf.float64 and current.shape == U.shape
    return(np.reshape(current.numpy(), (-1,)))


# ---- the current ---------------------------------------------------------------
def test_the_reference_form_matches_the_reference_expression():
    np.testing.assert_allclose(_current(_V, reference_form=True), _reference_Ia(_V), rtol=1.0e-12)
    params = {'VtakeOff': 130.0, 'slopeFac': 2.5}
    np.testing.assert_allclose(_current(_V, reference_form=True, params=params),
                               _reference_Ia(_V, 130.0, 2.5), rtol=1.0e-12)


def test_the_default_form_is_cheng_equation_1():
    np.testing.assert_allclose(_current(_V), _cheng_Ia(_V), rtol=1.0e-12)
    # 1 uA/uF at +100 mV, negligible at rest
    np.testing.assert_allclose(_current(np.array([100.0])), [1.0], rtol=1.0e-14)
    assert _current(np.array([-80.0]))[0] < 1.0e-7


@pytest.mark.parametrize('VtakeOff', [100.0, 160.0, 210.0])
def test_the_default_form_is_continuous_at_the_take_off_potential(VtakeOff):
    params = {'VtakeOff': VtakeOff}
    eps = 1.0e-6
    below, at, above = _current(np.array([VtakeOff - eps, VtakeOff, VtakeOff + eps]), params=params)
    np.testing.assert_allclose([below, above], [at, at], rtol=1.0e-6)
    # slopeFac = 1: the linear branch is the tangent of the exponential
    h = 1.0e-3
    left  = (at - _current(np.array([VtakeOff - h]), params=params)[0]) / h
    right = (_current(np.array([VtakeOff + h]), params=params)[0] - at) / h
    np.testing.assert_allclose(left, right, rtol=1.0e-3)


def test_the_reference_form_jumps_at_the_take_off_potential_and_the_forms_meet_at_100_mV():
    below, above = _current(np.array([160.0, 160.001]), reference_form=True)
    assert above / below > 200.0
    params = {'VtakeOff': 100.0}
    np.testing.assert_allclose(_current(_V, params=params),
                               _current(_V, reference_form=True, params=params), rtol=1.0e-14)


def test_the_current_stays_finite_at_any_potential():
    # the unselected lower branch would overflow above about +7990 mV
    V = np.array([-1.0e4, 8.0e3, 1.0e4, 1.0e5])
    for reference_form in (False, True):
        current = _current(V, reference_form=reference_form)
        assert np.all(np.isfinite(current))
        np.testing.assert_allclose(current[1:], _cheng_Ia(V[1:]), rtol=1.0e-12)


def test_the_plugin_works_in_float64_whatever_the_potential():
    V = np.array([-80.0, 150.0, 300.0])
    np.testing.assert_allclose(_current(V, dtype=tf.float32),
                               _cheng_Ia(V.astype(np.float32).astype(np.float64)), rtol=1.0e-12)


def test_per_node_parameters():
    V = np.array([150.0, 150.0, 300.0])
    params = {'VtakeOff': np.reshape([140.0, 160.0, 160.0], (-1, 1)),
              'slopeFac': np.reshape([1.0, 1.0, 3.0], (-1, 1))}
    expected = [np.exp(0.09 * 40.0) * (0.09 * 10.0 + 1.0),
                np.exp(0.09 * 50.0),
                np.exp(0.09 * 60.0) * (0.27 * 140.0 + 1.0)]
    np.testing.assert_allclose(_current(V, params=params), expected, rtol=1.0e-12)


def test_the_plugin_has_no_state_and_only_its_own_parameters():
    plugin = IA(dt=_DT)
    assert plugin.state_variable_names() == ()
    assert plugin.tunable_parameter_names() == ('VtakeOff', 'slopeFac')
    assert not plugin.use_reference_form()
    with pytest.raises(ValueError, match='not a tunable parameter'):
        plugin.set_parameter('Ki', 140.0)
    with pytest.raises(ValueError, match='non-finite'):
        plugin.set_parameter('VtakeOff', np.nan)
    with pytest.raises(NotImplementedError, match='cannot run on its own'):
        plugin.differentiate(tf.zeros((2, 1), dtype=tf.float64))


def test_the_wrapper_subtracts_the_plugin_current_from_the_model_one():
    V = np.reshape([-80.0, -20.0, 10.0, 300.0], (-1, 1))
    model = IonicModelWithPlugins(dt=_DT)
    model.set_model(ModifiedMS2v(dt=_DT))
    model.add_plugin(IA(dt=_DT))
    U = tf.Variable(V, dtype=tf.float32)
    model.initialize_state_variables(U)
    parent = ModifiedMS2v(dt=_DT)
    parent.initialize_state_variables(tf.Variable(V, dtype=tf.float32))
    expected = parent.differentiate(tf.Variable(V, dtype=tf.float32)).numpy() \
        - _cheng_Ia(V).astype(np.float32)
    np.testing.assert_allclose(model.differentiate(U).numpy(), expected, rtol=1.0e-6)
    assert model.state_variable_names() == parent.state_variable_names()


# ---- the parameter-file front end ----------------------------------------------
def _mapper(argv: list) -> ParameterMapper:
    """Resolve a command line straight into a mapper."""
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


def test_plug_param_reaches_the_plugin_on_its_region_only():
    argv = ['-num_imp_regions', '2',
            '-imp_region[0].im', 'mMS', '-imp_region[0].ID', '1',
            '-imp_region[0].plugins', '{}:{}'.format(PAR_EP, PAR_IA),
            '-imp_region[0].plug_param', ':VtakeOff=150,slopeFac*2',
            '-imp_region[1].im', 'mMS', '-imp_region[1].ID', '2']
    mapper = _mapper(argv)
    assert mapper.ionic_plugin_classes() == [ElectroporationDeBruinKrassowska98, IA]
    model = IonicModelWithPlugins()
    model.set_model(ModifiedMS2v())
    model.add_plugin(ElectroporationDeBruinKrassowska98())
    model.add_plugin(IA())
    maps = mapper.plugin_parameter_maps(model, {1, 2})
    assert maps['{}.active'.format(IA_NAME)] == {1: 1.0, 2: 0.0}
    assert maps['{}.VtakeOff'.format(IA_NAME)] == {1: 150.0, 2: 160.0}
    assert maps['{}.slopeFac'.format(IA_NAME)] == {1: 2.0, 2: 1.0}


def _write_cable(folder: str):
    """A 1D cable in the three-file format, tag 1 on the first half and tag 2
    on the second."""
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * _DX))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} {}\n'.format(iel, iel + 1, 1 if iel < _NELEM // 2 else 2))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


def _build(extra: list = None) -> SimulationRunner:
    argv = ['-meshname', 'cable', '-simID', 'OUT', '-tend', '1.0', '-dt', str(_DT_US),
            '-num_imp_regions', '2',
            '-imp_region[0].im', 'mMS', '-imp_region[0].ID', '1',
            '-imp_region[0].plugins', PAR_IA, '-imp_region[0].plug_param', 'VtakeOff=150',
            '-imp_region[1].im', 'mMS', '-imp_region[1].ID', '2']
    runner = SimulationRunner({'verbose': False})
    runner.set_mapper(_mapper(argv + (extra if extra is not None else [])))
    runner.build()
    return(runner)


def test_a_parameter_file_run_and_its_checkpoint(tmp_path, monkeypatch):
    _write_cable(str(tmp_path))
    monkeypatch.chdir(str(tmp_path))
    runner = _build()
    ionic  = runner.model().ionic_model()
    VtakeOff = np.reshape(ionic.get_parameter('{}.VtakeOff'.format(IA_NAME)).numpy(), (-1,))
    assert np.all(VtakeOff[:_NELEM // 2] == 150.0) and np.all(VtakeOff[_NELEM // 2 + 1:] == 160.0)
    # the checkpoint names the plugin but holds only the model's state
    written = runner.model().checkpoint()
    runner.run()
    assert np.all(np.isfinite(runner.model().U().numpy()))
    assert written['ionic_model'] == 'ModifiedMS2v+{}'.format(IA_NAME)
    assert not any(name.startswith(IA_NAME) for name in written['state_variables'])
    StateWriter().write(written, 'saved.pkl')
    restored = _build(['-simID', 'OUT_RESTORED', '-start_statef', 'saved.pkl']).model().checkpoint()
    for name, values in written['state_variables'].items():
        np.testing.assert_allclose(restored['state_variables'][name], values, rtol=1.0e-12)

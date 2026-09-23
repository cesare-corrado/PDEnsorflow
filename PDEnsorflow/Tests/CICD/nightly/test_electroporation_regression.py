#!/usr/bin/env python
"""
    Tier-2 nightly GPU regression for the electroporation plugin
    (gpuSolve.ionic.plugins.electroporation_debruin_krassowska98) attached to a
    cell model through IonicModelWithPlugins, against the single-cell tool of
    the reference implementation, bench, for two protocols:

      * Tomek at rest, no stimulus, 50 ms: the plugin's leak (about -1.9 uA/uF
        at rest) depolarises the cell until it fires at about 48 ms.

            bench --imp Tomek --plug-in Electroporation_DeBruinKrassowska98 \
                  --stim-curr 0 --duration 50 --dt 0.01 --dt-out 0.01 -v

      * a passive membrane (the reference's Plonsey model, Vrest = -80 mV) under
        a 1000 uA/uF shock from 1 to 2 ms, 20 ms: V reaches +470.9 mV and the
        pore density grows 60-fold.

            bench --imp Plonsey --imp-par "Vrest=-80" \
                  --plug-in Electroporation_DeBruinKrassowska98 \
                  --stim-curr 1000 --duration 20 --dt 0.01 --dt-out 0.01 -v

    The step follows bench's order: the state is sampled, the stimulus is added
    to V, the model and then the plugin are advanced with that V, then
    V -= dt*(Iion + I_ep). Tomek uses forward-Euler gates, the reference scheme.
    The shock uses a passive parent because Tomek itself stops being comparable
    above about +100 mV, where the rates of its IKr Markov chain exceed 2/dt
    and forward Euler is unstable in both codes.

    Tolerances: 0.05 mV on V, 1e-6 relative on n. When the test was written the
    largest differences over every step were 6.9e-3 mV and 2.4e-8 at rest (on
    the upstroke, where a shift of about 1e-5 ms in the firing time is enough),
    and 6.9e-7 mV and 1.8e-8 under the shock.

    Marked nightly + gpu. Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.ionic.ionicmodel import IonicModel
from gpuSolve.ionic.tomek import Tomek
from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.ionic.plugins.electroporation_debruin_krassowska98 import ElectroporationDeBruinKrassowska98

_DT      = 0.01                                   # ms
_V_TOL   = 0.05                                   # mV
_N_RTOL  = 1.0e-6
_PLUGIN_N = 'ElectroporationDeBruinKrassowska98.n'

# bench, dt = 0.01 ms: (t in ms, V in mV, n in cm^-2), sampled before the step
_REST_REFERENCE = [(10.0, -78.48532423403559, 503199.50362319313),
                   (20.0, -73.50277466868911, 502098.10999644647),
                   (30.0, -69.76319500304328, 500665.4862420661),
                   (40.0, -65.81774942032713, 498957.6991272843),
                   (45.0, -62.222012559385334, 497973.4312140516),
                   (50.0, 18.075936624249177, 496377.611468477)]
_SHOCK_REFERENCE = [(1.5, 405.13060468965193, 566285.2758832473),
                    (2.0, 330.5265856041065, 25348528.50331273),
                    (5.0, 1.8558210736915384, 25285258.655471273),
                    (10.0, -8.000620577146059, 25118903.222031552),
                    (20.0, -8.162683931249042, 24790176.927421823)]
_SHOCK_PEAK = 470.8684683784649                   # mV, at t = 1.59 ms


class _PassiveMembrane(IonicModel):
    """ Test parent: Iion = (V - Vrest)/Rm with Rm = 10 kOhm cm^2, the
        reference's passive model (Plonsey R, J Franklin Inst 1974;297:317-324) """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)
        self._Vrest  : float = -80.0
        self._Rm     : float = 10.0
        self._V_init : float = -80.0

    def differentiate(self, U: tf.Variable) -> tf.Variable:
        return(-(1.0 / self._Rm) * (U - self._Vrest))


def _run(parent: IonicModel, stim: float, duration: float) -> tuple:
    """(V, n) sampled at every step, before the stimulus, as bench records them."""
    model = IonicModelWithPlugins(dt=_DT)
    model.set_model(parent)
    model.add_plugin(ElectroporationDeBruinKrassowska98(dt=_DT))
    U = tf.Variable([[parent.get_parameter('V_init')]], dtype=tf.float64)
    model.initialize_state_variables(U)
    n_state = model.state_variable(_PLUGIN_N)
    V, n = [], []
    for step in range(int(round(duration / _DT)) + 1):
        V.append(U.numpy().item())
        n.append(n_state.numpy().item())
        if 100 <= step < 200:
            U.assign_add(_DT * stim * tf.ones_like(U))
        U.assign_add(_DT * model.differentiate(U))
    return((np.array(V), np.array(n)))


def _check(V: np.ndarray, n: np.ndarray, reference: list):
    assert np.all(np.isfinite(V)) and np.all(np.isfinite(n))
    for t, V_ref, n_ref in reference:
        k = int(round(t / _DT))
        assert V[k] == pytest.approx(V_ref, abs=_V_TOL), 't = {} ms'.format(t)
        assert n[k] == pytest.approx(n_ref, rel=_N_RTOL), 't = {} ms'.format(t)


@pytest.mark.nightly
@pytest.mark.gpu
def test_the_plugin_on_a_resting_tomek_cell_matches_the_reference():
    parent = Tomek(dt=_DT)
    parent.set_use_rush_larsen(False)
    V, n = _run(parent, 0.0, 50.0)
    _check(V, n, _REST_REFERENCE)


@pytest.mark.nightly
@pytest.mark.gpu
def test_the_plugin_under_a_shock_matches_the_reference():
    V, n = _run(_PassiveMembrane(dt=_DT), 1000.0, 20.0)
    _check(V, n, _SHOCK_REFERENCE)
    assert V.max() == pytest.approx(_SHOCK_PEAK, abs=_V_TOL)

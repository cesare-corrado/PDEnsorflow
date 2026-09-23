#!/usr/bin/env python
"""
    Tier-2 nightly GPU regression for the Tomek (ToR-ORd) cell model
    (gpuSolve.ionic.tomek.Tomek): one paced beat of an ENDO, an EPI and an MCELL
    cell, held as the three nodes of one model so the per-node cell-type columns
    are exercised at the same time.

    The reference numbers are those of the single-cell tool of the reference
    implementation, bench, for exactly this protocol:

        bench --imp Tomek --imp-par celltype=<0|1|2> --numstim 1 --bcl 1000 \
              --duration 1000 --dt 0.01 --dt-out 1 -v

    i.e. dt = 0.01 ms, stimulus 60 uA/uF for 1 ms from t = 1 ms, output every
    1 ms. The step follows bench's order: the stimulus is added to V, the model
    is advanced with that V, then V -= dt*Iion. Peak Vm, APD90 and the Cai peak
    are measured on the 1 ms output (APD90 threshold from the resting and the
    peak potential), and Cai is converted to uM, the unit bench reports.

    Tolerances: 1 mV on the peak, 1 ms on APD90, 2% on the Cai peak. When the
    test was written, forward Euler (the reference's own scheme) matched bench
    to 4.3e-5 mV over the whole beat, and the default (Rush-Larsen gates; since
    then also the matrix exponential for the IKr Markov chain, which left these
    three numbers unchanged) was within 0.06 mV, 0 ms and 0.07%. Both schemes
    are run: forward Euler checks the equations against the reference step for
    step, the default checks that it stays within the limits.

    The 1 ms stepping loop is compiled with XLA: the model has several hundred
    small kernels per step, and uncompiled the beat takes about 20 minutes on
    the GPU instead of about 15 seconds.

    Marked nightly + gpu. Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.ionic.tomek import Tomek

_DT        = 0.01                                 # ms
_BCL       = 1000                                 # ms, one beat, 1 ms output
_STIM      = 60.0                                 # uA/uF
_PEAK_TOL  = 1.0                                  # mV
_APD_TOL   = 1.0                                  # ms
_CAI_RTOL  = 0.02

# bench, dt = 0.01 ms: (peak Vm in mV, APD90 in ms, Cai peak in uM)
_REFERENCE = {'ENDO':  (33.76054152423551, 270.0, 0.3825427693248492),
              'EPI':   (31.79242319646382, 269.0, 0.3960654856518713),
              'MCELL': (32.60911196405978, 399.0, 0.7370934858113921)}


def _apd90(V: np.ndarray) -> float:
    """APD90 on a 1 ms trace starting at rest."""
    threshold = V.max() - 0.9 * (V.max() - V[0])
    up   = int(np.argmax(V > threshold))
    down = up + int(np.argmax(V[up:] < threshold))
    return(float(down - up))


def _one_beat(rush_larsen: bool) -> tuple:
    """Pace ENDO, EPI and MCELL nodes for one beat; return (V, Cai in uM), 1 ms samples."""
    model = Tomek(dt=_DT)
    model.set_use_rush_larsen(rush_larsen)
    model.set_parameter('celltype', np.array([[0.0], [1.0], [2.0]]))
    U = tf.Variable(np.full((3, 1), model.get_parameter('V_init')), dtype=tf.float64)
    model.initialize_state_variables(U)
    steps_per_ms = int(round(1.0 / _DT))

    @tf.function(jit_compile=True)
    def advance(stimulate: tf.Tensor):
        for _i in tf.range(steps_per_ms):
            if stimulate:
                U.assign_add(_DT * _STIM * tf.ones_like(U))
            U.assign_add(_DT * model.differentiate(U))

    V, Cai = [], []
    for ms in range(_BCL):
        V.append(U.numpy().ravel().copy())
        Cai.append(1.0e3 * model.get_state_variables()['Cai'])
        advance(tf.constant(ms == 1))
    return(np.array(V), np.array(Cai))


@pytest.mark.nightly
@pytest.mark.gpu
@pytest.mark.parametrize('rush_larsen', [False, True], ids=['forward_euler', 'rush_larsen'])
def test_tomek_beat_matches_the_reference(rush_larsen):
    V, Cai = _one_beat(rush_larsen)
    assert np.all(np.isfinite(V)) and np.all(np.isfinite(Cai))
    for node, name in enumerate(['ENDO', 'EPI', 'MCELL']):
        peak, apd, cai_peak = _REFERENCE[name]
        assert V[:, node].max() == pytest.approx(peak, abs=_PEAK_TOL), name
        assert _apd90(V[:, node]) == pytest.approx(apd, abs=_APD_TOL), name
        assert Cai[:, node].max() == pytest.approx(cai_peak, rel=_CAI_RTOL), name

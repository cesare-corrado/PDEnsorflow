#!/usr/bin/env python
"""
    Tier-1 unit test for LatDetector (gpuSolve.physics.latDetector): local
    activation time monitoring, the equivalent of the reference simulator's LAT
    detection (user guide 22.2/22.8).

    The tests drive the detector with small synthetic signals whose activation
    instants are known in closed form, so the sub-step interpolation is pinned
    to a number, not just checked for "an activation happened":
      * threshold crossing (method 1), upstroke and downstroke, with the
        linearly interpolated crossing time;
      * maximum derivative (method 2), with the turning-point time and the
        two-step history warm-up that cannot trigger;
      * the start-time filter, the first-only (all = 0) nodal vector, and the
        CARP-format file output.

    CPU-only and small, so it runs in the Tier-1 suite.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.physics.latDetector import LatDetector

TOL = 1.0e-6


def test_threshold_upstroke_interpolation():
    """ upstroke crossing time is t - dt + (thr - p)/(c - p) * dt """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1, 'dt': 1.0})
    det.init(np.array([-2.0, -5.0]))
    # node 0: -2 -> 2 over the step ending at t=1, thr=0 -> tact = 0.5
    fired = det.check(np.array([2.0, -5.0]), 1.0)
    assert fired
    # node 1: -5 -> 1 over the step ending at t=2 -> tact = 2 - 1 + (5/6) = 1.83333
    det.check(np.array([3.0, 1.0]), 2.0)
    nodes, times = det.all_activations()
    assert list(nodes) == [0, 1]
    assert times[0] == pytest.approx(0.5, abs=TOL)
    assert times[1] == pytest.approx(1.0 + 5.0 / 6.0, abs=TOL)


def test_threshold_no_spurious_trigger_at_init():
    """ the previous signal is seeded with the initial value, so a first check
        that stays on the same side of the threshold does not fire """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1, 'dt': 1.0})
    det.init(np.array([-2.0]))
    assert not det.check(np.array([-1.0]), 1.0)
    nodes, times = det.all_activations()
    assert nodes.size == 0


def test_threshold_downstroke_time_inside_step():
    """ downstroke crossing time stays inside the step (no reference sgn factor):
        p=2 -> c=-2 over the step ending at t=1, thr=0 -> tact = 0.5 """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 1, 'all': 1, 'dt': 1.0})
    det.init(np.array([2.0]))
    det.check(np.array([-2.0]), 1.0)
    nodes, times = det.all_activations()
    assert list(nodes) == [0]
    assert times[0] == pytest.approx(0.5, abs=TOL)


def test_max_derivative_turning_point_and_warmup():
    """ method 2 skips the first two steps (history not yet filled) and reports
        the turning-point time tact = t - 2 dt + ddv0/(ddv0 - ddv1) * dt """
    det = LatDetector({'method': 2, 'threshold': 0.5, 'mode': 0, 'all': 1, 'dt': 1.0})
    signal = [0.0, 1.0, 3.0, 6.0, 8.0, 9.0]  # first differences 1,2,3,2,1
    det.init(np.array([signal[0]]))
    fired = [det.check(np.array([signal[k]]), float(k)) for k in range(1, len(signal))]
    # steps 1 and 2 (t=1,2) cannot fire: the three-point history is not filled.
    assert not fired[0] and not fired[1]
    nodes, times = det.all_activations()
    assert list(nodes) == [0]
    # at t=4: ddv0=1, ddv1=-1 -> tact = 4 - 2 + 1/(1-(-1)) = 2.5
    assert times[0] == pytest.approx(2.5, abs=TOL)


def test_start_time_filter():
    """ activations earlier than start are discarded """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1,
                       'dt': 1.0, 'start': 1.0})
    det.init(np.array([-1.0]))
    # crossing at t=0.5 < start=1.0 -> dropped
    assert not det.check(np.array([1.0]), 1.0)
    nodes, _ = det.all_activations()
    assert nodes.size == 0


def test_first_only_nodal_vector():
    """ all = 0 keeps the first activation per node and leaves the rest at -1 """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 0, 'dt': 1.0})
    det.init(np.array([-1.0, -1.0, -1.0]))
    det.check(np.array([1.0, -1.0, -1.0]), 1.0)   # node 0 at 0.5
    det.check(np.array([2.0, 1.0, -1.0]), 2.0)    # node 1 at 1.5; node 0 stays
    tm = det.activation_times()
    assert tm[0] == pytest.approx(0.5, abs=TOL)
    assert tm[1] == pytest.approx(1.5, abs=TOL)
    assert tm[2] == pytest.approx(-1.0, abs=TOL)
    assert det.all_activations()[0].size == 0


def test_write_all_mode(tmp_path):
    """ all = 1 writes a <ID>.dat table of 'node<TAB>tact' rows """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1,
                       'dt': 1.0, 'ID': 'lat_test'})
    det.init(np.array([-2.0]))
    det.check(np.array([2.0]), 1.0)
    fname = det.write(str(tmp_path))
    assert fname.endswith('lat_test.dat')
    rows = [ln.split('\t') for ln in open(fname).read().splitlines()]
    assert int(rows[0][0]) == 0
    assert float(rows[0][1]) == pytest.approx(0.5, abs=TOL)


def test_write_first_only_mode(tmp_path):
    """ all = 0 writes init_acts_<ID>.dat, the nodal vector one value per line """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 0,
                       'dt': 1.0, 'ID': 'lat_test'})
    det.init(np.array([-1.0, -1.0]))
    det.check(np.array([1.0, -1.0]), 1.0)
    fname = det.write(str(tmp_path))
    assert fname.endswith('init_acts_lat_test.dat')
    vals = [float(x) for x in open(fname).read().splitlines()]
    assert vals[0] == pytest.approx(0.5, abs=TOL)
    assert vals[1] == pytest.approx(-1.0, abs=TOL)


def test_column_shaped_input():
    """ the detector accepts (n, 1) column input as well as (n,) """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1, 'dt': 1.0})
    det.init(np.array([[-2.0], [-5.0]]))
    det.check(np.array([[2.0], [-5.0]]), 1.0)
    nodes, times = det.all_activations()
    assert list(nodes) == [0]
    assert times[0] == pytest.approx(0.5, abs=TOL)

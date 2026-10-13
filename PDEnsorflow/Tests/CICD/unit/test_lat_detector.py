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
        CARP-format file output;
      * the deferred per-step count and the compiled (graph) step, which must
        detect and record exactly what the default path does and differ only in
        how the work is launched and where the host waits for the device.

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


def test_deferred_count_is_off_by_default():
    """ the default path answers with a Python bool, so a caller written
        against it keeps the value it used to get """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1, 'dt': 1.0})
    assert not det.use_deferred_count()
    det.init(np.array([-2.0]))
    fired = det.check(np.array([2.0]), 1.0)
    assert isinstance(fired, bool)
    assert det.nacts() == 1


@pytest.mark.parametrize('all_flag', [0, 1])
def test_deferred_count_detects_and_records_the_same(all_flag):
    """ the deferral moves the host wait out of the per-step path and must
        change nothing that is detected, counted or recorded: same activation
        instants, same count, and a return value that tests the same way """
    signal = [np.array([-2.0, -5.0]), np.array([2.0, -5.0]), np.array([3.0, 1.0])]
    results = {}
    for deferred in (False, True):
        det = LatDetector({'method': 1, 'threshold': 0.0, 'mode': 0, 'all': all_flag,
                           'dt': 1.0, 'use_deferred_count': deferred})
        assert det.use_deferred_count() == deferred
        det.init(signal[0])
        fired = [bool(det.check(step, float(k))) for k, step in enumerate(signal[1:], start=1)]
        if all_flag:
            nodes, times = det.all_activations()
            recorded = (list(nodes), list(times))
        else:
            recorded = list(det.activation_times())
        results[deferred] = (fired, det.nacts(), recorded)
    assert results[False][0] == results[True][0] == [True, True]
    assert results[False][1] == results[True][1] == 1
    assert np.allclose(np.array(results[False][2], dtype=float),
                       np.array(results[True][2], dtype=float), atol=TOL)


# a two-node signal that activates both nodes, falls back and activates again,
# so every path has an upstroke, a downstroke and a quiet step to agree on
_SIGNAL = [np.array([-2.0, -5.0]), np.array([-1.0, -4.0]), np.array([2.0, -3.0]),
           np.array([4.0, 1.0]), np.array([3.0, 3.0]), np.array([-3.0, -1.0])]


def _drive(config: dict) -> tuple:
    """Run the detector over _SIGNAL and return everything it reports: the
    per-step answers, the final count and whatever it recorded."""
    det = LatDetector(config)
    det.init(_SIGNAL[0])
    fired = [bool(det.check(step, float(k))) for k, step in enumerate(_SIGNAL[1:], start=1)]
    if config.get('all', 1):
        nodes, times = det.all_activations()
        recorded = (list(nodes), list(times))
    else:
        recorded = (list(range(len(det.activation_times()))), list(det.activation_times()))
    return((fired, det.nacts(), recorded))


@pytest.mark.parametrize('method', [1, 2])
@pytest.mark.parametrize('all_flag', [0, 1])
@pytest.mark.parametrize('mode', [0, 1])
def test_graph_step_detects_exactly_what_the_eager_step_does(method, all_flag, mode):
    """ the compiled step is the same operations launched together, so it must
        agree with the default path to the bit: same per-step answers, same
        count, same nodes and the same activation instants. Every combination
        of method, recording mode and slope is covered, because each takes a
        different branch through __step (the derivative history of method 2,
        the host table of all = 1, the nodal vector of all = 0).
    """
    config = {'method': method, 'threshold': 0.0, 'mode': mode, 'all': all_flag,
              'dt': 1.0}
    eager = _drive(dict(config))
    graph = _drive(dict(config, use_graph_check=True))
    assert graph[0] == eager[0]
    assert graph[1] == eager[1]
    assert graph[2][0] == eager[2][0]
    assert np.array_equal(np.array(graph[2][1], dtype=float),
                          np.array(eager[2][1], dtype=float))


def test_graph_step_and_deferred_count_combine():
    """ the two knobs are independent: a compiled step whose count stays on the
        device must still record and report the same activations """
    config = {'method': 1, 'threshold': 0.0, 'mode': 0, 'all': 1, 'dt': 1.0}
    both = _drive(dict(config, use_graph_check=True, use_deferred_count=True))
    assert both == _drive(dict(config))


def test_graph_check_is_off_by_default():
    """ the default path stays eager, so an existing caller keeps the execution
        it had as well as the numerics """
    det = LatDetector({'method': 1, 'threshold': 0.0, 'dt': 1.0})
    assert not det.use_graph_check()
    det.init(np.array([-2.0]))
    det.check(np.array([2.0]), 1.0)
    assert det.nacts() == 1


@pytest.mark.parametrize('method', [1, 2])
def test_the_graph_step_is_traced_once_for_the_whole_run(method):
    """ the point of the compiled step is one launch per step, which it only is
        if the step is traced ONCE. Passing the time or the warm-up flag as a
        Python value instead of a tensor would retrace on every step and make
        the detector far slower than the eager path, so the tracing count is
        asserted here, over enough steps to cross the method-2 warm-up
        boundary. It reads the internal attribute on purpose: the tracing count
        is not part of the class API.
    """
    # first differences 1, 2, 3, 2, 1, 1, 1: a crossing of 0 for method 1 and a
    # turning point of the derivative for method 2, so both actually trigger
    signal = [-5.0, -4.0, -2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    det = LatDetector({'method': method, 'threshold': 0.5, 'mode': 0, 'all': 1,
                       'dt': 1.0, 'use_graph_check': True})
    det.init(np.array([signal[0]]))
    for k in range(1, len(signal)):
        det.check(np.array([signal[k]]), float(k))
    assert det._graph_step.experimental_get_tracing_count() == 1
    assert det.all_activations()[0].size > 0

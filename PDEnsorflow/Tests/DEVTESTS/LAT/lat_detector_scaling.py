#!/usr/bin/env python
"""
    Does the LatDetector knob measurement hold on a large mesh?

    The per-step cost of the detector is mostly size-independent overhead
    (launches and one host wait) plus a kernel term that grows with the node
    count, so a measurement on one mesh says little about a mesh 16 or 64
    times larger. This script measures it at several node counts, for both
    execution knobs and both recording modes.

    Measured on an idle RTX 5090 (TensorFlow 2.21, float32), detector cost per
    step in us, i.e. the loop time minus the signal-generation baseline:

        nodes      all=0 eager  all=0 graph   all=1 eager  all=1 graph
          63001          955.2        456.1        1138.2        632.2
         250000         1097.6        542.7        1288.1        675.9
        1000000         1418.3        625.3        1619.1        799.5
        4000000         2235.2       1085.3        2480.1       1290.4

    The compiled step is worth about 2x at every size, and the deferred count
    was slower in all sixteen configurations, by 100 to 220 us. An earlier run
    of the same comparison on a BUSY RTX A2000 (another process holding the
    card at 99%) showed no gain at all for all = 0, which is why these numbers
    are taken on an idle card and why a repeat should check that first.

    This script repeats the measurement at several node counts. The signal is
    a front that advances by n/nsteps nodes per step, generated on the device
    so that nothing is copied in from the host, and a baseline loop times that
    generation alone: the detector cost is the difference, which is what the
    knobs can act on.

    Run it where the GPU is idle:
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<package root> python latscaling.py
"""
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gpuSolve                                  # noqa: F401  (env shim, re-execs)
import numpy as np
import tensorflow as tf

from gpuSolve.physics import LatDetector

SIZES  = [63001, 250000, 1000000, 4000000]
NSTEPS = 300
REPS   = 3
WARMUP = 20
DT     = 0.1
VREST, VPEAK, THRESHOLD = -80.0, 20.0, -30.0


def make_signal(n: int, per: int):
    """ a front that has passed k*per nodes after k steps, built on the device """
    idx = tf.range(n, dtype=tf.int32)

    @tf.function
    def signal(k):
        return(tf.where(idx < k * per, tf.constant(VPEAK), tf.constant(VREST)))

    return(signal)


def time_baseline(signal) -> float:
    """ us per step of the signal generation alone """
    best = None
    for _ in range(REPS):
        for k in range(WARMUP):
            signal(tf.constant(k, tf.int32))
        t0 = time.time()
        for k in range(NSTEPS):
            out = signal(tf.constant(k, tf.int32))
        float(out[0])                             # one sync, as a real loop has
        us = 1.0e6 * (time.time() - t0) / NSTEPS
        best = us if best is None else min(best, us)
    return(best)


def time_detector(signal, n: int, all_flag: int, graph: bool, deferred: bool) -> tuple:
    """ us per step of signal + detection, and how many activations were seen """
    best = None
    for _ in range(REPS):
        det = LatDetector({'method': 1, 'threshold': THRESHOLD, 'mode': 0,
                           'all': all_flag, 'dt': DT,
                           'use_graph_check': graph,
                           'use_deferred_count': deferred})
        det.init(signal(tf.constant(0, tf.int32)))
        for k in range(WARMUP):
            det.check(signal(tf.constant(k, tf.int32)), DT * k)
        t0 = time.time()
        for k in range(NSTEPS):
            det.check(signal(tf.constant(k, tf.int32)), DT * k)
        det.nacts()                               # the one sync of a real loop
        us = 1.0e6 * (time.time() - t0) / NSTEPS
        best = us if best is None else min(best, us)
        seen = (det.all_activations()[0].size if all_flag
                else int(np.sum(det.activation_times() >= 0.0)))
    return((best, seen))


def main():
    print('GPU: {}'.format(tf.config.list_physical_devices('GPU')), flush=True)
    print('steps per configuration: {} (best of {}, {} warm-up)'.format(
        NSTEPS, REPS, WARMUP), flush=True)
    for n in SIZES:
        per    = max(1, n // NSTEPS)
        signal = make_signal(n, per)
        base   = time_baseline(signal)
        print('\n=== {} nodes ({:.1f} MB per float32 vector), front {} nodes/step'.format(
            n, 4.0 * n / 1.0e6, per), flush=True)
        print('    signal generation alone: {:8.1f} us/step'.format(base), flush=True)
        print('    {:5s} {:7s} {:9s} {:>12s} {:>12s} {:>10s}'.format(
            'all', 'graph', 'deferred', 'us/step', 'detector', 'activations'), flush=True)
        for all_flag in (0, 1):
            for graph in (False, True):
                for deferred in (False, True):
                    us, seen = time_detector(signal, n, all_flag, graph, deferred)
                    print('    {:5d} {!s:7s} {!s:9s} {:12.1f} {:12.1f} {:10d}'.format(
                        all_flag, graph, deferred, us, us - base, seen), flush=True)


if __name__ == '__main__':
    main()

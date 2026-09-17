#!/usr/bin/env python
"""
    A TensorFlow-based Cardiac Electrophysiology Modeler

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

    Single-cell pacing of the Tomek (ToR-ORd) model for the three cell types at
    once: one node each for ENDO, EPI and MCELL. The step follows the order of
    the reference single-cell tool (bench): the stimulus is added to V, the
    model is advanced with that V, then V -= dt*Iion. With --reference, the last
    beat is compared with bench dumps written by

        bench --imp Tomek --imp-par celltype=<c> --numstim <beats> --bcl <bcl> \
              --duration <beats*bcl> --dt <dt> --dt-out 1 -v

    run in <reference>/ct0, ct1 and ct2 (bench reports Cai in uM).
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import time
import numpy as np
import tensorflow as tf

from gpuSolve.ionic.tomek import Tomek

CELL_TYPES = ('ENDO', 'EPI', 'MCELL')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='single-cell pacing of the Tomek model')
    parser.add_argument('--dt', type=float, default=0.01, help='time step (ms)')
    parser.add_argument('--bcl', type=float, default=1000.0, help='basic cycle length (ms)')
    parser.add_argument('--beats', type=int, default=1, help='number of beats')
    parser.add_argument('--stim', type=float, default=60.0, help='stimulus (mV/ms), from t = 1 ms for 1 ms')
    parser.add_argument('--forward_euler', action='store_true',
                        help='forward Euler update of the gates (the reference scheme) instead of Rush-Larsen')
    parser.add_argument('--no_xla', action='store_true', help='do not compile the stepping loop with XLA')
    parser.add_argument('--reference', default='', help='folder with bench dumps in ct0, ct1, ct2')
    return(parser.parse_args())


def apd90(time_ms: np.ndarray, V: np.ndarray) -> float:
    """ APD at 90% repolarisation, on the output sampling """
    threshold = V.max() - 0.9 * (V.max() - V[0])
    up   = int(np.argmax(V > threshold))
    down = up + int(np.argmax(V[up:] < threshold))
    return(float(time_ms[down] - time_ms[up]))


def main():
    args  = parse_arguments()
    dt    = args.dt
    print('GPU device' if tf.config.list_physical_devices('GPU') else 'CPU device')
    model = Tomek(dt=dt)
    model.set_use_rush_larsen(not args.forward_euler)
    model.set_parameter('celltype', np.array([[0.0], [1.0], [2.0]]))
    U = tf.Variable(np.full((3, 1), model.get_parameter('V_init')), dtype=tf.float64, name='U')
    model.initialize_state_variables(U)

    steps_per_ms   = int(round(1.0 / dt))
    steps_per_beat = int(round(args.bcl / dt))
    stim_first     = steps_per_ms
    stim_last      = 2 * steps_per_ms

    # differentiate() is already compiled with XLA; compiling the whole 1 ms loop
    # as well removes the Python dispatch of every step, which dominates for a
    # single cell (about 90 times faster than uncompiled, same result to 1.8e-12 mV).
    @tf.function(jit_compile=not args.no_xla)
    def advance(first_step: tf.Tensor):
        for i in tf.range(steps_per_ms):
            k = (first_step + i) % steps_per_beat
            if k >= stim_first and k < stim_last:
                U.assign_add(dt * args.stim * tf.ones_like(U))
            U.assign_add(dt * model.differentiate(U))

    trace_t, trace_V, trace_Cai = [], [], []
    t0   = time.time()
    step = 0
    for beat in range(args.beats):
        last = (beat == args.beats - 1)
        for _ms in range(int(round(args.bcl))):
            if last:
                trace_t.append((step % steps_per_beat) * dt)
                trace_V.append(U.numpy().ravel().copy())
                trace_Cai.append(model.get_state_variables()['Cai'].copy())
            advance(tf.constant(step, dtype=tf.int32))
            step += steps_per_ms
        print('beat {}/{} ({:.1f} s)'.format(beat + 1, args.beats, time.time() - t0))
    print('device: {}'.format(U.device))

    time_ms = np.array(trace_t)
    V       = np.array(trace_V)
    Cai     = 1.0e3 * np.array(trace_Cai)                  # uM, as the reference reports it
    output_dir = os.path.dirname(os.path.abspath(__file__))
    scheme  = 'fe' if args.forward_euler else 'rl'
    for index, name in enumerate(CELL_TYPES):
        fname = os.path.join(output_dir, 'tomek_{}_{}_dt{}.npy'.format(name, scheme, dt))
        np.save(fname, np.column_stack([time_ms, V[:, index], Cai[:, index]]))
        line = '{:5s}: peak Vm {:.3f} mV, APD90 {:.0f} ms, Cai peak {:.5f} uM'.format(
            name, V[:, index].max(), apd90(time_ms, V[:, index]), Cai[:, index].max())
        if len(args.reference) > 0:
            folder  = os.path.join(args.reference, 'ct{}'.format(index))
            first   = (args.beats - 1) * len(time_ms)
            V_ref   = np.fromfile(os.path.join(folder, 'BENCH_REG.Vm.bin'))[first:first + len(time_ms)]
            Cai_ref = np.fromfile(os.path.join(folder, 'BENCH_REG_Tomek.Cai.bin'))[first:first + len(time_ms)]
            line += ' | reference: peak Vm {:.3f} mV, APD90 {:.0f} ms, Cai peak {:.5f} uM'.format(
                V_ref.max(), apd90(time_ms, V_ref), Cai_ref.max())
        print(line)


if __name__ == '__main__':
    main()

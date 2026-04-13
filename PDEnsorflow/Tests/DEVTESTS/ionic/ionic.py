#!/usr/bin/env python
"""
    A TensorFlow-based Cardiac Electrophysiology Modeler

    Copyright 2022-2023 Cesare Corrado (cesare.corrado@kcl.ac.uk)

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
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
import tensorflow as tf
tf.config.run_functions_eagerly(True)

if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))

from gpuSolve.ionic.fenton4v import Fenton4v
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.ionic.courtemanche_ramirez_nattel import CourtemancheRamirezNattel
from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov

# ---- Configuration ----
dt              = 0.02     # time step (ms)
BCL             = 1000.0   # basic cycle length (ms)
n_beats         = 100      # number of pacing beats
stim_intensity  = 60.0     # stimulus current amplitude
stim_duration   = 1.0      # stimulus duration (ms)
record_interval = 1.0      # recording interval (ms)

steps_per_beat  = int(BCL / dt)
stim_steps      = int(stim_duration / dt)
record_every    = int(record_interval / dt)

output_dir = os.path.dirname(os.path.abspath(__file__))


def run_pacing(model, model_name, V_init):
    """Run the 100-beat pacing protocol for a given ionic model
       and save the last beat trace as a numpy array [time, U].
    """
    print('=== %s ===' % model_name)

    U = tf.Variable(tf.fill([1, 1], tf.constant(V_init, dtype=tf.float32)), name='U')
    model.initialize_state_variables(U)

    trace_time = []
    trace_U    = []
    last_beat  = n_beats - 1

    t0 = time.time()
    for beat in range(n_beats):
        recording = (beat == last_beat)
        for step in range(steps_per_beat):
            # Record the state before update during the last beat
            if recording and step % record_every == 0:
                trace_time.append(step * dt)
                trace_U.append(float(U.numpy().flat[0]))

            # Compute ionic derivative and advance
            dU = model.differentiate(U)
            if step < stim_steps:
                U.assign(U + dt * (dU + stim_intensity))
            else:
                U.assign(U + dt * dU)

        # Record end-of-beat state (t = BCL)
        if recording:
            trace_time.append(BCL)
            trace_U.append(float(U.numpy().flat[0]))

        if (beat + 1) % 10 == 0:
            elapsed = time.time() - t0
            print('  Beat %d/%d  (elapsed: %.1fs)' % (beat + 1, n_beats, elapsed))

    data = np.column_stack([np.array(trace_time), np.array(trace_U)])
    fname = os.path.join(output_dir, '%s.npy' % model_name)
    np.save(fname, data)

    total = time.time() - t0
    print('  Saved %s' % fname)
    print('  Shape: %s' % str(data.shape))
    print('  V_min=%.2f  V_max=%.2f  V_end=%.2f' % (data[:,1].min(), data[:,1].max(), data[:,1][-1]))
    print('  Total time: %.1fs' % total)
    print()


if __name__ == '__main__':

    # ---- Fenton 4v (dimensional by default: vmin=-80, vmax=20) ----
    f4v = Fenton4v(dt=dt)
    run_pacing(f4v, 'fenton4v', V_init=-80.0)

    # ---- Modified Mitchell-Schaeffer 2v (dimensional by default: vmin=-80, vmax=20) ----
    mms = ModifiedMS2v(dt=dt)
    run_pacing(mms, 'mms2v', V_init=-80.0)

    # ---- Courtemanche-Ramirez-Nattel (human atrial) ----
    crn = CourtemancheRamirezNattel(dt=dt)
    run_pacing(crn, 'courtemanche_ramirez_nattel', V_init=-81.2)

    # ---- Ten Tusscher-Panfilov (human ventricular, EPI) ----
    ttp = TenTusscherPanfilov(dt=dt, cell_type='EPI')
    run_pacing(ttp, 'ten_tusscher_panfilov', V_init=-86.2)

    print('All models completed successfully.')

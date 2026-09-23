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

    Single cell with the DeBruin-Krassowska (1998) electroporation plugin,
    compared step by step with the reference single-cell tool (bench). The step
    follows bench's order: the state is recorded, the stimulus is added to V,
    the model and then the plugin are advanced with that V, then
    V -= dt*(Iion + I_ep).

    Two parent models:
      * --parent tomek (default): the Tomek (ToR-ORd) model, with forward-Euler
        gates, the reference's own scheme, at rest (--stim 0) or under a shock
        (--stim 1000). Above +222 mV (at dt = 0.01 ms) forward Euler on
        Tomek's IKr Markov chain is unstable in both codes: its states
        saw-tooth on the [0, 1] clamps. Both codes compute the same clamped
        iteration, so they still agree, but that part of the trajectory is a
        numerical artifact.
      * --parent passive: a passive membrane, Iion = (V - Vrest)/Rm with
        Rm = 10 kOhm cm^2, the reference's passive model (Plonsey), written
        out below. It stays stable at any potential, so a shock (--stim 1000,
        up to about +470 mV) tests the plugin alone.

    With --reference, V and the pore density n are compared at every step with
    the bench dumps written, in that folder, by

        bench --imp Tomek --plug-in Electroporation_DeBruinKrassowska98 \
              --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v
        bench --imp Plonsey --imp-par "Vrest=<vrest>" \
              --plug-in Electroporation_DeBruinKrassowska98 \
              --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v

    (stimulus from t = 1 ms for 1 ms, the bench default). With the passive
    parent and --vrest 0 the plugin starts at V = 0 exactly, where its pore
    conductance is 0/0: bench turns NaN from the first step, this code uses the
    finite limit.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import time
import numpy as np
import tensorflow as tf

from gpuSolve.ionic.ionicmodel import IonicModel
from gpuSolve.ionic.tomek import Tomek
from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.ionic.plugins.electroporation_debruin_krassowska98 import ElectroporationDeBruinKrassowska98

PLUGIN_N = 'ElectroporationDeBruinKrassowska98.n'


class PassiveMembrane(IonicModel):
    """
        Passive membrane, Iion = (V - Vrest)/Rm (uA/uF), with Rm = 10 kOhm cm^2.
        Plonsey R. The formulation of bioelectric source-field relationships in
        terms of surface discontinuities. J Franklin Inst 1974;297:317-324.
        A test parent for the plugin: it has no state and is stable at any
        potential. Not part of the library.
    """

    def __init__(self, dt: float = 0.0, n_nodes: int = 0):
        super().__init__(dt, n_nodes)
        self._Vrest  : float = 0.0     # mV
        self._Rm     : float = 10.0    # kOhm cm^2
        self._V_init : float = 0.0     # mV

    def set_vrest(self, vrest: float):
        """ set_vrest(vrest) sets the resting potential, which is also the initial one """
        self._Vrest  = vrest
        self._V_init = vrest

    def differentiate(self, U: tf.Variable) -> tf.Variable:
        """ differentiate(U) returns dU = -(U - Vrest)/Rm """
        return(-(1.0 / self._Rm) * (U - self._Vrest))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tomek + electroporation plugin, single cell')
    parser.add_argument('--dt', type=float, default=0.01, help='time step (ms)')
    parser.add_argument('--duration', type=float, default=50.0, help='simulated time (ms)')
    parser.add_argument('--stim', type=float, default=0.0,
                        help='stimulus (uA/uF), from t = 1 ms for 1 ms')
    parser.add_argument('--parent', choices=('tomek', 'passive'), default='tomek',
                        help='the cell model the plugin is attached to')
    parser.add_argument('--vrest', type=float, default=-80.0,
                        help='resting potential of the passive parent (mV)')
    parser.add_argument('--reference', default='', help='folder with the bench -v dumps')
    return(parser.parse_args())


def main():
    args = parse_arguments()
    dt   = args.dt
    print('GPU device' if tf.config.list_physical_devices('GPU') else 'CPU device')
    if args.parent == 'tomek':
        parent = Tomek(dt=dt)
        parent.set_use_rush_larsen(False)
    else:
        parent = PassiveMembrane(dt=dt)
        parent.set_vrest(args.vrest)
    model = IonicModelWithPlugins(dt=dt)
    model.set_model(parent)
    model.add_plugin(ElectroporationDeBruinKrassowska98(dt=dt))
    U = tf.Variable([[parent.get_parameter('V_init')]], dtype=tf.float64, name='U')
    model.initialize_state_variables(U)
    n_state = model.state_variable(PLUGIN_N)

    nsteps     = int(round(args.duration / dt))
    stim_first = int(round(1.0 / dt))
    stim_last  = int(round(2.0 / dt))
    trace_V, trace_n = [], []
    t0 = time.time()
    for step in range(nsteps):
        trace_V.append(U.numpy().item())
        trace_n.append(n_state.numpy().item())
        if stim_first <= step < stim_last:
            U.assign_add(dt * args.stim * tf.ones_like(U))
        U.assign_add(dt * model.differentiate(U))
    trace_V.append(U.numpy().item())
    trace_n.append(n_state.numpy().item())
    print('{} steps in {:.1f} s, device: {}'.format(nsteps, time.time() - t0, U.device))

    V = np.array(trace_V)
    n = np.array(trace_n)
    print('{}: V in [{:.3f}, {:.3f}] mV, final {:.6f} mV; n in [{:.6e}, {:.6e}] cm^-2'.format(
        model.model_name(), V.min(), V.max(), V[-1], n.min(), n.max()))
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(output_dir, 'electroporation_{}_stim{}_dt{}.npy'.format(args.parent, args.stim, dt))
    np.save(fname, np.column_stack([dt * np.arange(V.size), V, n]))

    if len(args.reference) > 0:
        V_ref = np.fromfile(os.path.join(args.reference, 'BENCH_REG.Vm.bin'))
        n_ref = np.fromfile(os.path.join(args.reference,
                                         'BENCH_REG_Electroporation_DeBruinKrassowska98.n.bin'))
        count = min(V.size, V_ref.size)
        if count != V.size:
            print('warning: the reference holds {} samples, this run {}'.format(V_ref.size, V.size))
        dV = np.abs(V[:count] - V_ref[:count])
        dn = np.abs(n[:count] - n_ref[:count]) / np.abs(n_ref[:count])
        print('reference: V in [{:.3f}, {:.3f}] mV, final {:.6f} mV'.format(
            V_ref[:count].min(), V_ref[:count].max(), V_ref[count - 1]))
        print('max |V - V_ref| = {:.3e} mV (at t = {:.2f} ms); max |n - n_ref|/n_ref = {:.3e}'.format(
            dV.max(), dt * int(np.argmax(dV)), dn.max()))


if __name__ == '__main__':
    main()

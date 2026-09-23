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

    Single cell with the Ashihara-Trayanova (2004) outward-current plugin,
    compared step by step with the reference single-cell tool (bench). The step
    follows bench's order: the potential is recorded, the stimulus is added to
    V, the model and then the plugin are evaluated with that V, then
    V -= dt*(Iion + Ia).

    Two parent models:
      * --parent passive (default): a passive membrane, Iion = (V - Vrest)/Rm
        with Rm = 10 kOhm cm^2, the reference's passive model (Plonsey),
        written out below. It is stable at any potential, so a shock tests the
        plugin alone.
      * --parent tomek: the Tomek (ToR-ORd) model with forward-Euler gates, the
        reference's own scheme.

    --reference-form selects the plugin's reference lower branch
    (set_use_reference_form(True)), the one bench computes; without it the
    plugin uses Cheng et al.'s lower branch and differs from bench by design.
    With --reference, V is compared at every step with the bench dump written,
    in that folder, by

        bench --imp Plonsey --imp-par "Vrest=<vrest>" \
              --plug-in Defib_AshiharaTrayanova \
              --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v
        bench --imp Tomek --plug-in Defib_AshiharaTrayanova \
              --stim-curr <stim> --duration <duration> --dt <dt> --dt-out <dt> -v

    (stimulus from t = 1 ms for 1 ms, the bench default).
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
from gpuSolve.ionic.plugins.defib_ashihara_trayanova import DefibAshiharaTrayanova


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
    parser = argparse.ArgumentParser(description='Ashihara-Trayanova outward-current plugin, single cell')
    parser.add_argument('--dt', type=float, default=0.01, help='time step (ms)')
    parser.add_argument('--duration', type=float, default=5.0, help='simulated time (ms)')
    parser.add_argument('--stim', type=float, default=400.0,
                        help='stimulus (uA/uF), from t = 1 ms for 1 ms')
    parser.add_argument('--parent', choices=('passive', 'tomek'), default='passive',
                        help='the cell model the plugin is attached to')
    parser.add_argument('--vrest', type=float, default=-80.0,
                        help='resting potential of the passive parent (mV)')
    parser.add_argument('--reference-form', action='store_true',
                        help='use the lower branch of the reference model description')
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
    plugin = DefibAshiharaTrayanova(dt=dt)
    plugin.set_use_reference_form(args.reference_form)
    model = IonicModelWithPlugins(dt=dt)
    model.set_model(parent)
    model.add_plugin(plugin)
    U = tf.Variable([[parent.get_parameter('V_init')]], dtype=tf.float64, name='U')
    model.initialize_state_variables(U)

    nsteps     = int(round(args.duration / dt))
    stim_first = int(round(1.0 / dt))
    stim_last  = int(round(2.0 / dt))
    trace_V = []
    t0 = time.time()
    for step in range(nsteps):
        trace_V.append(U.numpy().item())
        if stim_first <= step < stim_last:
            U.assign_add(dt * args.stim * tf.ones_like(U))
        U.assign_add(dt * model.differentiate(U))
    trace_V.append(U.numpy().item())
    print('{} steps in {:.1f} s, device: {}'.format(nsteps, time.time() - t0, U.device))

    V = np.array(trace_V)
    form = 'reference' if args.reference_form else 'Cheng et al.'
    print('{} ({} lower branch): V in [{:.3f}, {:.3f}] mV, final {:.6f} mV'.format(
        model.model_name(), form, V.min(), V.max(), V[-1]))
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(output_dir, 'defib_{}_stim{}_dt{}{}.npy'.format(
        args.parent, args.stim, dt, '_refform' if args.reference_form else ''))
    np.save(fname, np.column_stack([dt * np.arange(V.size), V]))

    if len(args.reference) > 0:
        V_ref = np.fromfile(os.path.join(args.reference, 'BENCH_REG.Vm.bin'))
        count = min(V.size, V_ref.size)
        if count != V.size:
            print('warning: the reference holds {} samples, this run {}'.format(V_ref.size, V.size))
        dV = np.abs(V[:count] - V_ref[:count])
        print('reference: V in [{:.3f}, {:.3f}] mV, final {:.6f} mV'.format(
            V_ref[:count].min(), V_ref[:count].max(), V_ref[count - 1]))
        print('max |V - V_ref| = {:.3e} mV (at t = {:.2f} ms)'.format(dV.max(), dt * int(np.argmax(dV))))


if __name__ == '__main__':
    main()

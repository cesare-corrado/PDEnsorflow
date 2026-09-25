#!/usr/bin/env python
"""
    LAT monitoring example: a planar wave on the triangulated square, with a
    LatDetector recording local activation times, the equivalent of the
    reference simulator's LAT detection (user guide 22.2/22.8).

    The left edge is raised above threshold at t = 0 (the initial condition), so
    a wave sweeps across the strip in +x. A threshold-crossing detector at
    -10 mV (upstroke) records the activation time of every node; because the
    front is planar, the activation time should grow monotonically with x, and
    the map divided by x gives the conduction velocity.

    Run it with the Claude_testing interpreter (see CLAUDE.md 1.1). It prints the
    device banner, the number of activations, the activation-time range and the
    apparent conduction velocity, and writes vm_act.dat.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print('GPU device')
else:
    print('CPU device')
print('Tensorflow version is: {0}'.format(tf.__version__))

from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import LatDetector


def dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """ empty function for mass properties """
    return(None)


def sigmaTens(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties) -> np.ndarray:
    """ function to evaluate the diffusion tensor """
    fib     = domain.Fibres()[iElem, :]
    rID     = domain.Elems()[elemtype][iElem, -1]
    sigma_l = matprop.ElementProperty('sigma_l', elemtype, iElem, rID)
    sigma_t = matprop.ElementProperty('sigma_t', elemtype, iElem, rID)
    Sigma   = sigma_t * np.eye(3)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii, jj] = Sigma[ii, jj] + (sigma_l - sigma_t) * fib[ii] * fib[jj]
    return(Sigma)


if __name__ == '__main__':
    dt      = 0.1
    diffusl = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    diffust = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    tin     = {1: 0.15,  2: 0.15,  3: 0.15,  4: 0.15}
    tout    = {1: 1.5,   2: 1.5,   3: 1.5,   4: 1.5}
    topen   = {1: 105,   2: 105,   3: 105,   4: 105}
    tclose  = {1: 120,   2: 120,   3: 120,   4: 120}

    config = {
        'mesh_file_name': os.path.join('..', '..', 'data', 'triangulated_square.pkl'),
        'use_renumbering': True,
        'dt': dt,
        'dt_per_plot': int(1.0 / dt),
        'Tend': 60.0,
    }

    ionic = ModifiedMS2v(dt=dt)
    model = MonodomainSolver(ionic, config)
    model.add_element_material_property('sigma_l', 'region', diffusl)
    model.add_element_material_property('sigma_t', 'region', diffust)
    model.add_nodal_material_property('tau_in', 'region', tin)
    model.add_nodal_material_property('tau_out', 'region', tout)
    model.add_nodal_material_property('tau_open', 'region', topen)
    model.add_nodal_material_property('tau_close', 'region', tclose)
    model.add_nodal_material_property('u_gate', 'uniform', 0.1)
    model.add_nodal_material_property('u_crit', 'uniform', 0.1)
    model.add_material_function('mass', dfmass)
    model.add_material_function('stiffness', sigmaTens)
    model.assign_nodal_properties()
    model.assemble_matrices()

    pts = model.domain().Pts()
    Lx  = pts[:, 0].max()
    # raise the left edge above threshold so a planar wave sweeps in +x.
    U0  = np.where(pts[:, 0] < 0.05 * Lx, 20.0, -80.0).astype(np.float32)
    model.set_initial_condition(U0)
    model.solver().set_maxiter(pts.shape[0] // 2)
    model.finalize_for_run()

    # first-activation detector, upstroke crossing at -10 mV, one time per node.
    lat = LatDetector({'method': 1, 'threshold': -10.0, 'mode': 0, 'all': 0,
                       'dt': dt, 'ID': 'vm_act'})

    then  = time.time()
    ctime = 0.0
    for i in range(model.nt()):
        ctime += model.dt()
        model.step(ctime)
        # feed the potential in the user's node order (U(), not the solver order)
        lat.check(model.U(), ctime)
    elapsed = time.time() - then
    print('solution, elapsed: %f sec' % elapsed)

    tm         = lat.activation_times()
    activated  = tm >= 0.0
    nact       = int(np.count_nonzero(activated))
    fname      = lat.write('.')
    print('activated nodes: %d / %d' % (nact, tm.shape[0]))
    print('LAT range: [%.3f, %.3f] ms' % (tm[activated].min(), tm[activated].max()))
    # planar front: fit LAT vs x on the activated nodes; CV = 1 / slope.
    x    = pts[activated, 0]
    slope, _ = np.polyfit(x, tm[activated], 1)
    print('apparent conduction velocity: %.4f (space units)/ms' % (1.0 / slope))
    print('wrote %s' % fname)

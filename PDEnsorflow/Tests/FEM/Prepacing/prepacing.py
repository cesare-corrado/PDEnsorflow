#!/usr/bin/env python
"""
    Prepacing example: condition the cell models of a tissue simulation before
    the run starts, the equivalent of the reference simulator's prepacing_*
    parameters (user guide 22.3 - 22.7).

    The two features compose, and this example shows the pair:

      1. a first, unconditioned run sweeps a planar wave across the strip and a
         LatDetector records when every node activates (this is the LAT example
         under Tests/FEM/LAT, run here so the example stands alone);
      2. a second run is prepaced from that activation map. One cell per mesh
         region is paced for `beats` beats at `bcl`, and every node is handed
         the state of its region's cell at the moment that matches its own place
         in the activation sequence: early nodes take a later state, late nodes
         an earlier one.

    What to look for in the output: after prepacing, the potential is no longer
    the same number at every node. The spread across the mesh is the point of
    the exercise, because it is the phase offset that a constant initial
    condition cannot express. The printed activation times of the prepaced run
    are then compared with the unconditioned ones.

    Run it with the Claude_testing interpreter (see CLAUDE.md 1.1).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
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
from gpuSolve.physics import Prepacer


DT      = 0.1                                     # ms
TEND    = 60.0                                    # ms
BCL     = 250.0                                   # ms, prepacing cycle length
BEATS   = 5                                       # prepacing beats
STIMDUR = 1.0                                     # ms
STIMSTR = 60.0                                    # uA/uF


def dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """ empty function for mass properties """
    return(None)


def sigmaTens(elemtype: str, iElem: int, domain: Triangulation,
              matprop: MaterialProperties) -> np.ndarray:
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


def build_model() -> MonodomainSolver:
    """ builds the monodomain problem, assembled and started from the planar
        wave initial condition, but not finalized: prepacing runs at that point
        of the setup, while the nodal quantities are still in the user's order
    """
    diffusl = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    diffust = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    tin     = {1: 0.15,  2: 0.15,  3: 0.15,  4: 0.15}
    tout    = {1: 1.5,   2: 1.5,   3: 1.5,   4: 1.5}
    topen   = {1: 105,   2: 105,   3: 105,   4: 105}
    tclose  = {1: 120,   2: 120,   3: 120,   4: 120}

    config = {
        'mesh_file_name': os.path.join('..', '..', 'data', 'triangulated_square.pkl'),
        'use_renumbering': True,
        'dt': DT,
        'dt_per_plot': int(1.0 / DT),
        'Tend': TEND,
    }
    model = MonodomainSolver(ModifiedMS2v(dt=DT), config)
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

    model.set_initial_condition()
    pts = model.domain().Pts()
    Lx  = pts[:, 0].max()
    # The wave is launched by a stimulus on the left edge, not by an initial
    # condition: prepacing rewrites the potential at every node, so a
    # depolarised block written into the initial condition would simply be
    # erased by it. The reference has the same semantics, and paces with a
    # stimulus for the same reason.
    model.add_stimulus(pts[:, 0] < 0.05 * Lx,
                       {'tstart': 0.0, 'nstim': 1, 'period': 1.0,
                        'duration': 2.0, 'intensity': 60.0, 'name': 'S1'})
    model.solver().set_maxiter(pts.shape[0] // 2)
    return(model)


def run(model: MonodomainSolver, label: str) -> np.ndarray:
    """ advances the solution to Tend with a first-activation detector attached,
        and returns the per-node activation times
    """
    model.finalize_for_run()
    lat = LatDetector({'method': 1, 'threshold': -10.0, 'mode': 0, 'all': 0,
                       'dt': DT, 'ID': 'vm_act'})
    then  = time.time()
    ctime = 0.0
    for _istep in range(model.nt()):
        ctime += model.dt()
        model.step(ctime)
        # the detector is fed in the user's node order (U(), not solver order)
        lat.check(model.U(), ctime)
    print('%s: solution, elapsed %.3f sec' % (label, time.time() - then))
    return(lat.activation_times())


def report(label: str, tm: np.ndarray):
    activated = tm >= 0.0
    print('%s: activated %d / %d nodes, LAT in [%.3f, %.3f] ms'
          % (label, int(np.count_nonzero(activated)), tm.shape[0],
             tm[activated].min(), tm[activated].max()))


if __name__ == '__main__':
    # ---- 1. the unconditioned run, which produces the activation map --------
    print('\n--- reference run (no prepacing) ---')
    reference = build_model()
    tm_ref    = run(reference, 'reference')
    report('reference', tm_ref)

    # ---- 2. the same problem, prepaced from that map ------------------------
    print('\n--- prepaced run ---')
    model    = build_model()
    resting  = np.reshape(model.checkpoint()['Vm'], (-1,)).copy()
    prepacer = Prepacer({'bcl': BCL, 'beats': BEATS, 'stimdur': STIMDUR,
                         'stimstr': STIMSTR, 'dt': DT, 'tend': TEND})
    prepacer.set_model_factory(lambda: ModifiedMS2v(dt=DT))
    prepacer.set_lats(tm_ref)
    prepacer.prepace(model)

    checkpoint  = model.checkpoint()
    conditioned = np.reshape(checkpoint['Vm'], (-1,))
    print('paced %d cell(s) over %d steps in %.2f s (%s)'
          % (prepacer.ncells(), prepacer.nsteps(), prepacer.elapsed(),
             'one per node' if prepacer.per_node() else 'one per region'))
    print('save times span [%.3f, %.3f] ms of the prepacing train'
          % (prepacer.save_times().min(), prepacer.save_times().max()))
    print('resting  Vm: %d distinct value(s), spread %.6f mV'
          % (len(np.unique(np.round(resting, 6))), np.ptp(resting)))
    print('prepaced Vm: %d distinct value(s), spread %.6f mV'
          % (len(np.unique(np.round(conditioned, 6))), np.ptp(conditioned)))
    # With a 60 ms activation spread against a 250 ms cycle length, every save
    # time falls in diastole, where the potential of this model is flat: the
    # phase offset is carried by the gating variable instead. That is why
    # prepacing distributes the whole state and not only the potential, and why
    # the potential alone is the wrong thing to look at to see it working.
    for name, values in checkpoint['state_variables'].items():
        values = np.asarray(values)
        print('prepaced %s: %d distinct value(s), spread %.6e'
              % (name, len(np.unique(np.round(values, 9))), np.ptp(values)))

    tm_pre = run(model, 'prepaced')
    report('prepaced', tm_pre)

    both = np.logical_and(tm_ref >= 0.0, tm_pre >= 0.0)
    print('\nactivation time shift on the %d nodes both runs activate: '
          'mean %+.4f ms, max |shift| %.4f ms'
          % (int(np.count_nonzero(both)),
             float(np.mean(tm_pre[both] - tm_ref[both])),
             float(np.max(np.abs(tm_pre[both] - tm_ref[both])))))

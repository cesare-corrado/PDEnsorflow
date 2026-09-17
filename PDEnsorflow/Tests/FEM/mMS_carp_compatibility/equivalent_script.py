#!/usr/bin/env python
"""
    The same simulation as parameters.par, written against the Python API.

    The point of this script is to show that the two ways of driving gpuSolve
    are one solver: it sets up exactly what the parameter file asks for, in the
    units the library works in, and writes its result next to the one the
    front end produced so the two can be compared node by node
    (see check_result.py).

    Note where the unit conversion has already happened. The parameter file
    carries g = 0.01 S/m and cellSurfVolRatio = 1, and the mapping turns that
    into sigma = 1e5 * g = 1000 um^2/ms with beta = 1; this script registers
    those converted values directly, because the Python API has always taken
    the diffusion coefficient in mesh units.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import numpy as np
import tensorflow as tf

from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import no_mass_property
from gpuSolve.IO.writers import IGBWriter


if __name__ == '__main__':
    if tf.config.list_physical_devices('GPU'):
        print('GPU device')
    else:
        print('CPU device')
    print('Tensorflow version is: {0}'.format(tf.__version__))

    dt     = 0.1                       # ms, the parameter file's dt = 100 us
    Tend   = 250.0
    SIGMA  = 1000.0                    # um^2/ms, from 0.01 S/m with beta = 1
    BETA   = 1.0
    OUTDIR = 'OUT_mMS_api'

    tags     = (1, 2, 3, 4)
    sigma_l  = {tag: SIGMA for tag in tags}
    sigma_t  = {tag: SIGMA for tag in tags}
    beta     = {tag: BETA for tag in tags}
    tau_in   = {tag: 0.15 for tag in tags}
    tau_out  = {tag: 1.5 for tag in tags}
    tau_open = {tag: 105.0 for tag in tags}
    tau_close = {1: 120.0, 2: 120.0, 3: 60.0, 4: 120.0}
    u_gate   = {tag: 0.1 for tag in tags}
    u_crit   = {tag: 0.1 for tag in tags}

    config = {'mesh_file_name': 'square',       # the micrometre mesh, no suffix
              'use_renumbering': True,
              'dt': dt,
              'dt_per_plot': int(round(1.0 / dt)),
              'Tend': Tend}

    cfgstim = {'tstart': 0.0, 'nstim': 1, 'period': Tend,
               'duration': 2.0, 'intensity': 60.0, 'name': 'S1'}

    ionic = ModifiedMS2v(dt=dt)
    model = MonodomainSolver(ionic, config)
    model.add_element_material_property('sigma_l', 'region', sigma_l)
    model.add_element_material_property('sigma_t', 'region', sigma_t)
    model.add_element_material_property('beta', 'region', beta)
    model.add_nodal_material_property('tau_in', 'region', tau_in)
    model.add_nodal_material_property('tau_out', 'region', tau_out)
    model.add_nodal_material_property('tau_open', 'region', tau_open)
    model.add_nodal_material_property('tau_close', 'region', tau_close)
    model.add_nodal_material_property('u_gate', 'region', u_gate)
    model.add_nodal_material_property('u_crit', 'region', u_crit)
    model.add_material_function('mass', no_mass_property)
    model.add_material_function('stiffness', conductivity_tensor)
    model.assign_nodal_properties()
    model.assemble_matrices()
    model.solver().set_toll(1.0e-8)
    model.solver().set_toll_rel(0.0)
    model.solver().set_maxiter(100)

    model.set_initial_condition()
    S1 = model.domain().Pts()[:, 0] <= 500.0
    model.add_stimulus(S1, cfgstim)
    S1 = None

    os.makedirs(OUTDIR, exist_ok=True)
    dt_per_plot = model.dt_per_plot()
    nframes = 1 + (model.nt() + dt_per_plot - 1) // dt_per_plot
    im = IGBWriter({'fname': os.path.join(OUTDIR, 'vm.igb'),
                    'Tend': model.Tend(),
                    'nt': nframes,
                    'nx': model.domain().Pts().shape[0]})
    model.finalize_for_run()
    im.imshow(model.U())

    then  = time.time()
    ctime = 0.0
    for i in range(model.nt()):
        ctime += model.dt()
        model.step(ctime)
        if i % dt_per_plot == 0:
            im.imshow(model.U())
    print('solution, elapsed: %f sec' % (time.time() - then))
    im.wait()
    im = None
    print('results written to {}'.format(OUTDIR))

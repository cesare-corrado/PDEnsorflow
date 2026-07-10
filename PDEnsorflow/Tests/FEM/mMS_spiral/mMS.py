#!/usr/bin/env python
"""
    A TensorFlow-based 3D Cardiac Electrophysiology Modeler

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
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
from gpuSolve.IO.writers import IGBWriter
import tensorflow as tf
tf.config.run_functions_eagerly(True)
if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))

from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.physics import MonodomainSolver


def dfmass(elemtype:str, iElem:int,domain:Triangulation,matprop:MaterialProperties):
    """ empty function for mass properties"""
    return(None)

def sigmaTens(elemtype:str, iElem:int,domain:Triangulation,matprop:MaterialProperties) -> np.ndarray :
    """ function to evaluate the diffusion tensor """
    fib   = domain.Fibres()[iElem,:]
    rID   = domain.Elems()[elemtype][iElem,-1]
    sigma_l = matprop.ElementProperty('sigma_l',elemtype,iElem,rID)
    sigma_t = matprop.ElementProperty('sigma_t',elemtype,iElem,rID)
    Sigma = sigma_t *np.eye(3)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii,jj] = Sigma[ii,jj]+ (sigma_l-sigma_t)*fib[ii]*fib[jj]
    return(Sigma)


if __name__=='__main__':
    dt      = 0.01
    TS2     = 260.0
    vg      = 0.1
    diffusl = {1: 0.175, 2: 0.175, 3: 0.175, 4: 0.175}
    diffust = {1: 0.175, 2: 0.175, 3: 0.175, 4: 0.175}
    tin     = {1: 0.15,  2: 0.15,  3: 0.15,  4: 0.15}
    tout    = {1: 1.5,   2: 1.5,   3: 1.5,   4: 1.5}
    topen   = {1: 105,   2: 105,   3: 105,   4: 105}
    tclose  = {1: 185,   2: 185,   3: 185,   4: 185}

    config  = {
        'mesh_file_name': os.path.join('..','..','data','triangulated_square_fine_mm.pkl'),
        'use_renumbering': True,
        'dt' : dt,
        'dt_per_plot': int(1.0/dt),   #record every ms
        'Tend': 5000
        }

    cfgstim1 = {'tstart': 0.0,
                       'nstim': 1,
                       'period':100,
                       'duration':np.max([0.4,dt]),
                       'intensity':60.0,
                       'name':'crossstim'
              }

    cfgstim2 = {'tstart': TS2,
                       'nstim': 1,
                       'period':100,
                       'duration':np.max([0.4,dt]),
                       'intensity':60.0,
                       'name':'crossstim'
              }

    ionic = ModifiedMS2v(dt=dt)
    model = MonodomainSolver(ionic, config)
    # Define the materials
    model.add_element_material_property('sigma_l','region',diffusl)
    model.add_element_material_property('sigma_t','region',diffust)
    model.add_nodal_material_property('tau_in','region',tin)
    model.add_nodal_material_property('tau_out','region',tout)
    model.add_nodal_material_property('tau_open','region',topen)
    model.add_nodal_material_property('tau_close','region',tclose)
    model.add_nodal_material_property('u_gate','uniform',vg)
    model.add_nodal_material_property('u_crit','uniform',vg)

    model.add_material_function('mass',dfmass)
    model.add_material_function('stiffness',sigmaTens)
    model.assign_nodal_properties()
    model.assemble_matrices()
    Lx = model.domain().Pts()[:,0].max()
    Ly = model.domain().Pts()[:,1].max()
    S1 = model.domain().Pts()[:,0]<0.05*Lx
    S2 = np.logical_and(model.domain().Pts()[:,0]<Lx,model.domain().Pts()[:,1]<0.5*Ly)
    # Set the initial condition to U=-80,H=1 everywhere
    model.set_initial_condition()
    model.add_stimulus(S1,cfgstim1 )
    model.add_stimulus(S2,cfgstim2 )
    S1 = None
    S2 = None
    model.solver().set_maxiter(model.domain().Pts().shape[0]//2)
    model.domain().exportCarpFormat('square')
    nt = 1 + model.nt()//model.dt_per_plot()
    im = IGBWriter({'fname': 'square.igb',
                    'Tend': model.Tend(),
                     'nt':1+nt,
                     'nx':model.domain().Pts().shape[0]
                     })
    model.finalize_for_run()
    im.imshow(model.U())

    # ---- external time loop (MonodomainSolver advances one step per call) --
    then = time.time()
    ctime = 0.0
    for i in range(model.nt()):
        ctime += model.dt()
        model.step(ctime)
        if im and i % model.dt_per_plot() == 0:
            im.imshow(model.U())
    elapsed = (time.time() - then)
    tf.print('solution, elapsed: %f sec' % elapsed)
    if im:
        im.wait()
    im = None

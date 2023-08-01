#!/usr/bin/env python
"""
    A TensorFlow-based 3D Cardiac Electrophysiology Modeler

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

EAGERMODE=True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
import tensorflow as tf
from time import time
tf.config.run_functions_eagerly(EAGERMODE)
if EAGERMODE:
    print('running in eager mode')
else:
    print('running in graph mode')

if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.matrices.globalMatrices import compute_reverse_cuthill_mckee_indexing
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.linearsolvers.jacobi_precond import JacobiPrecond


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



def assign_material_properties(diff_l : dict = None, diff_t : dict = None) -> MaterialProperties :
    if diff_l is None:
        diff_l = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    if diff_t is None:
        diff_t = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    materials : MaterialProperties = MaterialProperties() 
    materials.add_element_property('sigma_l','region',diff_l)
    materials.add_element_property('sigma_t','region',diff_t)
    materials.add_ud_function('mass',dfmass)
    materials.add_ud_function('stiffness',sigmaTens)
    return(materials)


def assemble_matrix(mesh_file_name : str, use_renumbering : bool = False, dt: float = 0.1) -> dict:
    Domain : Triangulation = Triangulation()
    Domain.readMesh(mesh_file_name)
    materials: MaterialProperties = assign_material_properties()
    connectivity = Domain.mesh_connectivity('True')
    pattern      = compute_coo_pattern(connectivity)

    renumbering = None
    if use_renumbering:
        renumbering = compute_reverse_cuthill_mckee_indexing(pattern)
    lmatr       = {'mass':localMass,'stiffness':localStiffness}
    MATRICES    =  assemble_matrices_dict(lmatr,pattern,Domain,materials,connectivity, renumbering=renumbering)
    MASS        = MATRICES['mass']
    STIFFNESS   = MATRICES['stiffness']
    A           = tf.sparse.add(MASS,tf.sparse.map_values(tf.multiply,STIFFNESS,dt))
    return({'A': A, 'renumbering': renumbering})    





if __name__=='__main__':
    nb_of_tests             = 2000
    mesh_file_name          = os.path.join('..','..','data','triangulated_square.pkl')
    matrix_dict             = assemble_matrix(mesh_file_name, use_renumbering = False, dt= 1.0)
    Amatr                   = matrix_dict['A']
    renumbering             = matrix_dict['renumbering']
    Solver                  = ConjGrad()
    Precond                 = JacobiPrecond()
    Solver.set_matrix(Amatr)
    Solver.set_toll(1.e-7)
    Precond.build_preconditioner(Amatr.indices.numpy()[:,0], Amatr.indices.numpy()[:,1], Amatr.values.numpy(),Amatr.shape[0])
    Solver.set_precond(Precond)
    Solver.set_maxiter(Amatr.shape[0]//2)
    X      = tf.Variable(tf.zeros(shape=(Amatr.shape[0],1), dtype=tf.float32), name="X")
    etimes = np.zeros(nb_of_tests)


    tf.print('solving with CG %d times' % nb_of_tests )
    for jj in tf.range(nb_of_tests):
        Uref = tf.constant(np.random.normal(loc=1.0,size=(Amatr.shape[0],1)).astype(np.float32))
        RHS  = tf.sparse.sparse_dense_matmul(Amatr,Uref)
        X.assign(tf.constant(np.random.normal(loc=np.random.normal(),scale=np.random.uniform(low=1,high=2), size=(Amatr.shape[0],1)).astype(np.float32)) )
        err0  = tf.norm(Uref-X).numpy()        
        Solver.set_X0(X)
        Solver.set_RHS(RHS)
        t0    = time()
        Solver.solve()
        tesp  = time()-t0
        etimes[jj]=tesp
        
        X1    = Solver.X()
        err1  = tf.norm(Uref-X1).numpy()
        #tf.print('test %d' %(1+jj))
        #tf.print('|x-xtrue|_0: {:3.2f}\t|x-xtrue|: {:3.2f}\telapsed: {:3.2f} s'.format(err0,err1,tesp))
        #Solver.summary()
        
    sumstr = '{} repetitions; time, total: {:3.2f}; average: {:4.3g}; std: {:4.3g} '.format(nb_of_tests, etimes.sum(),etimes.mean(),etimes.std())
    tf.print(sumstr)     






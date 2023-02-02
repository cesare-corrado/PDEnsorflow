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

import os
import numpy as np
import time
import tensorflow as tf
tf.config.run_functions_eagerly(True)
if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))

from gpuSolve.ionic.fenton4v import *
from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.force_terms import Stimulus
from gpuSolve.IO.writers import IGBWriter


def dfmass(elemtype, iElem,domain,matprop):
    """ empty function for mass properties"""
    return(None)

def sigmaTens(elemtype, iElem,domain,matprop):
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


class Fenton4vSimple(Fenton4v):
    """
    The monodomain model with Fenton-Cherry ionic model
    """

    def __init__(self, cfgdict=None):
        super().__init__()
        self._mesh_file_name = None
        self._dt             = 0.1
        self._dt_per_plot    = 2
        self._Tend           = 10
        self.__Domain        = Triangulation()
        self.__materials     = MaterialProperties()
        self.__Solver        = ConjGrad()
        self.__Stimulus      = None
        self.__MASS          = None
        self.__U             = None
        self.__V             = None
        self.__W             = None
        self.__S             = None
        self.__ctime         = 0.0
        if cfgdict is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in cfgdict.keys():
                    setattr(self, attribute, cfgdict[attribute[1:]])

        if self._mesh_file_name is not None:
            self.__Domain.readMesh('{}'.format(self._mesh_file_name))

        self.__nt = self._Tend//self._dt 

    def loadMesh(self,fname):
        """ Loads the mesh"""
        self._mesh_file_name = fname
        self.__Domain.readMesh('{}'.format(self._mesh_file_name))


    def add_element_material_property(self,pname,ptype,prop):
        """ adds material properties to elements"""
        self.__materials.add_element_property(pname,ptype,prop)

    def add_material_function(self,fname,fsign):
        """adds functions to map material properties when assembling matrices"""
        self.__materials.add_ud_function(fname,fsign)
    
    def assemble_matrices(self):
        #Compute the domain connectivity
        connectivity = self.__Domain.mesh_connectivity('True')
        # Assemble the matrices
        pattern     = compute_coo_pattern(connectivity)
        lmatr       = {'mass':localMass,'stiffness':localStiffness}
        MATRICES    =  assemble_matrices_dict(lmatr,pattern,self.__Domain,self.__materials,connectivity)
        self.__MASS = MATRICES['mass']
        STIFFNESS   = MATRICES['stiffness']
        A           = tf.sparse.add(self.__MASS,tf.sparse.map_values(tf.multiply,STIFFNESS,self._dt))
        self.__Domain.release_connnectivity()
        self.__Solver.set_matrix(A)

    def set_initial_condition(self,U0:np.ndarray,V0:np.ndarray = None, W0:np.ndarray = None,S0:np.ndarray = None):
        if U0.ndim==1:
            self.__U = tf.Variable(U0[:,np.newaxis], name="U")
        else:
            self.__U = tf.Variable(U0, name="U")
    
        if V0 is not None:
            if V0.ndim==1:
                self.__V = tf.Variable(V0[:,np.newaxis], name="V")
            else:
                self.__V = tf.Variable(V0, name="V")
        else:
            self.__V = tf.Variable(np.full(shape=self.__U.shape,fill_value=1.0), name="V",dtype=U0.dtype)

        if W0 is not None:
            if W0.ndim==1:
                self.__W = tf.Variable(W0[:,np.newaxis], name="W")
            else:
                self.__W = tf.Variable(W0, name="W")
        else:
            self.__W = tf.Variable(np.full(shape=self.__U.shape,fill_value=1.0), name="W",dtype=U0.dtype)

        if S0 is not None:
            if S0.ndim==1:
                self.__S = tf.Variable(S0[:,np.newaxis], name="S")
            else:
                self.__S = tf.Variable(S0, name="S")
        else:
            self.__S = tf.Variable(np.full(shape=self.__U.shape,fill_value=0.0), name="S",dtype=U0.dtype)

    def set_stimulus(self,stimreg,stimprops):
        self.__Stimulus = Stimulus(stimprops)
        self.__Stimulus.set_stimregion(stimreg)    

    @tf.function
    def solve(self,U, V, W, S):
        """ Explicit Euler ODE solver + implicit solver for diffusion"""
        dU, dV, dW, dS = self.differentiate(U, V, W, S)
        self.__Solver.set_X0(U)
        if self.__Stimulus is not None:
            I0 = self.__Stimulus.stimApp(self.__ctime)
            RHS0 = tf.add_n([U,self._dt*dU,self._dt*I0])
        else:
            RHS0 = tf.add(U,self._dt*dU)
        RHS = tf.sparse.sparse_dense_matmul(self.__MASS,RHS0)
        self.__Solver.set_RHS(RHS)
        self.__Solver.solve()
        U1 = self.__Solver.X()
        V1 = V + self._dt * dV
        W1 = W + self._dt * dW
        S1 = S + self._dt * dS
        return(U1, V1, W1, S1)


    @tf.function
    def run(self, im=None):
        """
            Runs the model. 

            Args:
                im: A Screen/writer used to paint/write the transmembrane potential

            Returns:
                None
        """
        then = time.time()
        for i in tf.range(1,self.__nt):
            self.__ctime += self._dt
            U1,V1,W1,S1 = self.solve(self.__U,self.__V,self.__W,self.__S)
            self.__U = U1
            self.__V = V1
            self.__W = W1
            self.__S = S1
            # draw a frame every 1 ms
            if im and i % self._dt_per_plot == 0:
                image = U1.numpy()
                im.imshow(image)
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        if im:
            im.wait()   # wait until the window is closed
    
    def domain(self):
        return(self.__Domain)
    
    def solver(self):
        return(self.__Solver)
    
    def stimulus(self):
        return(self.__Stimulus)

    def U(self):
        return(self.__U)        

    def V(self):
        return(self.__V)        

    def W(self):
        return(self.__W)

    def S(self):
        return(self.__S)

    def nt(self):
        return(self.__nt)

    def dt_per_plot(self):
        return(self._dt_per_plot)

    def ctime(self):
        return(self.__ctime)

if __name__=='__main__':
    dt      = 0.1
    diffusl = 0.001
    diffust = 0.001
    config  = {
        'mesh_file_name': os.path.join('..','..','data','triangulated_square.pkl'),
        'dt' : dt,
        'dt_per_plot': 10,
        'Tend': 1000
        }
    
    cfgstim = {'tstart': 210, 
                       'nstim': 1, 
                       'period':100,
                       'duration':2*dt,
                       'intensity':1.0,
                       'name':'crossstim'
              }
    
    model = Fenton4vSimple(config)
    
    # Define the materials
    model.add_element_material_property('sigma_l','region',{1: diffusl, 2: diffusl, 3: diffusl, 4: diffusl})
    model.add_element_material_property('sigma_t','region',{1: diffust, 2: diffust, 3: diffust, 4: diffust})
    model.add_material_function('mass',dfmass)
    model.add_material_function('stiffness',sigmaTens)
    model.assemble_matrices()
    U0 = 1.0*(model.domain().Pts()[:,0]<2.).astype(np.float32)
    S2 = np.logical_and(model.domain().Pts()[:,0]<10.,model.domain().Pts()[:,1]<5.)
    model.set_initial_condition(U0 )
    model.set_stimulus(S2,cfgstim )
    U0 = None    
    S2 = None
    model.solver().set_maxiter(model.domain().Pts().shape[0]//2)
    model.domain().exportCarpFormat('square')
    nt = 1 + model.nt()//model.dt_per_plot()
    
    im = IGBWriter({'fname': 'square.igb', 
                    'Tend': 100, 
                     'nt':1+nt,
                     'nx':model.domain().Pts().shape[0]
                     })
    im.imshow(model.U().numpy())
    
    model.run(im)
    im = None




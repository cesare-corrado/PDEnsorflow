#!/usr/bin/env python
"""
    A TensorFlow-based 3D Heat Equation Solver

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
from gpuSolve.IO.writers import IGBWriter
import tensorflow as tf
tf.config.run_functions_eagerly(True)
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
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.force_terms import Stimulus


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


class HeatEquation:
    """
    The heat equation
    """

    def __init__(self, cfgdict=None):
        self._mesh_file_name: str               = None
        self._dt: float                         = 0.1
        self._dt_per_plot: int                  = 2
        self._Tend: float                       = 10
        self.__Domain: Triangulation            = Triangulation()
        self.__materials:MaterialProperties     = MaterialProperties()
        self.__Solver:ConjGrad                  = ConjGrad()
        self.__StimulusDict: dict               = None
        self.__MASS                             = None
        self.__X: tf.Variable                   = None
        self.__ctime:float                      = 0.0
        self.__nbstim:int                       = 0
        if cfgdict is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in cfgdict.keys():
                    setattr(self, attribute, cfgdict[attribute[1:]])

        if self._mesh_file_name is not None:
            self.__Domain.readMesh('{}'.format(self._mesh_file_name))

        self.__nt: int = int(self._Tend//self._dt) 

    def loadMesh(self,fname: str):
        """ Loads the mesh"""
        self._mesh_file_name = fname
        self.__Domain.readMesh('{}'.format(self._mesh_file_name))


    def add_element_material_property(self,pname:str,ptype:str,prop:dict):
        """ adds material properties to elements"""
        self.__materials.add_element_property(pname,ptype,prop)

    def add_material_function(self,fname:str,fsign):
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
        self.__Domain.release_connectivity()
        self.__materials.remove_all_element_properties()
        self.__Solver.set_matrix(A)

    def set_initial_condition(self,X0:np.ndarray = None):
        npt = self.__Domain.Pts().shape[0]
        if X0 is not None:
            if X0.ndim==1:
                self.__X = tf.Variable(X0[:,np.newaxis], name="X")
            else:
                self.__X = tf.Variable(X0, name="X")
        else:
            self.__X = tf.Variable(np.full(shape=(npt,1),fill_value=0.0), name="X",dtype=tf.float32)

    def add_stimulus(self,stimreg:np.ndarray,stimprops:dict):
        self.__nbstim +=1
        if self.__StimulusDict is None:
            self.__StimulusDict = {}
        self.__StimulusDict[self.__nbstim] = Stimulus(stimprops)
        self.__StimulusDict[self.__nbstim].set_stimregion(stimreg) 

    @tf.function
    def solve(self,X):
        """ Implicit solver """
        self.__Solver.set_X0(X)
        RHS0 = X
        if self.__StimulusDict is not None:
            for stimname,stimulus in self.__StimulusDict.items():
                I0   = stimulus.stimApp(self.__ctime)
                RHS0 = tf.add(RHS0,self._dt*I0)

        RHS = tf.sparse.sparse_dense_matmul(self.__MASS,RHS0)
        self.__Solver.set_RHS(RHS)
        self.__Solver.solve()
        X1 = self.__Solver.X()
        return(X1)


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
            X1 = self.solve(self.__X)
            self.__X = X1
            # draw a frame every 1 ms
            if im and i % self._dt_per_plot == 0:
                image = X1.numpy()
                im.imshow(image)
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        if im:
            im.wait()   # wait until the window is closed

    def domain(self) -> Triangulation:
        return(self.__Domain)
    
    def solver(self) -> ConjGrad:
        return(self.__Solver)
    
    def stimulus(self) ->dict:
        return(self.__StimulusDict)

    def X(self) -> tf.Variable:
        return(self.__X)        

    def nt(self) -> int:
        return(self.__nt)

    def dt_per_plot(self) -> int:
        return(self._dt_per_plot)

    def ctime(self) -> float:
        return(self.__ctime)

    def Tend(self) ->float:
        return(self._Tend)

if __name__=='__main__':
    dt      = 0.1
    TS2     = 100.0
    diffusl = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    diffust = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}

    config  = {
        'mesh_file_name': os.path.join('..','..','data','triangulated_square.pkl'),
        'dt' : dt,
        'dt_per_plot': int(1.0/dt),   #record every ms
        'Tend': 1000
        }
    
    cfgstim2 = {'tstart': TS2, 
                       'nstim': 3, 
                       'period':200,
                       'duration':np.max([0.4,dt]),
                       'intensity':5.0,
                       'name':'crossstim'
              }
    
    model = HeatEquation(config)
    # Define the materials
    model.add_element_material_property('sigma_l','region',diffusl)
    model.add_element_material_property('sigma_t','region',diffust)
    model.add_material_function('mass',dfmass)
    model.add_material_function('stiffness',sigmaTens)
    model.assemble_matrices()
    Lx = model.domain().Pts()[:,0].max()
    Ly = model.domain().Pts()[:,1].max()
    X0 = 5.0*(model.domain().Pts()[:,0]<2.).astype(np.float32)
    S2 = np.logical_and(model.domain().Pts()[:,0]<Lx,model.domain().Pts()[:,1]<0.5*Ly)
    model.set_initial_condition(X0 )
    model.add_stimulus(S2,cfgstim2 )
    U0 = None    
    S2 = None
    model.solver().set_maxiter(model.domain().Pts().shape[0]//2)
    model.domain().exportCarpFormat('square')
    nt = 1 + model.nt()//model.dt_per_plot()
    im = IGBWriter({'fname': 'square.igb', 
                    'Tend': model.Tend(), 
                     'nt':1+nt,
                     'nx':model.domain().Pts().shape[0]
                     })
    im.imshow(model.X().numpy())
    model.run(im)
    im = None




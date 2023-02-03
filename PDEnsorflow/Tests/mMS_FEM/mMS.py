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

from gpuSolve.ionic.mms2v import *
from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.force_terms import Stimulus
from gpuSolve.IO.writers import IGBWriter


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


class ModifiedMS2vSimple(ModifiedMS2v):
    """
    The monodomain model with modified Mitchell-Shaeffer ionic model
    """

    def __init__(self, cfgdict=None):
        super().__init__()
        self._mesh_file_name: str = None
        self._dt: float           = 0.1
        self._dt_per_plot: int    = 2
        self._Tend: float         = 10
        self.__Domain             = Triangulation()
        self.__materials          = MaterialProperties()
        self.__Solver             = ConjGrad()
        self.__Stimulus           = None
        self.__MASS               = None
        self.__U                  = None
        self.__H                  = None
        self.__ctime              = 0.0
        if cfgdict is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in cfgdict.keys():
                    setattr(self, attribute, cfgdict[attribute[1:]])

        if self._mesh_file_name is not None:
            self.__Domain.readMesh('{}'.format(self._mesh_file_name))

        self.__nt = self._Tend//self._dt 

    def loadMesh(self,fname: str):
        """ Loads the mesh"""
        self._mesh_file_name = fname
        self.__Domain.readMesh('{}'.format(self._mesh_file_name))

    def add_nodal_material_property(self,pname:str,ptype:str,prop:dict):
        """ adds material properties to elements"""
        self.__materials.add_nodal_property(pname,ptype,prop)


    def add_element_material_property(self,pname:str,ptype:str,prop:dict):
        """ adds material properties to elements"""
        self.__materials.add_element_property(pname,ptype,prop)

    def add_material_function(self,fname:str,fsign):
        """adds functions to map material properties when assembling matrices"""
        self.__materials.add_ud_function(fname,fsign)


    def assign_nodal_properties(self):
        nodal_properties = self.__materials.nodal_property_names() 
        if nodal_properties is not None:
            point_region_ids = self.__Domain.point_region_ids()
            npt = point_region_ids.shape[0]
            for mat_prop in nodal_properties:
                refval = self.get_parameter(mat_prop)
                if refval is not None:
                    pvals  = np.full(shape=(npt,1),fill_value=refval.numpy())
                    for pointID,regionID in enumerate(point_region_ids):
                        new_val = self.__materials.NodalProperty(mat_prop,pointID,regionID)
                        pvals[pointID] = new_val
                    self.set_parameter(mat_prop, pvals)
        self.__materials.remove_all_nodal_properties()    
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

    def set_initial_condition(self,U0:np.ndarray,H0:np.ndarray = None):
        if U0.ndim==1:
            self.__U = tf.Variable(U0[:,np.newaxis], name="U")
        else:
            self.__U = tf.Variable(U0, name="U")
    
        if H0 is not None:
            if H0.ndim==1:
                self.__H = tf.Variable(V0[:,np.newaxis], name="V")
            else:
                self.__H = tf.Variable(V0, name="V")
        else:
            self.__H = tf.Variable(np.full(shape=self.__U.shape,fill_value=1.0), name="V",dtype=U0.dtype)


    def set_stimulus(self,stimreg:np.ndarray,stimprops:dict):
        self.__Stimulus = Stimulus(stimprops)
        self.__Stimulus.set_stimregion(stimreg)    

    @tf.function
    def solve(self,U:tf.Variable, H:tf.Variable) -> (tf.Variable, tf.Variable):
        """ Explicit Euler ODE solver + implicit solver for diffusion"""
        dU, dH = self.differentiate(U, H)
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
        H1 = H + self._dt * dH
        return(U1, H1)


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
            U1,H1,= self.solve(self.__U,self.__H)
            self.__U = U1
            self.__H = H1
            # draw a frame every 1 ms
            if im and i % self._dt_per_plot == 0:
                image = U1.numpy()
                im.imshow(image)
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        if im:
            im.wait()   # wait until the window is closed
    
    def domain(self) -> Triangulation:
        return(self.__Domain)
    
    def solver(self) -> ConjGrad:
        return(self.__Solver)
    
    def stimulus(self):
        return(self.__Stimulus)

    def U(self) -> tf.Variable:
        return(self.__U)        

    def H(self) -> tf.Variable:
        return(self.__H)        

    def nt(self) -> int:
        return(self.__nt)

    def dt_per_plot(self) -> int:
        return(self._dt_per_plot)

    def ctime(self) -> float:
        return(self.__ctime)

if __name__=='__main__':
    dt      = 0.1
    diffusl = 0.001
    diffust = 0.001
    tclose0 = 120
    tclose3 = 60
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
    
    model = ModifiedMS2vSimple(config)
    
    # Define the materials
    model.add_element_material_property('sigma_l','region',{1: diffusl, 2: diffusl, 3: diffusl, 4: diffusl})
    model.add_element_material_property('sigma_t','region',{1: diffust, 2: diffust, 3: diffust, 4: diffust})
    model.add_nodal_material_property('tau_close','region',{1: tclose0, 2: tclose0, 3: tclose3, 4: tclose0})
    model.add_material_function('mass',dfmass)
    model.add_material_function('stiffness',sigmaTens)
    model.assign_nodal_properties()
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



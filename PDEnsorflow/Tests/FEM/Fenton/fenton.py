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
from gpuSolve.IO.writers import IGBWriter
import tensorflow as tf
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

from gpuSolve.ionic.fenton4v import *
from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.matrices.globalMatrices import compute_reverse_cuthill_mckee_indexing
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.linearsolvers.jacobi_precond import JacobiPrecond
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


class Fenton4vSimple(Fenton4v):
    """
    The monodomain model with Fenton-Cherry ionic model
    """

    def __init__(self, cfgdict=None):
        super().__init__()
        self._mesh_file_name : str               = None
        self._dt : float                         = 0.1
        self._dt_per_plot : int                  = 2
        self._Tend : float                       = 10
        self._use_renumbering : bool             = False

        if cfgdict is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in cfgdict.keys():
                    setattr(self, attribute, cfgdict[attribute[1:]])

        self._Domain : Triangulation             = Triangulation()
        self._materials : MaterialProperties     = MaterialProperties()
        self._Solver : ConjGrad                  = ConjGrad()
        self._Precond : JacobiPrecond            = JacobiPrecond()
        self._MASS                               = None
        self._U: tf.Variable                     = None
        self._V: tf.Variable                     = None
        self._W: tf.Variable                     = None
        self._S: tf.Variable                     = None
        self._ready_for_run : bool               = False
        self._ctime : float                      = 0.0
        self._nbstim : int                       = 0
        self._renumbering : dict                 = None
        self._StimulusDict: dict                 = None        
        self._nt : int                           = int(self._Tend//self._dt) 

        if self._mesh_file_name is not None:
            self._Domain.readMesh('{}'.format(self._mesh_file_name))


    def loadMesh(self,fname: str):
        """ Loads the mesh"""
        self._mesh_file_name = fname
        self._Domain.readMesh('{}'.format(self._mesh_file_name))

    def add_nodal_material_property(self,pname:str,ptype:str,prop:dict):
        """ adds material properties to elements"""
        self._materials.add_nodal_property(pname,ptype,prop)

    def add_element_material_property(self,pname:str,ptype:str,prop:dict):
        """ adds material properties to elements"""
        self._materials.add_element_property(pname,ptype,prop)

    def add_material_function(self,fname:str,fsign):
        """adds functions to map material properties when assembling matrices"""
        self._materials.add_ud_function(fname,fsign)

    def assign_nodal_properties(self):
        uniform_only = True
        nodal_properties = self._materials.nodal_property_names() 
        if nodal_properties is not None:
            point_region_ids = self._Domain.point_region_ids()
            npt = point_region_ids.shape[0]
            for mat_prop in nodal_properties:
                prtype = self._materials.nodal_property_type(mat_prop)
                refval = self.get_parameter(mat_prop)
                if refval is not None:
                    if prtype =='uniform':
                        pvals = self._materials.NodalProperty(mat_prop,-1,-1)
                    else:
                        uniform_only = False
                        pvals  = np.full(shape=(npt,1),fill_value=refval.numpy())
                        for pointID,regionID in enumerate(point_region_ids):
                            new_val = self._materials.NodalProperty(mat_prop,pointID,regionID)
                            pvals[pointID] = new_val
                    self.set_parameter(mat_prop, pvals)
        if( uniform_only or (not self._use_renumbering)):
            self._materials.remove_all_nodal_properties()
    
    def assemble_matrices(self):
        #Compute the domain connectivity
        connectivity = self._Domain.mesh_connectivity('True')
        # Assemble the matrices
        pattern     = compute_coo_pattern(connectivity)
        if self._use_renumbering:
            self._renumbering = compute_reverse_cuthill_mckee_indexing(pattern)
        lmatr       = {'mass':localMass,'stiffness':localStiffness}
        MATRICES    =  assemble_matrices_dict(lmatr,pattern,self._Domain,self._materials,connectivity, renumbering=self._renumbering)
        self._MASS = MATRICES['mass']
        STIFFNESS   = MATRICES['stiffness']
        A           = tf.sparse.add(self._MASS,tf.sparse.map_values(tf.multiply,STIFFNESS,self._dt))
        self._Domain.release_connectivity()
        self._materials.remove_all_element_properties()
        self._Solver.set_matrix(A)
        self._Precond.build_preconditioner(A.indices.numpy()[:,0], A.indices.numpy()[:,1], A.values.numpy(),A.shape[0])
        self._Solver.set_precond(self._Precond)

    def set_initial_condition(self,U0:np.ndarray = None, V0:np.ndarray = None, W0:np.ndarray = None,S0:np.ndarray = None):
        npt = self._Domain.Pts().shape[0]
        if U0 is not None:
            if U0.ndim==1:
                self._U = tf.Variable(U0[:,np.newaxis], name="U",dtype=tf.float32, trainable=False)
            else:
                self._U = tf.Variable(U0, name="U",dtype=tf.float32, trainable=False)
        else:
            self._U = tf.Variable(np.full(shape=(npt,1),fill_value=0.0), name="U",dtype=tf.float32, trainable=False)

        if V0 is not None:
            if V0.ndim==1:
                self._V = tf.Variable(V0[:,np.newaxis], name="V",dtype=tf.float32, trainable=False)
            else:
                self._V = tf.Variable(V0, name="V",dtype=tf.float32, trainable=False)
        else:
            self._V = tf.Variable(np.full(shape=self._U.shape,fill_value=1.0), name="V",dtype=tf.float32, trainable=False)

        if W0 is not None:
            if W0.ndim==1:
                self._W = tf.Variable(W0[:,np.newaxis], name="W",dtype=tf.float32, trainable=False)
            else:
                self._W = tf.Variable(W0, name="W",dtype=tf.float32, trainable=False)
        else:
            self._W = tf.Variable(np.full(shape=self._U.shape,fill_value=1.0), name="W",dtype=tf.float32, trainable=False)

        if S0 is not None:
            if S0.ndim==1:
                self._S = tf.Variable(S0[:,np.newaxis], name="S",dtype=tf.float32, trainable=False)
            else:
                self._S = tf.Variable(S0, name="S",dtype=tf.float32, trainable=False)
        else:
            self._S = tf.Variable(np.full(shape=self._U.shape,fill_value=0.0), name="S",dtype=tf.float32, trainable=False)

    def add_stimulus(self,stimreg:np.ndarray,stimprops:dict):
        self._nbstim +=1
        if self._StimulusDict is None:
            self._StimulusDict = {}
        self._StimulusDict[self._nbstim] = Stimulus(stimprops)
        self._StimulusDict[self._nbstim].set_stimregion(stimreg) 

    @tf.function
    def solve(self,U:tf.Variable, V:tf.Variable, W:tf.Variable, S:tf.Variable,I0:tf.constant) -> (tf.constant, tf.constant, tf.constant, tf.constant):
        """ Explicit Euler ODE solver + implicit solver for diffusion"""
        dU, dV, dW, dS = self.differentiate(U, V, W, S)
        dU     = tf.add(dU,I0)
        RHS0 = tf.add(U,tf.math.scalar_mul(self._dt,dU) )
        RHS = tf.sparse.sparse_dense_matmul(self._MASS,RHS0)
        self._Solver.set_X0(U)
        self._Solver.set_RHS(RHS)
        self._Solver.solve()
        U1 = self._Solver.X()
        V1 = V + tf.math.scalar_mul(self._dt, dV)
        W1 = W + tf.math.scalar_mul(self._dt, dW)
        S1 = S + tf.math.scalar_mul(self._dt, dS)
        return(U1, V1, W1, S1)

    @tf.function
    def update(self,U1 : tf.constant,V1 : tf.constant,W1 : tf.constant, S1 : tf.constant):
        self._U.assign(U1)
        self._V.assign(V1)
        self._W.assign(W1)
        self._S.assign(S1)


    #@tf.function
    def run(self, im=None):
        """
            Runs the model. 

            Args:
                im: A Screen/writer used to paint/write the transmembrane potential

            Returns:
                None
        """
        if not self._ready_for_run:
            raise Exception("model not initialised for run!")
            
        then = time.time()
        for i in tf.range(self._nt):
            self._ctime += self._dt
            I0 = tf.constant(np.zeros(shape=self._U.shape), name="I", dtype=tf.float32  )
            if self._StimulusDict is not None:
                for stimname,stimulus in self._StimulusDict.items():
                    I0 = tf.add(I0, stimulus.stimApp(self._ctime) )
            U1,V1,W1,S1 = self.solve(self._U,self._V,self._W,self._S,I0)
            self.update(U1, V1, W1, S1)
            # draw a frame every dt_per_plot ms
            if im and i % self._dt_per_plot == 0:
                image = self.U().numpy()
                im.imshow(image)
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        if im:
            im.wait()   # wait until the window is closed

    def finalize_for_run(self):
        if self._use_renumbering:
            # permutation of the initial condition
            self._U.assign(tf.gather(self._U,self._renumbering['perm']) )
            self._V.assign(tf.gather(self._V,self._renumbering['perm']))
            self._W.assign(tf.gather(self._W,self._renumbering['perm']))
            self._S.assign(tf.gather(self._S,self._renumbering['perm']) )
            # permutation of the stimulus indices
            for key ,stim in self._StimulusDict.items():
                stim.apply_indices_permutation(self._renumbering['perm'])    
            nodal_properties = self._materials.nodal_property_names() 
            if nodal_properties is not None:
                for mat_prop in nodal_properties:
                    prtype = self._materials.nodal_property_type(mat_prop)
                    refval = self.get_parameter(mat_prop)
                    if refval is not None and not (prtype =='uniform'):
                        pvals = tf.gather(refval,self._renumbering['perm']).numpy()
                        self.set_parameter(mat_prop, pvals)                        
                self._materials.remove_all_nodal_properties()
        self._ready_for_run = True

    def domain(self) -> Triangulation:
        return(self._Domain)
    
    def solver(self) -> ConjGrad:
        return(self._Solver)

    def precond(self) -> JacobiPrecond:
        return(self._Precond)

    def stimulus(self) ->dict:
        return(self._StimulusDict)

    def U(self) -> tf.Variable:
        if self._use_renumbering:
            return(tf.gather(self._U,self._renumbering['iperm']) )
        else:
            return(self._U)

    def V(self) -> tf.Variable:
        if self._use_renumbering:
            return(tf.gather(self._V,self._renumbering['iperm']) )
        else:
            return(self._V)

    def W(self) -> tf.Variable:
        if self._use_renumbering:
            return(tf.gather(self._W,self._renumbering['iperm']) )
        else:
            return(self._W)

    def S(self) -> tf.Variable:
        if self._use_renumbering:
            return(tf.gather(self._S,self._renumbering['iperm']) )
        else:
            return(self._S)

    def nt(self) -> int:
        return(self._nt)

    def dt_per_plot(self) -> int:
        return(self._dt_per_plot)

    def ctime(self) -> float:
        return(self._ctime)

    def Tend(self) ->float:
        return(self._Tend)

if __name__=='__main__':
    dt      = 0.1
    TS2     = 210.0
    diffusl = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    diffust = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}

    config  = {
        'mesh_file_name': os.path.join('..','..','data','triangulated_square.pkl'),
        'use_renumbering': True,
        'dt' : dt,
        'dt_per_plot': int(1.0/dt),   #record every ms
        'Tend': 1000
        }

    cfgstim1 = {'tstart': 0.0, 
                       'nstim': 1, 
                       'period':100,
                       'duration':np.max([0.4,dt]),
                       'intensity':1.0,
                       'name':'crossstim'
              }
    
    cfgstim2 = {'tstart': TS2, 
                       'nstim': 1, 
                       'period':100,
                       'duration':np.max([0.4,dt]),
                       'intensity':1.0,
                       'name':'crossstim'
              }
    
    model = Fenton4vSimple(config)
    # Define the materials
    model.add_element_material_property('sigma_l','region',diffusl)
    model.add_element_material_property('sigma_t','region',diffust)
    model.add_material_function('mass',dfmass)
    model.add_material_function('stiffness',sigmaTens)
    model.assign_nodal_properties()
    model.assemble_matrices()
    Lx = model.domain().Pts()[:,0].max()
    Ly = model.domain().Pts()[:,1].max()
    S1 = model.domain().Pts()[:,0]<0.05*Lx
    S2 = np.logical_and(model.domain().Pts()[:,0]<Lx,model.domain().Pts()[:,1]<0.5*Ly)
    # Set the initial condition to U=0,H=1 everywhere
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
    im.imshow(model.U().numpy())
    model.run(im)
    im = None




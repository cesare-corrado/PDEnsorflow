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

import numpy as np
import time
from  gpuSolve.IO.writers import ResultWriter
import tensorflow as tf
tf.config.run_functions_eagerly(True)
if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))

from gpuSolve.ionic.fenton4v import *
from gpuSolve.entities.domain3D import Domain3D
from gpuSolve.diffop3D import laplace_homog as laplace
from gpuSolve.diffop3D import laplace_conv_homog as conv_laplace
from gpuSolve.force_terms import Stimulus


@tf.function
def enforce_boundary(X):
    """
        Enforcing the no-flux (Neumann) boundary condition
    """
    padmode = 'symmetric'
    paddings = tf.constant([[1,1], [1,1], [1,1]])
    return( tf.pad(X[1:-1,1:-1,1:-1], paddings=paddings,mode=padmode,name='boundary' ) )



class Fenton4vSimple(Fenton4v):
    """
    The monodomain model with Fenton-Cherry ionic model
    """

    def __init__(self, props):
        super().__init__()
        self._domain     = Domain3D(props)
        self.min_v       = 0.0
        self.max_v       = 1.0
        self.dt          = 0.1
        self.diff        = 1.0
        self.convl       = False
        self.samples     = 10000
        self.s2_time     = 200
        self.dt_per_plot = 10
        self.tinit       = 0.0
        for attribute in self.__dict__.keys():
            if attribute in props.keys():
                setattr(self, attribute, props[attribute])

        then = time.time()
        self._domain.load_geometry_file()
        self.DX = self._domain.DX()
        self.DY = self._domain.DY()
        self.DZ = self._domain.DZ()
        elapsed = (time.time() - then)
        tf.print('initialisation, elapsed: %f sec' % elapsed)
        self.tinit += elapsed

    def  domain(self):
        return(self._domain.geometry())

    @tf.function
    def solve(self, state):
        """ Explicit Euler ODE solver """
        U, V, W, S = state
        U0 = enforce_boundary(U)
        dU, dV, dW, dS = self.differentiate(U, V, W, S)
        if self.convl:
            U1 = U0 + self.dt * dU + self.diff * self.dt * conv_laplace(U0,self.DX,self.DY,self.DZ)
        else:
            U1 = U0 + self.dt * dU + self.diff * self.dt * laplace(U0,self.DX,self.DY,self.DZ)
        V1 = V + self.dt * dV
        W1 = W + self.dt * dW
        S1 = S + self.dt * dS
        return U1, V1, W1, S1


    @tf.function
    def run(self, im=None):
        """
            Runs the model. 

            Args:
                im: A Screen/writer used to paint/write the transmembrane potential

            Returns:
                None
        """
        width  = self._domain.width()
        height = self._domain.height()
        depth  = self._domain.depth()

        # the initial values of the state variables
        # initial values (u, v, w, s) = (0.0, 1.0, 1.0, 0.0)
        u_init  = np.full([width,height, depth], self.min_v, dtype=np.float32)
        s2_init = np.full([width,height, depth], self.min_v, dtype=np.float32)
        u_init[0:2,:,:] = self.max_v
        s2_init[:width//2,:height//2,:] = self.max_v
        then = time.time()
        U = tf.Variable(u_init, name="U" )
        V = tf.Variable(np.full([width,height,depth], 1.0, dtype=np.float32), name="V"    )
        W = tf.Variable(np.full([width,height,depth], 1.0, dtype=np.float32), name="W"    )
        S = tf.Variable(np.full([width,height,depth], 0.0, dtype=np.float32), name="S"    )
        elapsed = (time.time() - then)
        tf.print('U,V,W,S variables, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed
        u_init=[]

        #define a source that is triggered at t=s2_time: : vertical (2D) along the left face
        then = time.time()
        s2 = Stimulus({'tstart': self.s2_time, 
                       'nstim': 1, 
                       'period':800,
                       'duration':self.dt,
                       'dt': self.dt,
                       'intensity':self.max_v})
        s2.set_stimregion(s2_init)
        elapsed = (time.time() - then)
        tf.print('s2 tensor, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed
        tf.print('total initialization: %f sec' % self.tinit)
        
        s2_init=[]
        then = time.time()
        if im:
            image = U.numpy()
            im.imshow(image)
        for i in tf.range(self.samples):
            state = [U, V, W, S]
            U1, V1, W1, S1 = self.solve(state)
            U = U1
            V = V1
            W = W1
            S = S1
            #if s2.stimulate_tissue_timevalue(float(i)*self.dt):
            if s2.stimulate_tissue_timestep(i,self.dt):
                U = tf.maximum(U, s2())
            # draw a frame every 1 ms
            if im and i % self.dt_per_plot == 0:
                image = U.numpy()
                im.imshow(image)
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        print('TOTAL, elapsed: %f sec' % (elapsed+self.tinit))
        if im:
            im.wait()   # wait until the window is closed


if __name__ == '__main__':
    print('=======================================================================')

    config = {
        'width': 32,
        'height': 32,
        'depth': 32,
        'dx': 1,
        'dy': 1,
        'dz': 1,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 0.8,
        'samples': 10000,
        's2_time': 200,
         'convl': False
    }

    print('config:')
    for key,value in config.items():
        print('{0}\t{1}'.format(key,value))
    
    print('=======================================================================')
    model = Fenton4vSimple(config)
    im = ResultWriter(config)
    [im.width,im.height,im.depth] = model.domain().numpy().shape
    model.run(im)
    im = None




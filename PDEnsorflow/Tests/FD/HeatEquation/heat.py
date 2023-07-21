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

EAGERMODE=False
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
from  gpuSolve.IO.writers import ResultWriter
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



class HeatEquation:
    """
    The heat equation model
    """

    def __init__(self, props):
        self._domain : Domain3D      = Domain3D(props)
        self.min_v   : float         = 0.0
        self.max_v   : float         = 1.0
        self.dt      : float         = 0.1
        self.diff    : float         = 1.0
        self.convl   : bool          = False
        self.samples : int           = 10000
        self.s2_time : float         = 200
        self.dt_per_plot : int       = 10
        self.tinit       : float          = 0.0
        self.U           : tf.Variable    = None
        self.s2          : Stimulus       = None
        for attribute in self.__dict__.keys():
            if attribute in props.keys():
                setattr(self, attribute, props[attribute])

        then = time.time()
        self._domain.load_geometry_file()
        self.DX = tf.constant(self._domain.dx(), dtype=np.float32)
        self.DY = tf.constant(self._domain.dy(), dtype=np.float32)
        self.DZ = tf.constant(self._domain.dz(), dtype=np.float32)
        elapsed = (time.time() - then)
        print('initialisation, elapsed: %f sec' % elapsed)
        self.tinit += elapsed

    def  domain(self):
        return(self._domain.geometry())

    @tf.function
    def solve(self):
        """ Explicit Euler ODE solver """
        U0 = enforce_boundary(self.U)
        if self.convl:
          U1 = U0 + self.diff * self.dt * conv_laplace(U0,self.DX,self.DY,self.DZ)
        else:
          U1 = U0 + self.diff * self.dt * laplace(U0,self.DX,self.DY,self.DZ)
        return U1

    def initialize(self):
        width  = self._domain.width()
        height = self._domain.height()
        depth  = self._domain.depth()
        # the initial value of the state variable
        u_init  = np.full([width,height, depth], self.min_v, dtype=np.float32)
        s2_init = np.full([width,height, depth], self.min_v, dtype=np.float32)
        u_init[0:2,:,:] = self.max_v
        s2_init[:width//2,:height//2,:] = self.max_v
        then = time.time()
        self.U = tf.Variable(u_init, name="U" )
        elapsed = (time.time() - then)
        tf.print('U variable, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed

        #define a source that is triggered at t=s2_time: : vertical (2D) along the left face
        then = time.time()
        self.s2 = Stimulus({'tstart': self.s2_time, 
                       'nstim': 1, 
                       'period':800,
                       'duration':self.dt,
                       'dt': self.dt,
                       'intensity':self.max_v})
        self.s2.set_stimregion(s2_init)
        elapsed = (time.time() - then)
        tf.print('s2 tensor, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed
        tf.print('total initialization: %f sec' % self.tinit)


    #@tf.function
    def run(self, im=None):
        """
            Runs the model. 

            Args:
                im: A Screen/writer used to paint/write the transmembrane potential

            Returns:
                None
        """
        then = time.time()
        if im:
            image = self.U.numpy()
            im.imshow(image)
        for i in tf.range(self.samples):
            U1 = self.solve()
            self.U.assign(U1)
            #if self.s2.stimulate_tissue_timevalue(float(i)*self.dt):
            if self.s2.stimulate_tissue_timestep(i,self.dt):
                self.U.assign(tf.maximum(self.U, self.s2()))
            # draw a frame every 1 ms
            if im and i % self.dt_per_plot == 0:
                image = self.U.numpy()
                im.imshow(image)
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        print('TOTAL, elapsed: %f sec' % (elapsed+self.tinit))
        if im:
            im.wait()   # wait until the window is closed


if __name__ == '__main__':
    print('=======================================================================')

    config = {
        'width': 64,
        'height': 64,
        'depth': 64,
        'dx': 1,
        'dy': 1,
        'dz': 1,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 1.5,
        'samples': 10000,
        's2_time': 200,
         'convl': False
    }

    print('config:')
    for key,value in config.items():
        print('{0:9}\t{1}'.format(key,value))
    
    print('=======================================================================')
    model = HeatEquation(config)
    im = ResultWriter(config)
    [im.width,im.height,im.depth] = model.domain().shape
    model.initialize()
    model.run(im)
    im = None



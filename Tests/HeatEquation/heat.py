#!/usr/bin/env python
"""
    A TensorFlow-based 2D Heat Equation Solver

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

try:
  import vedo 
  is_vedo = True
  from  gpuSolve.IO.writers import VedoPlotter

except:  
    is_vedo = False
    print('Warning: no vedo found; using matplotlib',flush=True)


import tensorflow as tf
tf.config.run_functions_eagerly(True)
if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))

  
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
        self.width     = 1
        self.height    = 1
        self.depth     = 1
        self.min_v     = 0.0
        self.max_v     = 1.0
        self.dx        = 1.0
        self.dy        = 1.0
        self.dz        = 1.0
        self.diff      = 1.0
        self.convl     = False
        for key, val in props.items():
            setattr(self, key, val)

        self._config={}
        for attribute in self.__dict__.keys():
            if attribute[:1] != '_':
              self._config[attribute] = getattr(self,attribute)


        then = time.time()
        self.DX    = tf.constant(self.dx, dtype=np.float32)
        self.DY    = tf.constant(self.dy, dtype=np.float32)
        self.DZ    = tf.constant(self.dz, dtype=np.float32)
        elapsed = (time.time() - then)
        tf.print('initialisation of DXYZ, elapsed: %f sec' % elapsed)
        self.tinit = elapsed

        for attribute in self._config.keys():
              self._config[attribute] = getattr(self,attribute)


    def  config(self):
        return(self._config)


    @tf.function
    def solve(self, U):
        """ Explicit Euler ODE solver """
        U0 = enforce_boundary(U)
        if self.convl:
          U1 = U0 + self.diff * self.dt * conv_laplace(U0,self.dx,self.dy,self.dz)
        else:
          U1 = U0 + self.diff * self.dt * laplace(U0,self.DX,self.DY,self.DZ)
        return U1


    @tf.function
    def run(self, im=None):
        """
            Runs the model. 

            Args:
                im: A Screen/writer used to paint/write the transmembrane potential

            Returns:
                None
        """
        # the initial value of the variable
        u_init = np.full([self.height, self.width,self.depth], self.min_v, dtype=np.float32)
        s2_init = np.full([self.height, self.width,self.depth], self.min_v, dtype=np.float32)

        u_init[:,0:2,:] = self.max_v
        s2_init[:self.height//2, :self.width//2,:] = self.max_v
        then = time.time()
        U = tf.Variable(u_init, name="U" )
        elapsed = (time.time() - then)
        tf.print('U variable, elapsed: %f sec' % elapsed)
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
        for i in tf.range(self.samples):
            U1 = self.solve(U)
            U = U1
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
        print('{0}\t{1}'.format(key,value))
    
    print('=======================================================================')
    model = HeatEquation(config)
    if is_vedo:
        im = ResultWriter(model.config())
    else:
        im = ResultWriter(model.config())
    model.run(im)
    im = None



  

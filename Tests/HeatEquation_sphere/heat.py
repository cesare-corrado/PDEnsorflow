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

  
from gpuSolve.diffop3D import laplace_heterog as laplace


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
        self.radius    = 1.0
        self.hole      = False
        self.cylindric = False

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

        c0=0.5*np.array([self.dx*self.width,self.dy*self.height,self.dz*self.depth])
        then = time.time()
        if self.hole:
            tf.print('create the domain with an hole')
        else:
            tf.print('create the spherical domain')
        
        xx, yy,zz = np.meshgrid(self.dx*np.arange(self.width), self.dy*np.arange(self.height),self.dz*np.arange(self.depth) )
        cyl_coef=np.logical_not(self.cylindric).astype(np.float32)
        if self.hole:
            img_vox = np.logical_not(((xx-c0[0])*(xx-c0[0])+(yy-c0[1])*(yy-c0[1])+cyl_coef*(zz-c0[2])*(zz-c0[2]) )<=(self.radius*self.radius)).astype(np.float32)
        else:
            img_vox = (((xx-c0[0])*(xx-c0[0])+(yy-c0[1])*(yy-c0[1])+cyl_coef*(zz-c0[2])*(zz-c0[2]) )<=(self.radius*self.radius)).astype(np.float32)
        
        self._domain = tf.constant(img_vox,dtype=np.float32, name='domain' )        
        img_vox = self.diff*img_vox
        self.conductivity = tf.constant(img_vox,dtype=np.float32, name='diffusion' )
        
        elapsed = (time.time() - then)
        tf.print('initialisation of conductivity tensor, elapsed: %f sec' % elapsed)
        self.tinit += elapsed
        for attribute in self._config.keys():
              self._config[attribute] = getattr(self,attribute)
        for attribute in self._config.keys():
              self._config[attribute] = getattr(self,attribute)


    def  config(self):
        return(self._config)


    @tf.function
    def solve(self, U):
        """ Explicit Euler ODE solver """
        U0 = enforce_boundary(U)
        U1 = U0 + self.dt * laplace(U0,self.conductivity,self.DX,self.DY,self.DZ)
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
        
        if self.hole:
            # first stimulus on one side; second stimulus on a brick
            u_init[:,0:2,:] = self.max_v
            s2_init[:self.height//2, :self.width//2,:] = self.max_v            
            
        else:
            u_init[:,(self.width//2-10):(self.width//2+10),:] = self.max_v
            s2_init[(self.width//2-10):(self.width//2+10),:,:] = self.max_v            

        then = time.time()
        U = tf.Variable(u_init, name="U" )
        U = tf.where(self._domain>0.0, U, self.min_v)
        elapsed = (time.time() - then)
        tf.print('U variable, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed
        u_init=[]

        #define a source that is triggered at t=s2_time: : vertical (2D) along the left face
        then = time.time()
        s2 = tf.where(self._domain>0.0, tf.constant(s2_init,dtype=np.float32), self.min_v,name="s2")
        elapsed = (time.time() - then)
        tf.print('s2 tensor, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed
        tf.print('total initialization: %f sec' % self.tinit)
        
        s2_init=[]
        then = time.time()
        for i in tf.range(self.samples):
            U1 = self.solve(U)
            U = U1
            if i == int(self.s2_time / self.dt):
                U = tf.maximum(U, s2)
            # draw a frame every 1 ms
            if im and i % self.dt_per_plot == 0:
                image = tf.where(self._domain>0.0, U, -1.0).numpy()
                im.imshow(image)                
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        print('TOTAL, elapsed: %f sec' % (elapsed+self.tinit))
        if im:
            im.wait()   # wait until the window is closed






if __name__ == '__main__':
    print('=======================================================================')

    config = {
        'width':  128,
        'height': 128,
        'depth':  128,
        'radius': 32,
        'hole': False,
        'cylindric':False,
        'dt': 0.1,
        'dt_per_plot' : 10,
        'diff': 1.0,
        'samples': 10000,
        's2_time': 210
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



  

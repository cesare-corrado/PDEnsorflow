#!/usr/bin/env python
"""
    A TensorFlow-based Laplace Solver

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
from gpuSolve.IO.readers.imagedata import ImageData


import tensorflow as tf
tf.config.run_functions_eagerly(True)
if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))


from gpuSolve.diffop3D import laplace_heterog as laplace  




@tf.function
def enforce_boundary(X,cval = 10.0):
    """
    Enforcing Dirichlet homogeneous B.C. on the cube faces
    """
    padmode = 'constant'
    paddings = tf.constant([[1,1], [1,1], [1,1]])
    return( tf.pad(X[1:-1,1:-1,1:-1], paddings=paddings,constant_values=cval,mode=padmode,name='boundary' ) )



@tf.function
def compute_convergence_norm(U1,U2):
    """
    Computes the norm of the difference between two vectors
    to check the convergence
    """
    DU=U1-U2
    return(tf.norm(DU))


class LaplaceSolver:
    """
    The Laplace solver to evaluate a scalar field on the entire cubic domain
    with a constant value of v on the atrial domain and a value =0 on the
    external boundary. This to determine the internal/ecternal areas 
    as the hotter areas
    """

    def __init__(self, props):
        
        self.image_threshold=1.e-4
        self.min_v   = 0.0
        self.max_v   = 1.0
        self.dx      = 1.0
        self.dy      = 1.0
        self.dz      = 1.0
        self.diff    = 1.0
        self.walltmp = 10.0
        self.fname = ''
        for key, val in props.items():
            setattr(self, key, val)

        self._config={}
        for attribute in self.__dict__.keys():
            if attribute[:1] != '_':
              self._config[attribute] = getattr(self,attribute)

        then = time.time()
        self.DX    = tf.constant(self.dx,dtype=np.float32)
        self.DY    = tf.constant(self.dy,dtype=np.float32)
        self.DZ    = tf.constant(self.dz,dtype=np.float32)
        elapsed = (time.time() - then)
        tf.print('initialisation of DXYZ, elapsed: %f sec' % elapsed)
        self.tinit = elapsed
        
        Image = ImageData()    
        then = time.time()
        tf.print('read image domain')
        Image.load_image(self.fname,self.Mx,self.My)            
        img_vox = Image.get_rescaled_data('unit').astype(np.float32)            
        [self.width,self.height,self.depth]=img_vox.shape 
        img_vox[img_vox>self.image_threshold]=1.0
        img_vox[img_vox<=self.image_threshold]=0.0        
        self.domain = tf.constant(img_vox,dtype=np.float32, name='domain' )
        #set conductivity to 0 inside the biatrial mesh and equal to diff outside
        img_vox = self.diff*np.logical_not(img_vox.astype(bool))
        self.conductivity = tf.constant(img_vox,dtype=np.float32, name='diffusion' )
        elapsed = (time.time() - then)        
        tf.print('initialisation, elapsed: %f sec' % elapsed)        
        tf.print('New domain dimensions:  ( {0},{1}, {2})'.format(self.width,  self.height, self.depth) )
        self.tinit += elapsed

        hmin = np.min([self.dx,self.dy,self.dz])
        dtmax=0.5*(hmin*hmin)/self.diff
        if dtmax<self.dt:
            tf.print('dt: {} -> {} (set for numerical stability)'.format(self.dt,dtmax))
            self.dt=dtmax        
        else:
            tf.print('dt: {} dtmax: {} '.format(self.dt,dtmax))

        for attribute in self._config.keys():
              self._config[attribute] = getattr(self,attribute)


    def  config(self):
        return(self._config)



    @tf.function
    def solve(self, U):
        """ Explicit Euler ODE solver """
        U0 = enforce_boundary(U,self.walltmp)
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
        then = time.time()
        U = tf.Variable(u_init, name="U" )    
        U = enforce_boundary(U)    
        #U = tf.where(self.domain>0.0, self.max_v, self.min_v)
        elapsed = (time.time() - then)
        tf.print('U variable, elapsed: %f sec' % elapsed)
        self.tinit = self.tinit + elapsed        
        u_init=[]
        tf.print('total initialization: %f sec' % self.tinit)

        then = time.time()
        for i in tf.range(self.samples):
            U1 = self.solve(U)
            #re impose BCs
            #U1 = tf.where(self.domain>0.0, self.max_v, U1)        
            #U1 = enforce_boundary(U)
            if i % self.dt_per_plot == 0:
                cnorm = compute_convergence_norm(U,U1).numpy()                
                if im:
                    image = U1.numpy()
                    im.imshow(image)
                if cnorm<self.toll:
                    tf.print("Convergence reached ({:3.2f} < {}; nb of iter: {})".format(cnorm.round(4), self.toll,i))
                    break;
                else:
                    tf.print("Iteration {}; norm {:3.2f}".format(i,cnorm.round(4)))
            U = U1
        elapsed = (time.time() - then)
        print('solution, elapsed: %f sec' % elapsed)
        print('TOTAL, elapsed: %f sec' % (elapsed+self.tinit))
        
        if im:
            im.wait()   # wait until the window is closed



if __name__ == '__main__':
    print('=======================================================================')
    config = {
        'width': 16,
        'height': 16,
        'depth': 16,
        'fname': '../data/structure.png',
        'Mx': 16,
        'My': 8,
        'dt': 0.1,
        'dt_per_plot' : 100,
        'diff': 1.0,
        'samples': 12000,
        'walltmp': 10,
        'toll': 0.01
    }
    
    print('config:')
    for key,value in config.items():
        print('{0}\t{1}'.format(key,value))
    
    print('=======================================================================')
    model = LaplaceSolver(config)
    
    im = ResultWriter(config)
    im.initval=np.nan
    im.initialise_cube()
    model.run(im)
    im = None


  







"""
diffop3D submodule of gpuSolve.
This module contains 3D partial differential operators
Contains:
  * laplace_heterogeneous_isotropic_diffusion:   3D laplace solver for heterogeneous isotropic diffusion
  * laplace_heterogeneous_anisotropic_diffusion: 3D laplace solver for heterogeneous anisotropic diffusion
   

"""
import tensorflow as tf

__version__=['1','0','1']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)




from gpuSolve.diffop3D.laplace_heterogeneous_isotropic_diffusion import laplace_heterogeneous_isotropic_diffusion as laplace_heterog
from gpuSolve.diffop3D.laplace_heterogeneous_anisotropic_diffusion import laplace_heterogeneous_anisotropic_diffusion as laplace_heterog_aniso
from gpuSolve.diffop3D.laplace_homogeneous_isotropic_diffusion import laplace_homogeneous_isotropic_diffusion as laplace_homog
from gpuSolve.diffop3D.laplace_convolution_homogeneous_isotropic_diffusion import laplace_convolution_homogeneous_isotropic_diffusion as laplace_conv_homog

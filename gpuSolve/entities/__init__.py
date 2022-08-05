"""
entities submodule of gpuSolve.
This module contains general entities to group data 
Contains:
  * domain3D: class that defines a domain in a 3D space

"""
import tensorflow as tf

__version__=['1','0','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


from gpuSolve.entities.domain3D import Domain3D 


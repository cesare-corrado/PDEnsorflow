"""
force_terms submodule of gpuSolve.
This module contains forcing terms
Contains:
  * Stimulus: a class to handle stimulus
   

"""
import tensorflow as tf

__version__=['1','1','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)




from gpuSolve.force_terms.stimulus import Stimulus


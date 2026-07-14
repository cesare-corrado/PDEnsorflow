"""
IO submodule of gpuSolve.
Contains:
    * writers:  a collection of objects to write and plots the results
    * readers:  a collection of objects to read in handle the data

"""

from gpuSolve._version import __version__


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)
    



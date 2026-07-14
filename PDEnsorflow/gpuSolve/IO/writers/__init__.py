"""
writers and plotters
submodule of IO
Contains:
    * BaseWriter: a base class that manages the GPU buffering/chunking of the solution
    * ResultWriter: a class to write results in npy format
    * VedoPlotter: a class implementing a plotter based on ved library
    * CarpMeshWriter: a class to write the mesh in carp {.pts, .elem, .lon} format
    * IGBWriter: a class to write the output in .igb format (meshalyzer)

"""



from gpuSolve._version import __version__


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


from gpuSolve.IO.writers.basewriter import BaseWriter
from gpuSolve.IO.writers.resultwriter import ResultWriter
#from gpuSolve.IO.writers.vedoplotter import VedoPlotter
from gpuSolve.IO.writers.carpmeshwriter import CarpMeshWriter
from gpuSolve.IO.writers.igbwriter import IGBWriter

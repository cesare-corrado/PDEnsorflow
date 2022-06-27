"""
writers and plotters  
submodule of IO
Contains:
    * ResultWriter: a class to write results in npy format
    * VedoPlotter: a class implementing a plotter based on ved library

"""



__version__=['1','0','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


from gpuSolve.IO.writers.resultwriter import ResultWriter as ResultWriter
from gpuSolve.IO.writers.vedoplotter import VedoPlotter as VedoPlotter


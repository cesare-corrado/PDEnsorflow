"""
readers
submodule of IO
Contains:
    * ImageData: a class to read images and transform to numpy data type
"""



__version__=['1','1','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


from gpuSolve.IO.readers.imagedata import ImageData
from gpuSolve.IO.readers.carpmeshreader import CarpMeshReader

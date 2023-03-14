"""
entities submodule of gpuSolve.
This module contains general entities to group data 
Contains:
  * domain3D: class that defines a domain in a 3D space
  * Triangulation: a class that handles domain triangulations
  * MaterialProperties: a container for material properties; allows function definitions.

"""

__version__=['1','1','0']


def version():
  verstr=''
  for x in __version__:
      verstr = verstr+'{}.'.format(x)
  verstr=verstr=verstr[:-1]
  return(verstr)


from gpuSolve.entities.domain3D import Domain3D 
from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties

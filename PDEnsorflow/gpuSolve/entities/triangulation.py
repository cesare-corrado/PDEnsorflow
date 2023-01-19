import tensorflow as tf
import numpy as np
import sys
import time
from gpuSolve.IO.readers.carpmeshreader import CarpMeshReader


class Triangulation:
    """
    Class Triangulation
    This class defines a domain from Triangulation for numerical simulations.
    It collects the domain mesh and the properties such as fibre directions and conductivities.
    """
    def __init__(self):
        self._Pts    = None
        self._Elems  = None
        self._Fibres = None

    def Pts(self):
        return(self._Pts)
        
    def Elems(self):
        return(self._Elems)

    def Fibres(self):
        return(self._Fibres)

    def readMesh(self,fsuffix):
        """function readMesh(suffix)
        reads a mesh in carp format
        """
        reader = CarpMeshReader()
        reader.read(fsuffix)
        self._Pts    = tf.constant(reader.Pts())
        self._Fibres = tf.constant(reader.Fibres())
        self._Elems  = {}
        for key,value in reader.Elems().items():
            if value is not None:
                self._Elems[key] = tf.constant(value,dtype=tf.int32)
    
    
    
    

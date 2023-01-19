import tensorflow as tf
import numpy as np
import pickle
import os
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

    def readMesh(self,filename):
        """function readMesh(filename)
        determines the mesh format from the file extension
        (carp format of a binary .pkl file) and reads a mesh 
        """
        fname = os.path.split(filename)[-1]
        tmp    = fname.rfind('.') 
        suffix = fname[tmp:] 
        if tmp ==-1:
            self.__readMeshCarpFormat(filename)
        elif suffix=='.pkl':
            self.__readMeshPickleFormat(filename)
        else:
            raise Exception('{}: unknown format'.format(suffix))
            
    
    def saveMesh(self,foutName):
        """function savemesh(foutName)
        saves the mesh in .pkl format.
        This function saves points, elements and fibres only
        """
        if not(foutName[-4:]=='.pkl'):
            foutName = '{}.pkl'.format(foutName)
        Pts    = self._Pts.numpy()        
        Fibres = self._Fibres.numpy()
        Elems  = {}
        for key,value in self._Elems.items():
            Elems[key]=value.numpy()
        
        mesh0 = {'Pts': Pts,
                 'Elems': Elems,
                 'Fibres': Fibres
                }                 
        with open(foutName,'wb') as fout:
            pickle.dump(mesh0,fout,protocol=pickle.HIGHEST_PROTOCOL)     
        
            
    def __readMeshPickleFormat(self,fname):
        '''This function reads a mesh in .pkl format
        The input data is a pkl file with a mesh having
        data of type numpy and containing the basic entries 
        of a carp mesh
        '''
        try:
            mesh0        = pickle.load(open(fname,'rb'))
            self._Pts    = tf.constant(mesh0['Pts'])
            self._Fibres = tf.constant(mesh0['Fibres'])
            self._Elems  = {}
            for key,value in mesh0['Elems'].items():
                if value is not None:
                    self._Elems[key] = tf.constant(value,dtype=tf.int32)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    def __readMeshCarpFormat(self,fsuffix):
        '''This function reads a mesh in a carp format. It takes
        the prefix of the carp files (.elem, .lon, .pts) as the input
        '''
        try:
            reader = CarpMeshReader()
            reader.read(fsuffix)
            self._Pts    = tf.constant(reader.Pts())
            self._Fibres = tf.constant(reader.Fibres())
            self._Elems  = {}
            for key,value in reader.Elems().items():
                if value is not None:
                    self._Elems[key] = tf.constant(value,dtype=tf.int32)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise



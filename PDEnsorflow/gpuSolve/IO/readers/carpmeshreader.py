import numpy as np
import sys
import os
from time import time
    
def elemDeCode(elemName):
    '''
    elemDecode(elemName) converts the carp file name ElemName
    into the PDEnsorflow elem name.
    '''
    elem_to_code={ 'Cx': 'Edges',
                   'Tr': 'Trias',
                   'Qd': 'Quads',
                   'Tt': 'Tetras',
                   'Hx': 'Hexas',
                   'Py':  'Pyras',
                   'Pr':  'Prisms'
                   }
    return(elem_to_code[elemName])



class CarpMeshReader:
    """ 
    class CarpMeshReader: utility to read a mesh file in carp format
    It requires files .pts, .elem and .lon
    Provided a prefix (with the path), this class reads 
    Carp format and fills in the entities that describe the mesh.
    NOTE: this DOES NOT provide a final domain.
    """
    
    def __init__(self):
        self.__Pts    = None
        self.__nElem  = None
        self.__Elems  = None
        self.__Fibres = None
        self.__timeR  = 0.0


    def Pts(self):
        return(self.__Pts)

    def Edges(self):
        return(self.__Elems['Edges'])

    def Trias(self):
        return(self.__Elems['Trias'])

    def Quads(self):
        return(self.__Elems['Quads'])

    def Tetras(self):
        return(self.__Elems['Tetras'])

    def Hexas(self):
        return(self.__Elems['Hexas'])

    def Pyras(self):
        return(self.__Elems['Pyras'])

    def Prisms(self):
        return(self.__Elems['Prisms'])
    
    def Elems(self):
        """function Elems()
        Returns the entire dict of elements
        """
        return(self.__Elems)

    def read(self,fsuffix):
        """
        function read(fsuffix)
        This function reads a mesh in carp format and parses the contents 
        """
        self.__readPoints(fsuffix)
        self.__readElements(fsuffix)
        self.__readFibres(fsuffix)
        print('total: {:4.2f} s.'.format(self.__timeR) )


    def __readPoints(self,fsuffix):
        '''
        This function reads the .pts file
        '''
        nodeFname = '{}.pts'.format(fsuffix)
        try:
            print('reading Nodes',flush=True)
            tstart       = time()
            with open(nodeFname,'r') as fnod:
                npt       = int(fnod.readline().strip())
                self.__Pts = np.zeros(shape=(npt,3),dtype=np.float32)
                for jj in range(npt):
                    row = fnod.readline()
                    row = row.strip().split()
                    for ic,crd in enumerate(row):
                        self.__Pts[jj,ic]=float(crd)
            treadNodes   = time()-tstart
            self.__timeR  += treadNodes
            print('read {} nodes in {:4.2f} s.'.format(npt,treadNodes) )
            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __readElements(self,fsuffix):
        '''
        This function reads the .elem file
        '''
        elemFname ='{}.elem'.format(fsuffix)
        try:
            print('reading Elements',flush=True)
            self.__Elems = {'Edges': [],
                            'Trias': [],
                            'Quads': [],
                            'Tetras': [],
                            'Hexas': [],
                            'Pyras': [],
                            'Prisms': []
                        }  

            tstart       = time()
            with open(elemFname,'r') as felem:
                self.__nElem  = int(felem.readline().strip())
                for jj in range(self.__nElem):
                    row      = felem.readline()
                    row      = row.strip().split()
                    elemType = elemDeCode(row[0])
                    row      = row[1:]
                    elem     = []
                    for iEle,iEntry in enumerate(row):
                        elem.append(int(iEntry))
                    self.__Elems[elemType].append(elem)    

            for key,value in self.__Elems.items():
                if(len(value)>0):
                    self.__Elems[key] = np.array(value,dtype=int)
                else:
                    self.__Elems[key] = None
            treadElems   = time()-tstart
            self.__timeR  += treadElems
            print('read {} elements in {:4.2f} s.'.format(self.__nElem,treadElems) )
            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    def __readFibres(self,fsuffix):
        '''
        This function reads the .lon file
        '''
    
        fibFname = '{}.lon'.format(fsuffix)
        try:
            print('reading Fibers',flush=True)
            tstart       = time()
            with open(fibFname,'r') as ffib:
                ftype    = int(ffib.readline().strip())
                if ftype==1:
                    self.__Fibres = np.zeros(shape=(self.__nElem,3),dtype=np.float32)
                else:
                    self.__Fibres = np.zeros(shape=(self.__nElem,6),dtype=np.float32)
                for jj in range(self.__nElem):
                    row = ffib.readline()
                    row = row.strip().split()
                    for idir,fdir in enumerate(row):
                        self.__Fibres[jj,idir]=float(fdir)
            treadFibers   = time()-tstart
            self.__timeR  += treadFibers
            print('read {} Fibers in {:4.2f} s.'.format(self.__nElem,treadFibers) )
            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
        
        
    

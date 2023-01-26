import numpy as np
import pickle
import os
import sys
from time import time
from gpuSolve.IO.readers.carpmeshreader import CarpMeshReader



def element_contravariant_basis(elemtype,VertPts,localcoords=[]):
    """function element_contravariant_basis(elemType, VertPts,localcoords=[])
    This function evaluates the contravariant basis and the measure
    of an element of type  elemtype.
    Input:
      * VertPts the space position of the element vertices
         Each row corresponds to a point      
      * localCoords = []  local coordinates of the point where the base is evaluated
    Output:
      * vi contra-variant basis vectors
      * meas: the element measure
    """ 

    function_dict = {'Edges': Edge_contravariant_basis,
                     'Trias': Triangle_contravariant_basis,
                     'Quads': None,
                     'Tetras': Tetrahedron_contravariant_basis,
                     'Hexas': None,
                     'Pyras': None,
                     'Prisms': None
                    }  
    return(function_dict[elemtype](VertPts, localcoords) )


def Edge_contravariant_basis(VertPts,localcoords=[]):
    """function Edges_contravariant_basis(VertPts,localcoords=[])
    This function evaluates the contravariant basis and the length
    of an edge element.
    Input:
      * VertPts the space position of the element vertices
         Each row corresponds to a point      
      * localCoords = [] is a dummy input for signature consistence with 
        (future) functions defined on non-linear elements.
    Output:
      * v1, v2, v3: covariant basis
      * meas: the edge measure (length)
    """ 
    try:
        # Tangent space
        v10   = VertPts[1,:]-VertPts[0,:]
        # compute the svd to build a basis of the orthogonal space 
        u,s,v = np.linalg.svd(v10[:,np.newaxis]
        v20   = u[:,1]
        v30   = u[:,2]
        E_len = np.linalg.norm(v10,keepdims=True)        
        # Covariant basis: each row is a vector of the basis
        covbT = np.zeros(shape=(3,3),dtype=np.float32)
        covbT[0,:] = v10
        covbT[1,:] = v20
        covbT[2,:] = v30
        # first two column is the contravariant basis in tangent space
        contrb     = np.linalg.inv(covbT)
        v1contra   = contrb[:,0]
        v2contra   = contrb[:,1]
        v3contra   = contrb[:,2]
        return {'v1':v1contra,'v2':v2contra,'v3':v3contra,'meas':E_len}
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise



def Triangle_contravariant_basis(VertPts,localcoords=[]):
    """function Triangle_contravariant_basis(VertPts,localcoords=[])
    This function evaluates the contravariant basis and the surface
    of a triangular element.
    Input:
      * VertPts the space position of the element vertices
         Each row corresponds to a point      
      * localCoords = [] is a dummy input for signature consistence with 
        (future) functions defined on non-linear elements.
    Output:
      * v1, v2: contra-variant basis vectors
      * v3: the normal vector to the contravariant space
      * meas: the triangle measure (area)
    """ 
    try:
        # Tangent space
        v10   = VertPts[1,:]-VertPts[0,:]
        v20   = VertPts[2,:]-VertPts[0,:]
        # Normal vector
        N12   = np.cross(v10,v20)
        area  = np.linalg.norm(N12,keepdims=True)
        v12   = N12/area
        # Covariant basis: each row is a vector of the basis
        covbT = np.zeros(shape=(3,3),dtype=np.float32)
        covbT[0,:] = v10
        covbT[1,:] = v20
        covbT[2,:] = v12
        # first two columns are the contravariant basis in tangent space
        contrb     = np.linalg.inv(covbT)
        v1contra   = contrb[:,0]
        v2contra   = contrb[:,1]
        v3contra   = contrb[:,2]
        T_area     = 0.5*area
        return {'v1':v1contra,'v2':v2contra,'v3':v3contra,'meas':T_area}
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

def Tetrahedron_contravariant_basis(VertPts,localcoords=[]):
    """function Tetrahedron_contravariant_basis(VertPts,localcoords=[])
    This function evaluates the contravariant basis and the volume
    of a tetrahedral element.
    Input:
      * VertPts the space position of the element vertices.
        Each row corresponds to a point
      * localCoords = [] is a dummy input for signature consistence with 
        (future) functions defined on non-linear elements.#
    Output:
      * v1, v2,v3: contra-variant basis vectors
      * meas: the tetrahedron measure (volume)
    """ 

    try:
        # Tangent space
        v10   = VertPts[1,:]-VertPts[0,:]
        v20   = VertPts[2,:]-VertPts[0,:]
        v30   = VertPts[3,:]-VertPts[0,:]            
        vprod = np.cross(v10,v20)        
        T_vol = np.array(np.abs(np.dot(vprod,v30))/6.0,ndmin=1)
        covbT = np.zeros(shape=(3,3),dtype=np.float32)
        covbT[0,:] = v10
        covbT[1,:] = v20
        covbT[2,:] = v30
        contrb     = np.linalg.inv(covbT)
        v1contra   = contrb[:,0]
        v2contra   = contrb[:,1]
        v3contra   = contrb[:,3]
        return {'v1':v1contra,'v2':v2contra,'v3':v3contra,'meas':T_vol}        
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise


class Triangulation:
    """
    Class Triangulation
    This class defines a domain from Triangulation for numerical simulations.
    It collects the domain mesh and the properties such as fibre directions and conductivities.
    """
    def __init__(self):
        self._Pts          = None
        self._Elems        = None
        self._Fibres       = None
        self._connectivity = None
        self._contravbasis = None
        self._pointRegIDs  = None

    def Pts(self):
        """ function Pts():
        returns the mesh point coordinates
        """
        return(self._Pts)
        
    def Elems(self):
        """ function Elems():
        returns a python dict of the mesh elements.
        Each entry corresponds to an element type (if present on the mesh)
        and is a numpy array of integers (nb of rows equal to the number of elements;
        first n-1 columns are the nodes id; last column is the region id)
        """    
        return(self._Elems)

    def Fibres(self):
        """ function Fibres():
        returns the Fiber directions.
        This field is defined on the elements
        """

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
        Pts    = self._Pts        
        Fibres = self._Fibres
        Elems  = self._Elems        
        mesh0 = {'Pts': Pts,
                 'Elems': Elems,
                 'Fibres': Fibres
                }                 
        with open(foutName,'wb') as fout:
            pickle.dump(mesh0,fout,protocol=pickle.HIGHEST_PROTOCOL)     
        
    def mesh_connectivity(self,storeConn=False):
        """ function connectivity=mesh_connectivity(storeConn=False)
        returns the mesh connectivity. When storeConn=True, it keeps 
        the connectivity as an internal variable, 
        avoiding recomputing in subsequent calls.
        """
        if self._connectivity is None:
            if storeConn==False:
                return(self.__compute_mesh_connectivity())
            else:
                self._connectivity = self.__compute_mesh_connectivity()
                return(self._connectivity)   
        else:
            return(self._connectivity)

    def contravariant_basis(self,storeCbas=False):
        """ function connectivity=contravariant_basis(storeCbas=False)
        returns the contravariant basis evaluated on each element.
        For non-linear elements, it is evaluated at Gauss Points (NOT implemented yet!)
        When storeCbas = True, it keeps a copy of the contravariant_basis
        as an internal variable, avoiding recomputing in subsequent calls.
        """
        if self._contravbasis is None:
            if storeCbas==False:            
                return(self.__compute_contravariant_basis())
            else:
                self._contravbasis = self.__compute_contravariant_basis()
                return(self._contravbasis)   
        else:
            return(self._contravbasis)

    def point_region_ids(self,storeIDs=False):
        """ function regionIds = point_region_ids(storeIDs=False)
        returns the region IDs associated to the mesh vertices.
        When storeIDs = True, it keeps a copy of the point IDs
        as an internal variable, avoiding recomputing in subsequent calls.
        """
        if self._pointRegIDs is None:
            if storeIDs==False:            
                return(self.__compute_point_region_ids())
            else:
                self._pointRegIDs = self.__compute_point_region_ids()
                return(self._pointRegIDs)   
        else:
            return(self._pointRegIDs)

    def element_contravariant_basis(self,elemtype,elemID,localcoords=[]):
        """function element_contravariant_basis(elemtype,elemID,localcoords=[])
        computes the contravariant basis at coordinates localcoords for the 
        element elemID of type elemType
        Input:
            elemtype:     the type of element
            elemID:       the element ID
            localcoords:  the point (in local coordinates) where the contravariant basis is computed
        Output:
            a python dict with:
              v{1,2,3}: the contravariant basis vectors
              meas:       the element measure
        """
        try:
            Elem = self._Elems[elemtype][elemID]
            VertPts = self._Pts[Elem[:-1],:]            
            contraBas = element_contravariant_basis(elemtype,VertPts,localcoords=[])
            return(contraBas)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    
    def release_contravariant_basis(self):
        """function  release_contravariant_basis
        deletes the contravariant basis dictionary and releases the memory
        """
        if self._contravbasis is not None:
            del self._contravbasis
            self._contravbasis = None 
    
    def release_connnectivity(self):
        """function  release_connnectivity
        deletes the connectivity dictionary and releases the memory
        """    
        if self._connectivity is not None:
            del self._connectivity
            self._connectivity = None 

    def release_point_region_ids(self):
        """function  release_point_region_ids
        deletes the numpy array of the point region IDs releases the memory
        """    
        if self._pointRegIDs is not None:
            del self._pointRegIDs
            self._pointRegIDs = None 
    
    def __readMeshPickleFormat(self,fname):
        '''This function reads a mesh in .pkl format
        The input data is a pkl file with a mesh having
        data of type numpy and containing the basic entries 
        of a carp mesh
        '''
        try:
            mesh0        = pickle.load(open(fname,'rb'))
            self._Pts    = mesh0['Pts']
            self._Fibres = mesh0['Fibres']
            self._Elems  = {}
            for key,value in mesh0['Elems'].items():
                if value is not None:
                    self._Elems[key] = value.astype(np.int32)
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
            self._Pts    = reader.Pts()
            self._Fibres = reader.Fibres()
            self._Elems  = {}
            for key,value in reader.Elems().items():
                if value is not None:
                    self._Elems[key] = value.astype(np.int32)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __compute_mesh_connectivity(self):
        print('computing mesh connectivity')
        npt = self._Pts.shape[0]
        connectivity = {}
        t0 = time()
        for jpt in range(npt):
            connectivity[jpt] = []
            #each node is connected with itself
            connectivity[jpt].append(jpt)
        for elemName, Elements in self._Elems.items():
            for Elem in Elements:
                nnodes = Elem.shape[-1] -1
                for ilpt in range(nnodes):
                    iglobalPt = Elem[ilpt]
                    for jlpt in range(1+ilpt,nnodes): 
                        jglobalPt=Elem[jlpt]
                        connectivity[iglobalPt].append(jglobalPt)
                        connectivity[jglobalPt].append(iglobalPt)
        for key,value in connectivity.items():
            connectivity[key] = np.unique(np.array(value,dtype=np.int32))
        elapsed = time() - t0
        print('done in {:3.2f} s'.format(elapsed),flush=True)
        return(connectivity)


    def __compute_contravariant_basis(self):
        '''Temporary implementation; will be modified in the future
        to take into account of Gauss Nodes for non-linear Elements
        '''
        print('computing the contravariant basis and the element measures')
        contravbasis = {}
        t0 = time()
        for elemtype, Elements in self._Elems.items():
            contravbasis[elemtype]={}
            nEl = Elements.shape[0]
            for iElem,Elem in enumerate(Elements):
                VertPts = self._Pts[Elem[:-1],:]
                contraBas = element_contravariant_basis(elemtype,VertPts,localcoords=[])
                for key,value in contraBas.items():
                    if iElem==0:
                        ndim = value.shape[0]
                        contravbasis[elemtype][key] = np.zeros(shape=(nEl,ndim))
                    contravbasis[elemtype][key][iElem,:] = value
            for key,value in contravbasis[elemtype].items():
                contravbasis[elemtype][key]=np.squeeze(value)
        elapsed = time() - t0
        print('done in {:3.2f} s'.format(elapsed),flush=True)
        return(contravbasis)

    def __compute_point_region_ids(self):
        '''This function assigns to each point the region ID
        taking the most recurrent region ID of the elements
        that have the point as a vertex.       
        '''
        print('Associating a region ID to points')
        npt = self._Pts.shape[0]
        regions = {}
        t0 = time()
        for ipt in range(npt):
            regions[ipt] = []
        for elemtype, Elements in self._Elems.items():
            for iElem,Elem in enumerate(Elements):
                rID  = Elem[-1]
                Elem = Elem[:-1]
                for ID in Elem:
                    regions[ID].append(rID)
        pointRegIDs = np.zeros(npt)-1 
        for ipt in range(npt):    
            pointRegIDs = np.argmax(np.bincount(regions[ipt])).astype(np.int32)
        elapsed = time() - t0
        print('done in {:3.2f} s'.format(elapsed),flush=True)
        return(pointRegIDs)


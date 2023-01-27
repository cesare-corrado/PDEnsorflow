import numpy as np


def localStiffness(elemtype,elemData, Sigma=np.identity(3)):
    """function localStiffness(elemtype,elemData, Sigma=np.identity(3))
    returns the local stiffness matrix for element of type elemtype
    and diffusion tensor Sigma (default: identity).
    Material properties are considered uniform on the Element;
    the tensor sigma encodes the anisotropy.
    Input: 
        elemtype: the type of geometric element
        elemData: the dictionary containing the element data
        Sigma:    the diffusion tensor that encodes anisotropies (default: identity)
    Output:
         lmass: a numpy array of the local mass.
    """

    function_dict = {'Edges': linear_Edge_local_Stiffness,
                     'Trias': linear_triangular_local_Stiffness,
                     'Quads': None,
                     'Tetras': linear_tetrahedral_local_Stiffness,
                     'Hexas': None,
                     'Pyras': None,
                     'Prisms': None
                    }  
    return(function_dict[elemtype](elemData,Sigma) )


def linear_Edge_local_Stiffness(elemData, Sigma=np.identity(3) ):
    """linear_Edge_local_Stiffness(elemData, Sigma=np.identity(3) )
    returns the local stiffness matrix for linear 1D elements
    and diffusion tensor Sigma (default: identity).
    Material properties are considered uniform on the trianlge;
    the tensor sigma encodes the anisotropy.
    Input: 
        elemData: the dictionary containing the element data
        Sigma:    the diffusion tensor that encodes anisotropies (default: identity)
    Output:
         lstiff: a numpy array of shape (2X2).    
    """
    EdgeLen         = elemData['meas']
    contra_bas      = np.zeros(shape=(3,1))
    contra_bas[:,0] = elemData['v1']
    localgrad       = np.array([[-1.0,1.0] ] )
    grad            = np.matmul(contra_bas, localgrad)
    flux            = np.matmul(Sigma,grad)
    lstiff          = EdgeLen*np.matmul( grad.T,flux )
    return(lstiff)


def linear_triangular_local_Stiffness(elemData, Sigma=np.identity(3) ):
    """function linear_triangular_local_Stiffness(elemData, Sigma=np.identity(3) )
    returns the local stiffness matrix for linear triangular elements
    and diffusion tensor Sigma (default: identity).
    Material properties are considered uniform on the trianlge;
    the tensor sigma encodes the anisotropy.
    Input: 
        elemData: the dictionary containing the element data
        Sigma:    the diffusion tensor that encodes anisotropies (default: identity)
    Output:
         lstiff: a numpy array of shape (3X3).    
    """
    Tsurf           = elemData['meas']
    contra_bas      = np.zeros(shape=(3,2))
    contra_bas[:,0] = elemData['v1']
    contra_bas[:,1] = elemData['v2']
    localgrad       = np.array([[-1.0,1.0,0.0], [-1.0,0.0,1.0] ] )
    grad            = np.matmul(contra_bas, localgrad)
    flux            = np.matmul(Sigma,grad)
    lstiff          = Tsurf*np.matmul( grad.T,flux )
    return(lstiff)


def linear_tetrahedral_local_Stiffness(elemData, Sigma=np.identity(3) ):
    """function linear_tetrahedral_local_Stiffness(elemData, Sigma=np.identity(3) )
    returns the local stiffness matrix for linear tetrahedral elements
    and diffusion tensor Sigma (default: identity).
    Material properties are considered uniform on the trianlge;
    the tensor sigma encodes the anisotropy.
    Input: 
        elemData: the dictionary containing the element data
        Sigma:    the diffusion tensor that encodes anisotropies (default: identity)
    Output:
         lstiff: a numpy array of shape (4X4).    
    """
    TVol            = elemData['meas']
    contra_bas      = np.zeros(shape=(3,3))
    contra_bas[:,0] = elemData['v1']
    contra_bas[:,1] = elemData['v2']
    contra_bas[:,3] = elemData['v3']
    localgrad       = np.array([[-1.0,1.0,0.0,0.0], [-1.0,0.0,1.0,0.0], [-1.0,0.0,0.0,1.0] ])
    grad            = np.matmul(contra_bas, localgrad)
    flux            = np.matmul(Sigma,grad)
    lstiff          = TVol*np.matmul( grad.T,flux )
    return(lstiff)    

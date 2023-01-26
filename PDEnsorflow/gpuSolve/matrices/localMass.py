import numpy as np


def localMass(elemtype,elemData,props=None):
    """function localMass(elemtype,elemData)
    returns the local mass matrix for element of type elemtype.
    Material properties are considered uniform and unitary
    on the Element.
    Input: 
        elemtype: the type of geometric element
        elemData: the dictionary containing the element data
        props:    a dummy argument
    Output:
         lmass: a numpy array of the local mass.
    """

    function_dict = {'Edges': linear_Edge_local_Mass,
                     'Trias': linear_triangular_local_Mass,
                     'Quads': None,
                     'Tetras': None,
                     'Hexas': None,
                     'Pyras': None,
                     'Prisms': None
                    }  
    return(function_dict[elemtype](elemData,props) )


def linear_Edge_local_Mass(elemData,props=None):
    """function linear_Edge_local_Mass(elemData)
    returns the local mass matrix for linear 1D elements.
    Material properties are considered uniform and unitary
    on the triangle.
    Input: 
        elemData: the dictionary containing the element data
        props: a dummy argument.
    Output:
         lmass: a numpy array of shape (2X2).
    """
    EdgeLen = elemData['meas']
    lmass = np.zeros(shape=(2,2),dtype=np.float32)
    iientry = 1.0/3.0
    ijentry = 1.0/6.0
    for ipt in range(3):
            lmass[ipt,ipt] = iientry
            for jpt in range(1+ipt,3):
                lmass[ipt,jpt] = ijentry
                lmass[jpt,ipt] = ijentry
    lmass = EdgeLen*lmass
    return(lmass)


def linear_triangular_local_Mass(elemData,props=None):
    """function linear_triangular_local_Mass(elemData)
    returns the local mass matrix for linear triangular elements.
    Material properties are considered uniform and unitary
    on the triangle.
    Input: 
        elemData: the dictionary containing the element data
        props: a dummy argument.
    Output:
         lmass: a numpy array of shape (3X3).
    """
    Tsurf = elemData['meas']
    lmass = np.zeros(shape=(3,3),dtype=np.float32)
    iientry = 1.0/12.0
    ijentry = 1.0/24.0
    for ipt in range(3):
            lmass[ipt,ipt] = iientry
            for jpt in range(1+ipt,3):
                lmass[ipt,jpt] = ijentry
                lmass[jpt,ipt] = ijentry
    lmass = 2.0*Tsurf*lmass
    return(lmass)


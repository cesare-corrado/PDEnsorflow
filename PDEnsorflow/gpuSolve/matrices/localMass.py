import numpy as np


def localMass(elemtype: str,elemData: dict, props = None):
    """function localMass(elemtype,elemData,props)
    returns the local mass matrix for element of type elemtype.
    Material properties are considered uniform on the Element: props is the
    scalar coefficient of the mass integral, and None means 1.0, which is the
    mass matrix of a problem that has no such coefficient.
    A coefficient that is constant on the element multiplies the integral
    exactly, so it is a factor on the local matrix rather than a change of the
    quadrature (gpuSolve.physics.diffusion_tensor.mass_beta_weight is the
    monodomain case, where the coefficient is beta/beta_ref).
    Input:
        elemtype: the type of geometric element
        elemData: the dictionary containing the element data
        props:    the scalar coefficient of the integral, or None for 1.0
    Output:
         lmass: a numpy array of the local mass.
    """

    function_dict = {'Edges': linear_Edge_local_Mass,
                     'Trias': linear_triangular_local_Mass,
                     'Quads': None,
                     'Tetras': linear_tetrahedral_local_Mass,
                     'Hexas': None,
                     'Pyras': None,
                     'Prisms': None
                    }  
    lmass = function_dict[elemtype](elemData,props)
    if props is None:
        return(lmass)
    return(props * lmass)


def linear_Edge_local_Mass(elemData : dict, props = None) -> np.ndarray:
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
    nV      = 2
    el_meas = elemData['meas']
    lmass = np.zeros(shape=(nV,nV),dtype=float)
    iientry = 1.0/3.0
    ijentry = 1.0/6.0
    for ipt in range(nV):
            lmass[ipt,ipt] = iientry
            for jpt in range(1+ipt,nV):
                lmass[ipt,jpt] = ijentry
                lmass[jpt,ipt] = ijentry
    lmass = el_meas*lmass
    return(lmass)


def linear_triangular_local_Mass(elemData : dict,props=None) -> np.ndarray :
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
    nV      = 3
    el_meas = elemData['meas']
    lmass = np.zeros(shape=(nV,nV),dtype=float)
    iientry = 1.0/12.0
    ijentry = 1.0/24.0
    for ipt in range(nV):
            lmass[ipt,ipt] = iientry
            for jpt in range(1+ipt,nV):
                lmass[ipt,jpt] = ijentry
                lmass[jpt,ipt] = ijentry
    #Remember: |J|=2*area
    lmass = 2.0*el_meas*lmass
    return(lmass)


def linear_tetrahedral_local_Mass(elemData: dict, props=None) -> np.ndarray:
    """function linear_tetrahedral_local_Mass(elemData)
    returns the local mass matrix for linear tetrahedral elements.
    Material properties are considered uniform and unitary
    on the triangle.
    Input: 
        elemData: the dictionary containing the element data
        props: a dummy argument.
    Output:
         lmass: a numpy array of shape (4X4).
    """
    nV      = 4
    el_meas = elemData['meas']    
    lmass = np.zeros(shape=(nV,nV),dtype=float)
    iientry = 1.0/60.0
    ijentry = 1.0/120.0
    for ipt in range(nV):
            lmass[ipt,ipt] = iientry
            for jpt in range(1+ipt,nV):
                lmass[ipt,jpt] = ijentry
                lmass[jpt,ipt] = ijentry
    #Remember: |J|=6*vol
    lmass = 6.0*el_meas*lmass
    return(lmass)


#!/usr/bin/env python
"""
    The conductivity tensor of an anisotropic monodomain problem, as the
    `stiffness` material function of HeatSolver / MonodomainSolver.

    The tensor is transversely isotropic around the local fibre direction f:

        Sigma = sigma_t I + (sigma_l - sigma_t) f (x) f

    so it has sigma_l along the fibre and sigma_t in every direction across it.
    Without fibres the element is isotropic and the tensor is sigma_l I.

    The optional element property `beta` divides the tensor. It is the membrane
    surface area per unit tissue volume, which appears in the monodomain
    equation as

        div(sigma grad V) = beta (Cm dV/dt + I_ion)

    so the diffusion coefficient the solver needs is sigma/(beta Cm), while the
    ionic term carries no beta at all. Cm is folded into the cell model, so the
    division by beta is the whole of it. beta is OPTIONAL and its absence means
    1.0: a caller that never registers it gets exactly the numbers it got
    before this function existed.

    The identical division is applied by the vectorised assembly path in
    gpuSolve.matrices.globalMatrices, which builds the same tensor without
    calling this function when the properties are region-based.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import numpy as np

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties


def conductivity_tensor(elemtype: str, iElem: int, domain: Triangulation,
                        matprop: MaterialProperties) -> np.ndarray:
    """ conductivity_tensor(elemtype, iElem, domain, matprop) returns the 3x3
        conductivity tensor of one element, built from the element properties
        sigma_l, sigma_t and the optional beta
    """
    try:
        regionID = domain.Elems()[elemtype][iElem, -1]
        sigma_l  = matprop.ElementProperty('sigma_l', elemtype, iElem, regionID)
        sigma_t  = sigma_l
        if matprop.element_property_type('sigma_t') is not None:
            sigma_t = matprop.ElementProperty('sigma_t', elemtype, iElem, regionID)
        # probe first: ElementProperty raises on a property that was never
        # registered, and beta is optional by design
        if matprop.element_property_type('beta') is not None:
            beta = matprop.ElementProperty('beta', elemtype, iElem, regionID)
            if beta <= 0.0:
                raise ValueError('beta must be positive, got {} on element {} of '
                                 'region {}'.format(beta, iElem, regionID))
            sigma_l = sigma_l / beta
            sigma_t = sigma_t / beta
        fibres = domain.Fibres()
        if fibres is None:
            # no fibre field: there is no distinguished direction, so the
            # longitudinal value describes the whole element
            return(sigma_l * np.eye(3))
        fib   = fibres[iElem, :]
        Sigma = sigma_t * np.eye(3)
        for ii in range(3):
            for jj in range(3):
                Sigma[ii, jj] = Sigma[ii, jj] + (sigma_l - sigma_t) * fib[ii] * fib[jj]
        return(Sigma)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise


def no_mass_property(elemtype: str, iElem: int, domain: Triangulation,
                     matprop: MaterialProperties):
    """ no_mass_property(elemtype, iElem, domain, matprop) is the inert `mass`
        material function: the mass matrix of these problems carries no
        material coefficient, and the assembler expects a function to be
        registered for every matrix it builds
    """
    return(None)

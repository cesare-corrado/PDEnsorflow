#!/usr/bin/env python
"""
    The conductivity tensor of an anisotropic monodomain problem, as the
    `stiffness` material function of HeatSolver / MonodomainSolver.

    The tensor is transversely isotropic around the local fibre direction f:

        Sigma = sigma_t I + (sigma_l - sigma_t) f (x) f

    so it has sigma_l along the fibre and sigma_t in every direction across it.
    Without fibres the element is isotropic and the tensor is sigma_l I.

    The optional element property `beta` is the membrane surface area per unit
    tissue volume, which appears in the monodomain equation as

        div(sigma grad V) = beta (Cm dV/dt + I_ion)

    Dividing the equation by beta moves it out of sight, and that is what this
    function does when `beta` alone is registered: the diffusion coefficient
    becomes sigma/(beta Cm) and the ionic term carries no beta at all (Cm is
    folded into the cell model, so the division is the whole of it).

    THAT IS ONLY LEGITIMATE WHILE beta IS UNIFORM. Dividing by a function of
    space changes the operator, because

        div((sigma/beta) grad V) != (1/beta) div(sigma grad V)

    the difference being a spurious term in grad(beta). A beta that varies
    between regions therefore belongs where the equation puts it, on the
    membrane side, as the coefficient of the mass integral (see
    mass_beta_weight). Registering the uniform property `beta_ref` selects that
    form: the tensor is then divided by that single reference value, which is
    dividing the equation by a CONSTANT and is always allowed, and the varying
    part travels with the mass matrix as beta/beta_ref.

    So there are three cases, and the first two are the same arithmetic:
      * no `beta`:                 the bare conductivity, as before it existed;
      * `beta` alone:              sigma/beta, exact for a uniform beta and the
                                   behaviour of every caller written before
                                   `beta_ref`;
      * `beta` and `beta_ref`:     sigma/beta_ref here, beta/beta_ref on the
                                   mass. Identical to the previous case
                                   wherever beta equals beta_ref, so a uniform
                                   beta keeps its numbers to the bit.

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
            # beta_ref, when the caller registers it, is the only thing the
            # tensor is divided by: the rest of beta weights the mass integral
            if matprop.element_property_type('beta_ref') is not None:
                beta = matprop.ElementProperty('beta_ref', elemtype, iElem, regionID)
            else:
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


def mass_beta_weight(elemtype: str, iElem: int, domain: Triangulation,
                     matprop: MaterialProperties) -> float:
    """ mass_beta_weight(elemtype, iElem, domain, matprop) returns the
        coefficient of the mass integral of one element, beta/beta_ref, or None
        when the caller did not register `beta_ref` and the mass therefore
        carries no coefficient at all.
        It is the membrane side of the monodomain equation: the element's own
        beta divided by the single reference the conductivity was divided by
        (see conductivity_tensor). beta is piecewise constant per element, since
        an element carries one region tag and a region has one beta, so

            integral over the element of beta phi_i phi_j
                  = beta_e * (integral over the element of phi_i phi_j)

        is EXACT and the coefficient is a scalar factor on the local matrix. No
        averaging of nodal values is involved, and none is wanted: resolving
        beta onto the nodes first would discard, at every region interface,
        which of the elements meeting there had which beta.
    """
    if matprop.element_property_type('beta_ref') is None:
        return(None)
    if matprop.element_property_type('beta') is None:
        # beta_ref is the value the conductivity was divided by, so without the
        # beta it is a reference to there is no coefficient to form. Said here,
        # because the lookup below would only report a missing dictionary key
        raise ValueError('the mass integral carries beta/beta_ref, but only beta_ref is '
                         'registered: register the element property beta as well, or ask '
                         'for the mass with no coefficient (no_mass_property)')
    regionID = domain.Elems()[elemtype][iElem, -1]
    beta     = matprop.ElementProperty('beta', elemtype, iElem, regionID)
    beta_ref = matprop.ElementProperty('beta_ref', elemtype, iElem, regionID)
    if beta <= 0.0 or beta_ref <= 0.0:
        raise ValueError('beta and beta_ref must be positive, got {} and {} on element '
                         '{} of region {}'.format(beta, beta_ref, iElem, regionID))
    return(beta / beta_ref)


def no_mass_property(elemtype: str, iElem: int, domain: Triangulation,
                     matprop: MaterialProperties):
    """ no_mass_property(elemtype, iElem, domain, matprop) is the inert `mass`
        material function: the mass matrix carries no material coefficient, and
        the assembler expects a function to be registered for every matrix it
        builds. It is what a problem with no beta, or with a beta that divides
        the conductivity, registers; mass_beta_weight is the one that carries a
        varying beta.
    """
    return(None)

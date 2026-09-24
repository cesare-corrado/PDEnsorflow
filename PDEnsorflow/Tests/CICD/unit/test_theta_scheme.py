#!/usr/bin/env python
"""
    Tier-1 unit tests for the theta method of the diffusion step
    (HeatSolver, inherited by MonodomainSolver):

        (M + theta dt K) U^{n+1} = (M - (1 - theta) dt K) U^n

    On a 1D cable with no forcing the space-discrete problem M U' = -K U has
    the exact solution U(t) = expm(-t M^-1 K) U0, computed here densely. The
    error of the time stepping against it must fall with dt at the order of
    the scheme: 2 for Crank-Nicolson (theta = 0.5), 1 for implicit Euler
    (theta = 1). This separates the time error from the space error, which the
    exact solution shares.

    CPU-only, 21 nodes.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle

import numpy as np
import pytest
import scipy.linalg

from gpuSolve.physics import HeatSolver

_NELEM  = 20
_LENGTH = 1.0
_SIGMA  = 0.01
_TEND   = 2.0
_SOURCE = 0.5                                     # constant source on half the cable


def _write_cable(fname: str):
    npt   = _NELEM + 1
    Pts   = np.zeros(shape=(npt, 3), dtype=np.float64)
    Pts[:, 0] = np.linspace(0.0, _LENGTH, npt)
    Edges = np.zeros(shape=(_NELEM, 3), dtype=np.int32)
    Edges[:, 0] = np.arange(0, _NELEM)
    Edges[:, 1] = np.arange(1, _NELEM + 1)
    Edges[:, 2] = 1
    with open(fname, 'wb') as fout:
        pickle.dump({'Pts': Pts, 'Elems': {'Edges': Edges}, 'Fibres': None}, fout,
                    protocol=pickle.HIGHEST_PROTOCOL)


def _no_mass(elemtype, iElem, domain, matprop):
    return(None)


def _sigma(elemtype, iElem, domain, matprop) -> np.ndarray:
    return(_SIGMA * np.eye(3))


def _run(mesh: str, dt: float, theta: float, source: bool = False, split: bool = False) -> tuple:
    """Diffuse a cosine to _TEND, with the constant source _SOURCE on the left
    half of the cable when source is True; returns (final U, initial U) in
    float64."""
    model = HeatSolver({'mesh_file_name': mesh, 'dt': dt, 'Tend': _TEND, 'theta': theta,
                        'dt_per_plot': 1, 'split_source': split})
    model.add_material_function('mass', _no_mass)
    model.add_material_function('stiffness', _sigma)
    model.assemble_matrices()
    x  = model.domain().Pts()[:, 0]
    U0 = (np.cos(np.pi * x / _LENGTH) + 0.5 * np.cos(3.0 * np.pi * x / _LENGTH)).astype(np.float32)
    model.set_initial_condition(U0)
    if source:
        # on for the whole run; on half the cable only, because K maps a
        # uniform source to zero and the splitting error would vanish
        model.add_stimulus(x < 0.5 * _LENGTH, {'tstart': 0.0, 'nstim': 1, 'period': 1.0e6,
                                                'duration': 1.0e6, 'intensity': _SOURCE})
    model.solver().set_toll(1.0e-12)
    model.solver().set_maxiter(10 * _NELEM)
    model.finalize_for_run()
    ctime = 0.0
    for _ in range(model.nt()):
        ctime += dt
        model.step(ctime)
    return(np.reshape(model.U().numpy(), (-1,)).astype(np.float64), U0.astype(np.float64))


def _exact(U0: np.ndarray, source: bool = False) -> np.ndarray:
    """The exact solution of M U' = -K U + M f at _TEND with the linear-element
    M and K of the cable: expm(-T M^-1 K) U0, plus, with the source, the
    integral of expm(-s M^-1 K) f over [0, T], taken from the exponential of
    the augmented matrix [[-A, f], [0, 0]] (A = M^-1 K is singular, so A^-1
    cannot be used)."""
    h = _LENGTH / _NELEM
    n = _NELEM + 1
    M = np.zeros((n, n))
    K = np.zeros((n, n))
    for e in range(_NELEM):
        idx = np.ix_([e, e + 1], [e, e + 1])
        M[idx] += h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        K[idx] += _SIGMA / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
    A = np.linalg.solve(M, K)
    if not source:
        return(scipy.linalg.expm(-_TEND * A) @ U0)
    x = np.linspace(0.0, _LENGTH, n)
    f = np.where(x < 0.5 * _LENGTH, _SOURCE, 0.0)
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = -A
    aug[:n, n] = f
    E = scipy.linalg.expm(_TEND * aug)
    return(E[:n, :n] @ U0 + E[:n, n])


@pytest.fixture(scope='module')
def cable(tmp_path_factory) -> str:
    mesh = str(tmp_path_factory.mktemp('theta') / 'cable.pkl')
    _write_cable(mesh)
    return(mesh)


@pytest.mark.parametrize('theta,order', [(0.5, 2.0), (1.0, 1.0)])
def test_the_time_error_falls_at_the_order_of_the_scheme(cable, theta, order):
    errors = []
    for dt in (0.2, 0.1, 0.05):
        U, U0 = _run(cable, dt, theta)
        errors.append(np.max(np.abs(U - _exact(U0))))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(np.abs(rates - order) < 0.2), 'theta={}: errors {} rates {}'.format(theta, errors, rates)


def test_crank_nicolson_is_more_accurate_than_implicit_euler(cable):
    U_cn, U0 = _run(cable, 0.1, 0.5)
    U_ie, _  = _run(cable, 0.1, 1.0)
    exact    = _exact(U0)
    assert np.max(np.abs(U_cn - exact)) < 0.1 * np.max(np.abs(U_ie - exact))


def test_a_theta_outside_0_1_is_refused():
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match='theta'):
            HeatSolver({'theta': bad})


@pytest.mark.parametrize('split,order', [(False, 2.0), (True, 1.0)])
def test_with_a_source_only_the_unsplit_form_stays_second_order(cable, split, order):
    """With a source that does not depend on U, the unsplit Crank-Nicolson step
    keeps order 2; the split one (the reference's scheme) carries the
    -(1-theta) dt^2 K S splitting error and drops to order 1."""
    errors = []
    for dt in (0.2, 0.1, 0.05):
        U, U0 = _run(cable, dt, 0.5, source=True, split=split)
        errors.append(np.max(np.abs(U - _exact(U0, source=True))))
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(np.abs(rates - order) < 0.25), 'split={}: errors {} rates {}'.format(split, errors, rates)


def test_for_implicit_euler_the_two_forms_are_the_same_scheme(cable):
    U_unsplit, _ = _run(cable, 0.1, 1.0, source=True, split=False)
    U_split, _   = _run(cable, 0.1, 1.0, source=True, split=True)
    assert np.array_equal(U_unsplit, U_split)

#!/usr/bin/env python
"""
    What the place of beta changes, measured on a cable with a beta jump.

    beta, the membrane surface area per unit tissue volume, belongs on the
    membrane side of the monodomain equation

        beta (Cm dV/dt + I_ion) = div(sigma grad V)

    This library used to divide the equation through by beta and hand the
    solver sigma/beta, which is exact while beta is uniform and not an identity
    otherwise: div((sigma/beta) grad V) is not (1/beta) div(sigma grad V). The
    current formulation divides by one constant reference instead and keeps the
    varying part as the coefficient beta/beta_ref of the mass integral.

    Away from an interface the two agree, and that is worth understanding
    before reading the numbers: inside a region where beta is constant, the new
    form reduces to dV/dt = div((sigma/beta) grad V) + ..., the same local
    diffusivity the old form used. The whole difference is at the jump, where
    the old form effectively diffuses a quantity it should not and the new one
    weights the two sides by their own beta. So this script measures the
    arrival times along a cable whose second half has a different beta, and
    reports the difference the formulation makes at and after the interface.

    Run by hand, on either device:

        python beta_jump_cable.py

    Both runs use the same mesh, the same cell model and the same sigma; only
    where beta enters changes. The printed table is the point: the two columns
    agree in the first half to the time step and separate from the interface on.

    What it produced on an RTX A2000, with beta going from 0.14 to 0.56 across
    the middle of a 20 mm cable (sigma 174 um^2/ms, so the local diffusivity
    falls from 1243 to 311 um^2/ms):

              x (mm)  beta in sigma   beta in mass   difference
                0.00          0.000          0.000        0.000
                5.00         65.250         65.250        0.000
                9.95        145.300        145.300        0.000
               10.00        146.000        146.650        0.650  <- interface
               10.05        147.850        148.300        0.450
               15.00        319.950        320.350        0.400
               20.00        493.000        493.400        0.400

        max |difference| before the interface: 0.0500 ms
        max |difference| after  the interface: 0.4500 ms

    Read it as three statements. Before the interface the two formulations
    agree to one time step, which is what the algebra says: where beta is
    constant, dividing by it is an identity. At the interface they part, and
    the front needs 0.65 ms longer when beta sits on the mass matrix. After it
    the difference stays flat at about 0.4 ms, so the local conduction velocity
    in the second half is the same in both (10 mm in 347 ms, 28.8 um/ms,
    against 68.5 um/ms in the first half, the ratio of about 2.4 that a
    fourfold drop in diffusivity gives): the formulation changes the crossing,
    not the propagation. A simulation with regions of different
    cellSurfVolRatio therefore accumulates one such error per interface
    crossing, silently, in both codes.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle
import tempfile

import numpy as np

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import mass_beta_weight
from gpuSolve.physics import no_mass_property


_NELEM  = 400                                    # 20 mm of cable at 50 um
_DX     = 50.0                                   # micrometres
_SIGMA  = 174.0                                  # um^2/ms, i.e. 0.174 S/m with beta folded out
_BETA_L = 0.14                                   # the reference, left half
_BETA_R = 0.56                                   # four times as much membrane per volume
_DT     = 0.05                                   # ms
_TEND   = 700.0                                  # ms, enough for both halves
_THRESH = -40.0                                  # mV, the activation threshold
# the cell parameters of the Tests/FEM/mMS example, which this cable reproduces
# in 1D: they are the set that example shows a propagating front with
_TAU    = {'tau_in': 0.15, 'tau_out': 1.5, 'tau_open': 105.0, 'tau_close': 120.0}
_VGATE  = 0.1


def _write_cable(fname: str):
    """A uniform cable, tag 1 on the left half and tag 2 on the right.
    It carries a fibre field along its axis, which is what every mesh of the
    reference's format has (its .lon file) and what sends the assembly down the
    vectorised path: the numbers below are then the ones a real run gets.
    """
    npt         = _NELEM + 1
    Pts         = np.zeros(shape=(npt, 3), dtype=np.float64)
    Pts[:, 0]   = _DX * np.arange(npt)
    Edges       = np.zeros(shape=(_NELEM, 3), dtype=np.int32)
    Edges[:, 0] = np.arange(0, _NELEM)
    Edges[:, 1] = np.arange(1, _NELEM + 1)
    Edges[:, 2] = np.where(np.arange(_NELEM) < _NELEM // 2, 1, 2)
    Fibres       = np.zeros(shape=(_NELEM, 3), dtype=np.float64)
    Fibres[:, 0] = 1.0
    mesh = {'Pts': Pts, 'Elems': {'Edges': Edges}, 'Fibres': Fibres}
    with open(fname, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)


def _activation_times(mesh_file: str, weighted: bool) -> np.ndarray:
    """Runs the cable once and returns the first crossing time of every node.
    weighted=False is the old formulation (beta divides the conductivity),
    weighted=True the current one (beta_ref divides it, beta/beta_ref weights
    the mass integral).
    """
    # renumbering off: step() then returns U in the mesh's own node order, so
    # the arrival times below can be indexed by position along the cable
    model = MonodomainSolver(ModifiedMS2v(dt=_DT),
                             {'dt': _DT, 'tend': _TEND, 'use_renumbering': False})
    model.loadMesh(mesh_file)
    beta = {1: _BETA_L, 2: _BETA_R}
    model.add_element_material_property('sigma_l', 'region', {1: _SIGMA, 2: _SIGMA})
    model.add_element_material_property('sigma_t', 'region', {1: _SIGMA, 2: _SIGMA})
    model.add_element_material_property('beta', 'region', beta)
    if weighted:
        model.add_element_material_property('beta_ref', 'uniform', _BETA_L)
        model.add_material_function('mass', mass_beta_weight)
    else:
        model.add_material_function('mass', no_mass_property)
    for pname, value in _TAU.items():
        model.add_nodal_material_property(pname, 'uniform', value)
    model.add_nodal_material_property('u_gate', 'uniform', _VGATE)
    model.add_nodal_material_property('u_crit', 'uniform', _VGATE)
    model.add_material_function('stiffness', conductivity_tensor)
    model.assemble_matrices()
    # a depolarised block at the left end starts the front, as the 2D example
    # does: it needs no stimulus protocol and cannot fail to excite
    npt = _NELEM + 1
    U0  = np.where(np.arange(npt) < 20, 20.0, -80.0).astype(np.float32)
    model.set_initial_condition(U0)
    model.finalize_for_run()
    acts = np.zeros(shape=(npt,), dtype=np.float64) - 1.0
    nt   = int(round(_TEND / _DT))
    for istep in range(nt):
        ctime = istep * _DT
        U     = np.reshape(np.asarray(model.step(ctime)), (-1,))
        fresh = np.logical_and(acts < 0.0, U > _THRESH)
        acts[fresh] = ctime
        if np.all(acts >= 0.0):
            break
    return(acts)


def main():
    print('A cable with a beta jump: {} nodes, sigma = {} um^2/ms everywhere,'.format(
        _NELEM + 1, _SIGMA))
    print('beta = {} on the left half and {} on the right.'.format(_BETA_L, _BETA_R))
    print('Local diffusivity sigma/beta: {:.1f} and {:.1f} um^2/ms, so the right'.format(
        _SIGMA / _BETA_L, _SIGMA / _BETA_R))
    print('half is the slower one in BOTH formulations; only the jump differs.\n')
    with tempfile.TemporaryDirectory() as folder:
        mesh_file = os.path.join(folder, 'cable.pkl')
        _write_cable(mesh_file)
        old = _activation_times(mesh_file, weighted=False)
        new = _activation_times(mesh_file, weighted=True)
    mid = _NELEM // 2
    print('{:>12s} {:>14s} {:>14s} {:>12s}'.format('x (mm)', 'beta in sigma', 'beta in mass',
                                                   'difference'))
    for node in (0, mid // 2, mid - 1, mid, mid + 1, mid + mid // 2, _NELEM):
        mark = '  <- interface' if node == mid else ''
        print('{:12.2f} {:14.3f} {:14.3f} {:12.3f}{}'.format(
            node * _DX * 1.0e-3, old[node], new[node], new[node] - old[node], mark))
    reached_old = int(np.count_nonzero(old >= 0.0))
    reached_new = int(np.count_nonzero(new >= 0.0))
    print('\nnodes activated: {} of {} (beta in sigma), {} of {} (beta in mass)'.format(
        reached_old, _NELEM + 1, reached_new, _NELEM + 1))
    left  = np.max(np.abs(new[:mid] - old[:mid]))
    right = np.max(np.abs(new[mid + 1:] - old[mid + 1:]))
    print('max |difference| before the interface: {:.4f} ms'.format(left))
    print('max |difference| after  the interface: {:.4f} ms'.format(right))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
    Tier-1 regression tests for the node order the solver's accessors report in
    during the set-up, before finalize_for_run().

    The renumbering has two node orders: the user's, which the mesh and every
    input are in, and the solver's, which the RCM reordering produces.
    `finalize_for_run()` is what moves the nodal quantities from the first into
    the second. Everything that reports a nodal array to the caller has to undo
    that permutation *only once it has been applied*.

    `U()` and `MonodomainSolver.ionic_state()` used to undo it whenever
    renumbering was switched on, without asking whether it had been applied yet.
    Between `assemble_matrices()` (which computes the permutation) and
    `finalize_for_run()` (which applies it) they therefore returned a scrambled
    array: `U()` disagreed with the initial condition that had just been set and
    with `checkpoint()['Vm']`, which was right. Earlier still, before
    `assemble_matrices()`, the permutation is None and the gather raised.

    Every caller in the tree reads these during or after the run, so the defect
    was latent; a set-up step that works in the user's order, such as prepacing,
    is what meets it. These tests pin the contract at the two moments that
    matter, on a mesh where the permutation is not the identity.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import no_mass_property
from gpuSolve.ionic.mms2v import ModifiedMS2v


_NELEM = 40
_NPT   = _NELEM + 1
_DX    = 100.0                                    # micrometres
_DT    = 0.1                                      # ms


def _write_cable(folder: str):
    with open(os.path.join(folder, 'cable.pts'), 'w') as fout:
        fout.write('{}\n'.format(_NPT))
        for ipt in range(_NPT):
            fout.write('{:.6f} 0.000000 0.000000\n'.format(ipt * _DX))
    with open(os.path.join(folder, 'cable.elem'), 'w') as fout:
        fout.write('{}\n'.format(_NELEM))
        for iel in range(_NELEM):
            fout.write('Ln {} {} 1\n'.format(iel, iel + 1))
    with open(os.path.join(folder, 'cable.lon'), 'w') as fout:
        fout.write('1\n')
        for _iel in range(_NELEM):
            fout.write('1.0 0.0 0.0\n')


@pytest.fixture(scope='module')
def cable(tmp_path_factory) -> str:
    folder = str(tmp_path_factory.mktemp('node_order'))
    _write_cable(folder)
    return(folder)


def _solver(folder: str) -> MonodomainSolver:
    config = {'mesh_file_name': os.path.join(folder, 'cable'),
              'use_renumbering': True, 'dt': _DT, 'Tend': 1.0, 'dt_per_plot': 1}
    solver = MonodomainSolver(ModifiedMS2v(dt=_DT), config)
    solver.add_element_material_property('sigma_l', 'uniform', 0.174)
    solver.add_element_material_property('sigma_t', 'uniform', 0.174)
    solver.add_element_material_property('beta', 'uniform', 1400.0)
    solver.add_material_function('mass', no_mass_property)
    solver.add_material_function('stiffness', conductivity_tensor)
    return(solver)


def _ramp() -> np.ndarray:
    """An initial condition that differs at every node, so any permutation of
    it is visible. A uniform one would hide the defect completely."""
    return(np.linspace(-80.0, 20.0, _NPT).astype(np.float32))


def test_the_permutation_is_not_the_identity(cable):
    """Guards the tests below: on a mesh the reordering leaves alone, they
    would pass whatever the accessors do."""
    solver = _solver(cable)
    solver.assemble_matrices()
    perm = solver.renumbering()['perm']
    assert not np.array_equal(np.asarray(perm), np.arange(_NPT))


def test_the_potential_reads_back_as_it_was_set_before_finalize(cable):
    """The defect: U() undid a permutation that finalize_for_run() had not yet
    applied, so it returned neither the initial condition nor what
    checkpoint() reported."""
    solver = _solver(cable)
    solver.assemble_matrices()
    U0 = _ramp()
    solver.set_initial_condition(U0)
    assert not solver.ready_for_run()
    assert np.array_equal(np.reshape(solver.U().numpy(), (-1,)), U0)
    assert np.array_equal(np.reshape(solver.checkpoint()['Vm'], (-1,)), U0)


def test_the_potential_reads_back_as_it_was_set_after_finalize(cable):
    """The other half of the contract: once the permutation is applied, it
    must be undone, so the caller sees the same array either way."""
    solver = _solver(cable)
    solver.assemble_matrices()
    U0 = _ramp()
    solver.set_initial_condition(U0)
    solver.finalize_for_run()
    assert solver.ready_for_run()
    assert np.array_equal(np.reshape(solver.U().numpy(), (-1,)), U0)
    assert np.array_equal(np.reshape(solver.checkpoint()['Vm'], (-1,)), U0)


def test_the_potential_reads_back_before_the_permutation_exists(cable):
    """Before assemble_matrices() there is no renumbering at all; the accessor
    used to index None and raise."""
    solver = _solver(cable)
    U0     = _ramp()
    solver.set_initial_condition(U0)
    assert np.array_equal(np.reshape(solver.U().numpy(), (-1,)), U0)


def test_a_cell_state_variable_reads_back_in_the_user_order(cable):
    """ionic_state() carried the same defect as U(), and is the accessor a
    caller reaches for to look at the cell model node by node."""
    solver = _solver(cable)
    solver.assemble_matrices()
    solver.set_initial_condition(_ramp())
    name   = solver.ionic_model().state_variable_names()[0]
    # the state variable is uniform at rest, and a permutation of a uniform
    # array is invisible: it is given a value that differs at every node first,
    # or this test passes whatever the accessor does
    solver.ionic_model().set_state_variables(
        {name: np.linspace(0.1, 0.9, _NPT).astype(np.float32)})
    before = np.reshape(solver.ionic_state('_{}'.format(name)).numpy(), (-1,))
    # checkpoint() is in the user's order by construction, so it is the
    # reference the accessor has to agree with, at both moments
    assert np.array_equal(before, np.reshape(
        solver.checkpoint()['state_variables'][name], (-1,)))
    solver.finalize_for_run()
    after = np.reshape(solver.ionic_state('_{}'.format(name)).numpy(), (-1,))
    assert np.array_equal(after, np.reshape(
        solver.checkpoint()['state_variables'][name], (-1,)))
    assert np.array_equal(before, after)

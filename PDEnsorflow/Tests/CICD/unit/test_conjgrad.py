#!/usr/bin/env python
"""
    Tier-1 unit test for gpuSolve.linearsolvers.ConjGrad (Jacobi-preconditioned).

    Method (mirrors DEVTESTS/ConjugateGradients/ConjugateGradients.py):
      1. assemble the FEM system matrix A = alpha*M + beta*K on a small mesh;
      2. prescribe a gold-truth solution x_star (all-ones, and a fixed random
         vector);
      3. build the right-hand side b = A x_star;
      4. solve A x = b from a zero initial guess and check that the solver
         recovers x_star, and drives the relative residual down, within
         tolerance.

    A well conditioned system (alpha = beta = 1) is used so the gold-truth
    solution is recovered tightly in float32. The tolerance passed to the solver
    is relative to ||b|| (ConjGrad tests the absolute residual internally). This
    also guards against a regression of the JacobiPrecond._V drop
    (build_preconditioner must populate self._V).

    Note: the default ConjGrad path keeps intermediate vectors (r, p) on the
    solver object between @tf.function calls, so it must run eagerly -- exactly
    as the FEM demos do (tf.config.run_functions_eagerly(True)).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict, csr_axpby
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.linearsolvers.jacobi_precond import JacobiPrecond


@pytest.fixture(scope='module', autouse=True)
def _eager_mode():
    """ConjGrad stores r/p between traced calls, so it must run eagerly. Set the
    flag in fixture *setup* (not at import time) so it holds regardless of the
    eager state another test module may have left behind, then restore it."""
    prev = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(bool(prev))


def _dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """Empty mass-property callback."""
    return(None)


def _sigma_tensor(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties) -> np.ndarray:
    """Isotropic unit diffusion tensor for the stiffness matrix."""
    fib     = domain.Fibres()[iElem, :]
    rID     = domain.Elems()[elemtype][iElem, -1]
    sigma_l = matprop.ElementProperty('sigma_l', elemtype, iElem, rID)
    sigma_t = matprop.ElementProperty('sigma_t', elemtype, iElem, rID)
    Sigma   = sigma_t * np.eye(3)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii, jj] = Sigma[ii, jj] + (sigma_l - sigma_t) * fib[ii] * fib[jj]
    return(Sigma)


def _build_system_matrix(mesh_file: str, alpha: float, beta: float):
    """Assemble A = alpha*M + beta*K on mesh_file; return (A, npt)."""
    diffusl = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    diffust = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    domain = Triangulation()
    domain.readMesh(mesh_file)
    npt = domain.Pts().shape[0]

    materials = MaterialProperties()
    materials.add_element_property('sigma_l', 'region', diffusl)
    materials.add_element_property('sigma_t', 'region', diffust)
    materials.add_ud_function('mass', _dfmass)
    materials.add_ud_function('stiffness', _sigma_tensor)

    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    lmatr        = {'mass': localMass, 'stiffness': localStiffness}
    matrices     = assemble_matrices_dict(lmatr, pattern, domain, materials, connectivity)

    A = csr_axpby(matrices['mass'], alpha, matrices['stiffness'], beta)
    materials.remove_all_element_properties()
    domain.release_connectivity()
    return(A, npt)


def _solve(A, B: tf.Tensor, npt: int, maxiter: int, toll: float):
    """Solve A x = B from a zero guess with a Jacobi-preconditioned CG.
    Returns (X, niters).
    """
    A_st    = A.to_sparse_tensor()
    solver  = ConjGrad({'maxiter': maxiter, 'toll': toll, 'verbose': False, 'use_graph_loop': False})
    solver.set_matrix(A)
    precond = JacobiPrecond()
    precond.build_preconditioner(A_st.indices.numpy()[:, 0],
                                 A_st.indices.numpy()[:, 1],
                                 A_st.values.numpy(),
                                 int(A_st.dense_shape.numpy()[0]))
    solver.set_precond(precond)
    solver.set_X0(tf.zeros(shape=(npt, 1), dtype=tf.float32))
    solver.set_RHS(B)
    solver.solve()
    return(solver.X(), int(solver._niters))


@pytest.fixture(scope='module')
def system_matrix(data_dir):
    """Assemble A = M + K on the coarse square mesh once for the module."""
    mesh_file = os.path.join(data_dir, 'triangulated_square.pkl')
    A, npt    = _build_system_matrix(mesh_file, alpha=1.0, beta=1.0)
    return({'A': A, 'npt': npt})


@pytest.mark.parametrize('gold', ['ones', 'random'])
def test_cg_recovers_gold_solution(system_matrix, gold):
    """CG must recover a prescribed gold-truth solution x* from b = A x*."""
    A       = system_matrix['A']
    npt     = system_matrix['npt']
    maxiter = 1000

    if gold == 'ones':
        Xstar = np.ones(shape=(npt, 1), dtype=np.float32)
    else:
        rng   = np.random.default_rng(0)
        Xstar = rng.standard_normal(size=(npt, 1)).astype(np.float32)
    Xstar_tf = tf.constant(Xstar, dtype=tf.float32)

    # b = A x*
    B      = tf.raw_ops.SparseMatrixMatMul(a=A._matrix, b=Xstar_tf)
    b_norm = float(tf.norm(B))

    # ConjGrad's convergence test is on the *absolute* residual norm, so target
    # a small residual relative to ||b|| by scaling the tolerance.
    toll = 1.0e-6 * b_norm
    X, niters = _solve(A, B, npt, maxiter=maxiter, toll=toll)

    # relative residual ||A x - b|| / ||b|| and relative solution error
    AX      = tf.raw_ops.SparseMatrixMatMul(a=A._matrix, b=X)
    rel_res = float(tf.norm(AX - B) / b_norm)
    rel_err = float(tf.norm(X - Xstar_tf) / tf.norm(Xstar_tf))
    print('\n[CG {0}] npt={1} niters={2} rel_res={3:.3e} rel_err={4:.3e}'.format(
        gold, npt, niters, rel_res, rel_err))

    # Primary check: the solver recovers the gold-truth solution.
    assert niters < maxiter, 'CG hit maxiter ({0}) without converging'.format(maxiter)
    assert rel_err < 1.0e-3, 'gold-truth solution not recovered: rel_err={0:.3e}'.format(rel_err)
    # Loose residual sanity: for x*=ones, K annihilates the constant so ||b|| is
    # tiny and the recursively-updated CG residual drifts from the true residual
    # in float32 (true rel_res ~1e-3 while the solution itself is accurate to
    # ~3e-5). This bound only guards against gross non-convergence / divergence.
    assert rel_res < 1.0e-2, 'residual too large: rel_res={0:.3e}'.format(rel_res)

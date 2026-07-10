#!/usr/bin/env python
"""
    Tier-1 unit test for csr_axpby (gpuSolve.matrices.globalMatrices): the
    CSR-native "alpha*A + beta*B" used to build FEM system matrices (e.g.
    A = M + dt*K in heatSolver, and -- once per Newton iteration -- in a future
    nonlinear solve).

    The two-matrix case uses tf.raw_ops.SparseMatrixAdd, a GPU-only op with no
    kernel in the CPU-only TensorFlow build. csr_axpby therefore probes the op
    once and, on CPU, falls back to an equivalent COO add. This test pins the
    numerics of both branches against a dense NumPy reference and, where the
    native op is available (GPU runners), asserts the two paths agree. So Tier-1
    CPU CI covers the COO fallback + the single-matrix scaling branch, while the
    GPU nightly tier additionally covers the native path and the cross-check.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import pytest
from tensorflow.python.ops.linalg.sparse.sparse_csr_matrix_ops import CSRSparseMatrix

import gpuSolve.matrices.globalMatrices as gm
from gpuSolve.matrices.globalMatrices import csr_axpby

N           = 8
ALPHA, BETA = 1.0, 0.37


@pytest.fixture(autouse=True)
def _restore_add_cache():
    """Restore the module-level SparseMatrixAdd probe cache after each test, so
    forcing a branch here never leaks into other tests or the real dispatch."""
    prev = gm._CSR_NATIVE_ADD
    yield
    gm._CSR_NATIVE_ADD = prev


def _sym_tridiag(n: int, seed: int) -> np.ndarray:
    """A small symmetric tridiagonal dense matrix (float32)."""
    r    = np.random.default_rng(seed)
    D    = np.zeros(shape=(n, n), dtype=np.float32)
    off  = r.uniform(-1.0, 1.0, size=n - 1).astype(np.float32)
    D[np.arange(n),     np.arange(n)]     = r.uniform(1.0, 3.0, size=n)
    D[np.arange(n - 1), np.arange(1, n)]  = off
    D[np.arange(1, n),  np.arange(n - 1)] = off
    return(D)


def _to_csr(dense: np.ndarray) -> CSRSparseMatrix:
    """Wrap a dense array as a (reordered) CSRSparseMatrix."""
    return(CSRSparseMatrix(tf.sparse.reorder(tf.sparse.from_dense(tf.constant(dense)))))


def _dense(A: CSRSparseMatrix) -> np.ndarray:
    return(A.to_dense().numpy())


@pytest.mark.parametrize('force_native', [False, True], ids=['coo_fallback', 'native'])
def test_csr_axpby_matches_dense_reference(force_native):
    """alpha*A + beta*B equals the dense reference on whichever branch is used.
    The native branch is skipped where SparseMatrixAdd has no kernel (CPU-only
    build)."""
    if force_native and not gm._csr_native_add_available():
        pytest.skip('tf.raw_ops.SparseMatrixAdd has no kernel on this build (CPU-only)')
    Md           = _sym_tridiag(N, 1)
    Kd           = _sym_tridiag(N, 2)
    Kd[0, N - 1] = Kd[N - 1, 0] = 0.5        # entry in K but not M: index-union case
    gm._CSR_NATIVE_ADD = force_native
    A = csr_axpby(_to_csr(Md), ALPHA, _to_csr(Kd), BETA)
    np.testing.assert_allclose(_dense(A), ALPHA * Md + BETA * Kd, rtol=0, atol=1e-6)


def test_csr_axpby_native_and_coo_agree():
    """Where both branches exist (GPU), native SparseMatrixAdd and the COO
    fallback produce the same matrix."""
    if not gm._csr_native_add_available():
        pytest.skip('native SparseMatrixAdd unavailable (CPU-only build): nothing to compare')
    Md = _sym_tridiag(N, 3)
    Kd = _sym_tridiag(N, 4)
    gm._CSR_NATIVE_ADD = True
    native = _dense(csr_axpby(_to_csr(Md), ALPHA, _to_csr(Kd), BETA))
    gm._CSR_NATIVE_ADD = False
    coo    = _dense(csr_axpby(_to_csr(Md), ALPHA, _to_csr(Kd), BETA))
    np.testing.assert_allclose(native, coo, rtol=0, atol=1e-6)


def test_csr_axpby_single_matrix_scaling():
    """B is None -> alpha*A via the COO scaling branch (no SparseMatrixAdd)."""
    scale = 2.5
    Md    = _sym_tridiag(N, 5)
    A     = csr_axpby(_to_csr(Md), scale)
    np.testing.assert_allclose(_dense(A), scale * Md, rtol=0, atol=1e-6)

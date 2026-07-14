#!/usr/bin/env python
"""
    Tier-1 unit tests for the finite-element matrix assembly
    (gpuSolve.matrices.assemble_matrices_dict) on 1D edge meshes, where the
    assembled global matrices have simple, known closed forms.

    For a uniform 1D mesh of `nelem` linear (edge) elements over a segment of
    length L (npt = nelem+1 nodes, h = L/nelem):

    Mass (consistent) -- each edge contributes  h/6 * [[2, 1], [1, 2]],
        so M is tridiagonal with
            M[i, i]   = 2h/3 (interior),  M[0,0] = M[-1,-1] = h/3
            M[i, i+1] = M[i+1, i] = h/6
        invariants: symmetric, 3*npt-2 non-zeros, sum(M) = L (total measure).

    Stiffness (isotropic unit conductivity sigma) -- each edge contributes
        sigma/h * [[1, -1], [-1, 1]], so K is tridiagonal with
            K[i, i]   = 2 sigma/h (interior),  K[0,0] = K[-1,-1] = sigma/h
            K[i, i+1] = K[i+1, i] = -sigma/h
        invariants: symmetric, 3*npt-2 non-zeros, zero row sums (constants lie
        in the nullspace: K.1 = 0), positive semi-definite.

    Both matrices are assembled through the production assembly path (the one
    the FEM demos use) and checked entry-by-entry against the analytic form plus
    these invariants. CPU-only and fast; the same 1D mesh builder can later seed
    a 1D mMS solver test.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle
import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern, assemble_matrices_dict
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness

SIGMA = 1.0                                      # isotropic conductivity for the stiffness tests


def _dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """Empty mass-property callback (mass is material-independent)."""
    return(None)


def _sigma_iso(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties) -> np.ndarray:
    """Isotropic diffusion tensor SIGMA*I for the stiffness matrix (no fibres:
    the 1D edge mesh carries none, and an edge only sees the along-edge part)."""
    return(SIGMA * np.eye(3))


def _write_1d_mesh(fname: str, nelem: int, length: float):
    """Write a uniform 1D edge mesh of `nelem` elements over [0, length] as a
    .pkl readable by Triangulation.readMesh.
    """
    npt         = nelem + 1
    Pts         = np.zeros(shape=(npt, 3), dtype=np.float64)
    Pts[:, 0]   = np.linspace(0.0, length, npt)
    Edges       = np.zeros(shape=(nelem, 3), dtype=np.int32)
    Edges[:, 0] = np.arange(0, nelem)            # first node of each edge
    Edges[:, 1] = np.arange(1, nelem + 1)        # second node of each edge
    Edges[:, 2] = 1                              # region id (last column)
    mesh = {'Pts': Pts, 'Elems': {'Edges': Edges}, 'Fibres': None}
    with open(fname, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)


def _analytic_1d_mass(nelem: int, length: float) -> np.ndarray:
    """Analytic consistent mass matrix of a uniform 1D linear-element mesh."""
    npt = nelem + 1
    h   = length / nelem
    M   = np.zeros(shape=(npt, npt), dtype=np.float64)
    for i in range(nelem):
        # each edge adds  h/6 * [[2, 1], [1, 2]]  to its two nodes
        M[i, i]         += 2.0 * h / 6.0
        M[i + 1, i + 1] += 2.0 * h / 6.0
        M[i, i + 1]     += 1.0 * h / 6.0
        M[i + 1, i]     += 1.0 * h / 6.0
    return(M)


def _analytic_1d_stiffness(nelem: int, length: float, sigma: float = SIGMA) -> np.ndarray:
    """Analytic stiffness matrix of a uniform 1D linear-element mesh with
    isotropic conductivity sigma."""
    npt = nelem + 1
    h   = length / nelem
    K   = np.zeros(shape=(npt, npt), dtype=np.float64)
    for i in range(nelem):
        # each edge adds  sigma/h * [[1, -1], [-1, 1]]  to its two nodes
        K[i, i]         += sigma / h
        K[i + 1, i + 1] += sigma / h
        K[i, i + 1]     += -sigma / h
        K[i + 1, i]     += -sigma / h
    return(K)


def _assemble_dense(mesh_file: str, name: str, local_fn, ud_callback) -> np.ndarray:
    """Assemble the single matrix `name` for the mesh in mesh_file through the
    production path and return it dense."""
    domain = Triangulation()
    domain.readMesh(mesh_file)
    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    materials    = MaterialProperties()
    materials.add_ud_function(name, ud_callback)
    matrices     = assemble_matrices_dict({name: local_fn}, pattern, domain, materials, connectivity)
    st           = tf.sparse.reorder(matrices[name].to_sparse_tensor())
    return(tf.sparse.to_dense(st).numpy())


@pytest.mark.parametrize('nelem,length', [(4, 1.0), (7, 3.5)])
def test_mass_matrix_1d_closed_form(tmp_path, nelem, length):
    """The assembled 1D mass matrix must match the analytic tridiagonal form."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file, nelem, length)

    M    = _assemble_dense(mesh_file, 'mass', localMass, _dfmass)
    Mref = _analytic_1d_mass(nelem, length)
    npt  = nelem + 1

    # entry-by-entry match with the closed form
    np.testing.assert_allclose(M, Mref, rtol=1.0e-5, atol=1.0e-6)
    # symmetry
    np.testing.assert_allclose(M, M.T, rtol=1.0e-6, atol=1.0e-7)
    # tridiagonal sparsity: 3*npt - 2 non-zeros
    nnz = int(np.count_nonzero(np.abs(M) > 1.0e-12))
    assert nnz == 3 * npt - 2, 'expected tridiagonal nnz={0}, got {1}'.format(3 * npt - 2, nnz)
    # total mass equals the measure (length) of the domain
    assert abs(float(M.sum()) - length) < 1.0e-5, 'total mass {0} != length {1}'.format(M.sum(), length)


@pytest.mark.parametrize('nelem,length', [(4, 1.0), (7, 3.5)])
def test_stiffness_matrix_1d_closed_form(tmp_path, nelem, length):
    """The assembled 1D stiffness matrix must match the analytic tridiagonal
    form and satisfy the stiffness invariants (zero row sums, PSD)."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file, nelem, length)

    K    = _assemble_dense(mesh_file, 'stiffness', localStiffness, _sigma_iso)
    Kref = _analytic_1d_stiffness(nelem, length)
    npt  = nelem + 1

    # entry-by-entry match with the closed form
    np.testing.assert_allclose(K, Kref, rtol=1.0e-5, atol=1.0e-6)
    # symmetry
    np.testing.assert_allclose(K, K.T, rtol=1.0e-6, atol=1.0e-7)
    # tridiagonal sparsity: 3*npt - 2 non-zeros
    nnz = int(np.count_nonzero(np.abs(K) > 1.0e-12))
    assert nnz == 3 * npt - 2, 'expected tridiagonal nnz={0}, got {1}'.format(3 * npt - 2, nnz)
    # constants lie in the nullspace: every row sums to zero (K . 1 = 0)
    np.testing.assert_allclose(K.sum(axis=1), np.zeros(npt), rtol=0.0, atol=1.0e-5)
    # positive semi-definite: eigenvalues >= 0 (one is ~0, the constant mode)
    evals = np.linalg.eigvalsh(K)
    assert evals.min() > -1.0e-6, 'stiffness not PSD: min eigenvalue {0}'.format(evals.min())
    assert abs(evals[0]) < 1.0e-6, 'expected a zero (constant) mode, got {0}'.format(evals[0])

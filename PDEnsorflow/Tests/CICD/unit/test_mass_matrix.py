#!/usr/bin/env python
"""
    Tier-1 unit test for the finite-element mass matrix assembly
    (gpuSolve.matrices.assemble_matrices_dict with localMass, on 1D edge meshes).

    The consistent mass matrix of a uniform 1D mesh of linear (edge) elements has
    a known closed form. For a mesh of `nelem` elements over a segment of length L
    (npt = nelem+1 nodes, h = L/nelem), each edge contributes the local mass
    h/6 * [[2, 1], [1, 2]], so the assembled matrix M is tridiagonal with

        M[i, i]     = 2h/3   (interior)      M[0, 0] = M[-1, -1] = h/3
        M[i, i+1]   = M[i+1, i] = h/6

    The test builds such a mesh, assembles M through the production assembly path
    (the one the FEM demos use), and checks it entry-by-entry against the analytic
    matrix, together with the invariants a mass matrix must satisfy: symmetry,
    tridiagonal sparsity (3*npt-2 non-zeros), and total mass sum(M) = L (the
    measure of the domain).

    CPU-only and fast; the same 1D mesh builder can later seed a 1D mMS test.

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


def _dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """Empty mass-property callback (mass is material-independent)."""
    return(None)


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


def _assemble_mass_dense(mesh_file: str) -> np.ndarray:
    """Assemble the mass matrix for the mesh in mesh_file; return it dense."""
    domain = Triangulation()
    domain.readMesh(mesh_file)
    connectivity = domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)
    materials    = MaterialProperties()
    materials.add_ud_function('mass', _dfmass)
    matrices     = assemble_matrices_dict({'mass': localMass}, pattern, domain, materials, connectivity)
    M_st         = tf.sparse.reorder(matrices['mass'].to_sparse_tensor())
    return(tf.sparse.to_dense(M_st).numpy())


@pytest.mark.parametrize('nelem,length', [(4, 1.0), (7, 3.5)])
def test_mass_matrix_1d_closed_form(tmp_path, nelem, length):
    """The assembled 1D mass matrix must match the analytic tridiagonal form."""
    mesh_file = str(tmp_path / 'line.pkl')
    _write_1d_mesh(mesh_file, nelem, length)

    M    = _assemble_mass_dense(mesh_file)
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

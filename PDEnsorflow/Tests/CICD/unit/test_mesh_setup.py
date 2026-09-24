#!/usr/bin/env python
"""
    Tier-1 unit tests for the mesh set-up that precedes the assembly:
    the node connectivity (Triangulation.mesh_connectivity), the region ID of
    each node (Triangulation.point_region_ids) and the sparsity pattern
    (gpuSolve.matrices.compute_coo_pattern).

    All three used to be Python loops over every element or node, and are now
    whole-array operations. The loops are kept here, verbatim in what they
    compute, as the reference: the new code must give the SAME result, entry by
    entry and in the same dtype, not merely an equivalent one. The mesh is the
    coarse demo square (63001 nodes, four regions), whose region boundaries
    give nodes shared by elements of different regions, i.e. the ties the
    region rule has to break.

    CPU-only, no solve.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import pickle
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.matrices.globalMatrices import compute_coo_pattern


@pytest.fixture(scope='module')
def square(data_dir) -> Triangulation:
    domain = Triangulation()
    domain.readMesh(os.path.join(data_dir, 'triangulated_square.pkl'))
    return(domain)


def _reference_connectivity(domain: Triangulation) -> dict:
    """The per-node list loop the whole-array version replaced."""
    npt = domain.Pts().shape[0]
    connectivity = {jpt: [jpt] for jpt in range(npt)}
    for _name, Elements in domain.Elems().items():
        for Elem in Elements:
            nnodes = Elem.shape[-1] - 1
            for ilpt in range(nnodes):
                for jlpt in range(1 + ilpt, nnodes):
                    connectivity[Elem[ilpt]].append(Elem[jlpt])
                    connectivity[Elem[jlpt]].append(Elem[ilpt])
    return({key: np.unique(np.array(value, dtype=np.int32)) for key, value in connectivity.items()})


def _reference_point_region_ids(domain: Triangulation) -> np.ndarray:
    """The per-node majority loop: argmax(bincount) keeps the smallest region
    among equally frequent ones."""
    npt = domain.Pts().shape[0]
    regions = {ipt: [] for ipt in range(npt)}
    for _name, Elements in domain.Elems().items():
        for Elem in Elements:
            for ID in Elem[:-1]:
                regions[ID].append(Elem[-1])
    out = np.zeros(npt) - 1
    for ipt in range(npt):
        out[ipt] = np.argmax(np.bincount(regions[ipt]))
    return(out)


def _reference_pattern(connectivity: dict) -> dict:
    """The per-entry loop of compute_coo_pattern."""
    npt = len(connectivity)
    nzero = sum(value.shape[0] for value in connectivity.values())
    I = np.zeros(nzero, dtype=int)
    J = np.zeros(nzero, dtype=int)
    start = np.zeros(npt + 1, dtype=int)
    k = -1
    for jpt in range(npt):
        loc = connectivity[jpt]
        start[jpt + 1] = start[jpt] + loc.shape[0]
        for jloc in range(loc.shape[0]):
            k += 1
            I[k] = jpt
            J[k] = loc[jloc]
    return({'I': I.astype(np.int32), 'J': J.astype(np.int32), 'StartIndex': start.astype(np.int32)})


def test_connectivity_matches_the_loop(square):
    """Every node's neighbours (itself included), sorted, int32."""
    new = square.mesh_connectivity()
    ref = _reference_connectivity(square)
    assert list(new.keys()) == list(ref.keys())
    for key in ref:
        assert new[key].dtype == np.int32
        assert np.array_equal(new[key], ref[key]), key


def test_point_region_ids_match_the_loop(square):
    """Same winner per node, ties to the smallest region, same dtype."""
    new = square.point_region_ids()
    ref = _reference_point_region_ids(square)
    assert new.dtype == ref.dtype
    assert np.array_equal(new, ref)
    # the square has region boundaries, so the tie rule was exercised
    assert np.unique(ref).size > 1


def test_point_region_ids_break_ties_to_the_smallest_region(tmp_path):
    """Node 1 sits on one element of region 5 and one of region 2: a tie."""
    mesh = {'Pts': np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]),
            'Elems': {'Edges': np.array([[0, 1, 5], [1, 2, 2]], dtype=np.int32)},
            'Fibres': None}
    fname = str(tmp_path / 'tie.pkl')
    with open(fname, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)
    domain = Triangulation()
    domain.readMesh(fname)
    assert np.array_equal(domain.point_region_ids(), np.array([5.0, 2.0, 2.0]))


def test_pattern_matches_the_loop(square):
    new = compute_coo_pattern(square.mesh_connectivity())
    ref = _reference_pattern(_reference_connectivity(square))
    for key in ('I', 'J', 'StartIndex'):
        assert new[key].dtype == ref[key].dtype, key
        assert np.array_equal(new[key], ref[key]), key

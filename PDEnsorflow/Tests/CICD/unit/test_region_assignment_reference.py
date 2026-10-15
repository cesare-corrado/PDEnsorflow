#!/usr/bin/env python
"""
    Tier-1 equivalence test of the region assignment against the reference
    simulator, for parameter-file runs.

    Giving a node its region is a purely combinatorial function of the element
    tags and of the ID lists of the regions: no floating point, no mesh
    partitioning, no iteration order left free. So it can be required to agree
    with the reference EXACTLY, element for element and node for node, and that
    is what this module does. The reference is transcribed here from
    `region_mask(..., mask_elem=false)` and `check_*` in physics/ionics.cc of
    openCARP, in the same order of operations:

      * one region defined means the whole array stays at the default index 0
        (the function returns before it builds anything);
      * the tag-to-region map is built over the regions 1..n ONLY, because
        "region 0 is the implicit default"; a tag two regions claim ends up with
        the one of higher index, since the map is overwritten as the regions are
        walked in order;
      * a node takes the HIGHEST TAG it meets among the claimed ones, and the
        region of that tag. The tie is broken on the tags, not on the region
        indices;
      * a node that no claimed tag reaches keeps index 0, i.e. imp_region[0].

    Why this module exists: gpuSolve carries the assignment as an element tag
    plus the no-region sentinel, while the reference carries a resolved region
    index. The two hold the same information, but the last step, tag to region,
    used to be left to each consumer. One consumer did it and one did not, which
    silently dropped `imp_region[0].im_sv_init` and started 92 per cent of a
    mesh from the cell model's constructor defaults. The test therefore checks
    the COMPOSITION that every consumer uses, point_region_ids() mapped through
    tag_to_entry(), rather than either half on its own.

    CPU-only, no solve: the assignment is resolved while a run is built.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle

import numpy as np
import pytest

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.parametermapper import NODAL_DEFAULT_TAG


# ---- the reference, transcribed ---------------------------------------------
def _reference_assignment(edges: list, npt: int, regions: list) -> np.ndarray:
    """The region index of every node, as region_mask(mask_elem=false) computes
    it (physics/ionics.cc). `edges` is [(node, node, tag)] and `regions` is the
    ID list of every imp_region, index by index.
    """
    if len(regions) == 1:
        # ionics.cc:395, "if(regspec.size() == 1) return;". The function returns
        # BEFORE it assigns anything, so the caller's vector stays empty, its
        # data() is NULL, and setup_MIIF skips the copy ("if (mask)") and keeps
        # the zeros its own calloc gave the mask. Every node is region 0, and
        # the tag checks at 407-408 never run either, so nothing is reported.
        return(np.zeros(npt, dtype=np.int64))
    tag_to_reg : dict = {}
    for reg in range(1, len(regions)):                      # ionics.cc:415
        for tag in regions[reg]:
            tag_to_reg[tag] = reg                           # ionics.cc:426
    mx_tag = np.zeros(npt, dtype=np.int64) - 1              # ionics.cc:430
    for edge in edges:
        tag = edge[-1]
        if tag not in tag_to_reg:                           # ionics.cc:436
            continue
        for node in edge[:-1]:
            if tag > mx_tag[node]:                          # ionics.cc:446
                mx_tag[node] = tag
    assignment = np.zeros(npt, dtype=np.int64)              # ionics.cc:410
    for node in range(npt):
        if mx_tag[node] >= 0:                               # ionics.cc:461
            assignment[node] = tag_to_reg[mx_tag[node]]
    return(assignment)


def _reference_element_assignment(edges: list, regions: list) -> np.ndarray:
    """The region index of every element, as region_mask(mask_elem=true) does
    it: the element's own tag looked up in the same map, and index 0 for a tag
    no entry claims (ionics.cc:410 and 439). This is the path gregion takes
    (electrics.cc:51 and 56).
    """
    if len(regions) == 1:
        return(np.zeros(len(edges), dtype=np.int64))
    tag_to_reg : dict = {}
    for reg in range(1, len(regions)):
        for tag in regions[reg]:
            tag_to_reg[tag] = reg
    return(np.array([tag_to_reg.get(edge[-1], 0) for edge in edges], dtype=np.int64))


# ---- the same thing, through gpuSolve ---------------------------------------
def _mesh(folder: str, edges: list, npt: int) -> Triangulation:
    """A 1D mesh of tagged edges, written in the format the reader takes."""
    mesh = {'Pts': np.array([[float(ipt), 0.0, 0.0] for ipt in range(npt)]),
            'Elems': {'Edges': np.array(edges, dtype=np.int32)},
            'Fibres': None}
    fname = os.path.join(folder, 'regions.pkl')
    with open(fname, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)
    domain = Triangulation()
    domain.readMesh(fname)
    return(domain)


def _mapper(regions: list) -> ParameterMapper:
    """A mapper for a file that defines one imp_region per entry of `regions`,
    with that entry's ID list. A region with an empty list writes no ID at all,
    which is the file that used to give it the whole mesh.
    """
    argv = ['-num_imp_regions', str(len(regions))]
    for index, tags in enumerate(regions):
        argv += ['-imp_region[{}].im'.format(index), 'MitchellSchaeffer']
        if len(tags) > 0:
            argv += ['-imp_region[{}].ID'.format(index),
                     ','.join(str(tag) for tag in tags)]
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


def _gpusolve_assignment(folder: str, edges: list, npt: int, regions: list) -> np.ndarray:
    """The region index of every node as a parameter-file run resolves it: the
    nodal rule of the mesh, then the tag-to-entry map of the mapper. This is the
    composition every consumer of point_region_ids() performs.
    """
    mapper = _mapper(regions)
    domain = _mesh(folder, edges, npt)
    domain.set_regionIDrule('carp_compatibility')
    domain.set_regionIDtags(mapper.nodal_region_tags())
    nodal_ids  = np.reshape(np.asarray(domain.point_region_ids()), (-1,)).astype(np.int64)
    mesh_tags  = set(int(tag) for tag in np.unique(np.asarray(edges)[:, -1]))
    assignment = mapper.tag_to_entry('imp_region', mesh_tags | {NODAL_DEFAULT_TAG})
    return(np.array([assignment[int(tag)] for tag in nodal_ids], dtype=np.int64))


# ---- the scenarios ----------------------------------------------------------
# (name, edges, number of nodes, the ID list of each imp_region). Every one of
# them is a case the two implementations could disagree on, and the last four
# are cases where they did or nearly did.
_SCENARIOS = [
    ('one region claims nothing',
     [[0, 1, 1], [1, 2, 2]], 3, [[]]),
    ('one region claims some tags',
     [[0, 1, 1], [1, 2, 2]], 3, [[1]]),
    ('two regions, disjoint tags',
     [[0, 1, 1], [1, 2, 2]], 3, [[1], [2]]),
    ('a node between two claimed tags takes the higher',
     [[0, 1, 1], [1, 2, 2]], 3, [[], [1, 2]]),
    ('a region from 1 onwards that lists no ID claims nothing',
     [[0, 1, 1], [1, 2, 2]], 3, [[], []]),
    ('an unclaimed tag does not compete',
     [[0, 1, 1], [1, 2, 7], [2, 3, 2]], 4, [[], [2]]),
    ('a tag claimed twice goes to the higher index',
     [[0, 1, 1], [1, 2, 2], [2, 3, 3]], 4, [[], [2, 3], [3]]),
    ('the default region lists IDs of its own',
     [[0, 1, 0], [1, 2, 1], [2, 3, 3]], 4, [[0, 3], [1]]),
    ('a lower claimed tag beats a higher one the default region lists',
     [[0, 1, 3], [1, 2, 1]], 3, [[0, 3], [1]]),
    ('a region claims a tag the mesh does not carry',
     [[0, 1, 1], [1, 2, 2]], 3, [[], [99]]),
    # node 1 meets tags 2 and 3, claimed by regions 2 and 1 respectively. The
    # winner is the region of the HIGHEST TAG, 3, so region 1, and not the
    # highest region index, 2. The reference breaks the tie on the tags
    # themselves and says why (ionics.cc:450-453): reducing region indices
    # instead would let the highest index win at the nodes shared between MPI
    # ranks and the highest tag win everywhere else, so the assignment would
    # depend on the partitioning whenever the tags are not listed in ascending
    # order. This is the one scenario that separates the two rules.
    ('the tie is on the tags, not on the region indices',
     [[0, 1, 2], [1, 2, 3]], 3, [[], [3], [2]]),
]


@pytest.mark.parametrize('name,edges,npt,regions',
                         _SCENARIOS, ids=[case[0] for case in _SCENARIOS])
def test_the_assignment_is_the_reference_assignment(tmp_path, name, edges, npt, regions):
    """Node for node, the resolved region index is the reference's."""
    expected = _reference_assignment(edges, npt, regions)
    got      = _gpusolve_assignment(str(tmp_path), edges, npt, regions)
    assert np.array_equal(got, expected), '{}: {} != {}'.format(name, got, expected)


@pytest.mark.parametrize('name,edges,npt,regions',
                         _SCENARIOS, ids=[case[0] for case in _SCENARIOS])
def test_the_element_assignment_is_the_reference_assignment(name, edges, npt, regions):
    """The element path of the same map, which is what the conductivities take
    (region_mask with mask_elem=true, called for gregion at electrics.cc:51).
    An element is governed by its own tag, with no maximum to take, so the only
    thing that can go wrong is the map itself, and it is the same map.
    """
    mapper   = _mapper(regions)
    tags     = set(int(tag) for tag in np.unique(np.asarray(edges)[:, -1]))
    entry    = mapper.tag_to_entry('imp_region', tags)
    got      = np.array([entry[int(edge[-1])] for edge in edges], dtype=np.int64)
    expected = _reference_element_assignment(edges, regions)
    assert np.array_equal(got, expected), '{}: {} != {}'.format(name, got, expected)


def test_the_mapper_resolves_the_nodal_map_itself():
    """The resolution must live in the mapper, not in this test's helper: the
    entries of state_init_files() carry the ID of the nodes no region claims,
    so a consumer only has to select on them. If the sentinel stopped being
    resolved there, a file on the default region would reach no node again.
    The file itself is never opened here; the mapper only resolves which nodes
    it would reach.
    """
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-num_imp_regions', '2',
                                        '-imp_region[0].im', 'MitchellSchaeffer',
                                        '-imp_region[0].ID', '1',
                                        '-imp_region[0].im_sv_init', 'A.sv',
                                        '-imp_region[1].im', 'MitchellSchaeffer',
                                        '-imp_region[1].ID', '2']))
    entries = mapper.state_init_files({1, 2})
    assert len(entries) == 1
    assert entries[0]['index'] == 0
    assert NODAL_DEFAULT_TAG in entries[0]['tags']


def test_a_counter_smaller_than_the_assigned_ids_does_not_drop_a_tag(tmp_path):
    """num_IDs = 1 with ID[1] also assigned claims BOTH tags. The reference's
    parser resolves a counter to the larger of the declared value and the count
    inferred from the assigned slots, so honouring the small counter would run a
    different cell model on every node of the dropped tag.
    """
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-num_imp_regions', '2',
                                        '-imp_region[1].num_IDs', '1',
                                        '-imp_region[1].ID[0]', '2',
                                        '-imp_region[1].ID[1]', '3']))
    assert mapper.region_tags('imp_region', 1) == [2, 3]
    assert mapper.nodal_region_tags() == {2, 3}
    edges   = [[0, 1, 1], [1, 2, 2], [2, 3, 3]]
    regions = [[], [2, 3]]
    assert np.array_equal(_gpusolve_assignment(str(tmp_path), edges, 4, regions),
                          _reference_assignment(edges, 4, regions))


def test_a_lower_claimed_tag_can_beat_a_higher_unclaimed_one(tmp_path):
    """The case that makes the rule worth transcribing rather than guessing: a
    node meeting tags 1 and 3, where 3 is listed on imp_region[0] and 1 on
    imp_region[1], belongs to region 1. A rule that let every listed tag
    compete would give it region 0 and run the wrong cell model there. It occurs
    wherever a right-ventricular element meets a border-zone element.
    """
    edges   = [[0, 1, 3], [1, 2, 1]]
    regions = [[0, 3], [1]]
    assert np.array_equal(_reference_assignment(edges, 3, regions),
                          np.array([0, 1, 1]))
    assert np.array_equal(_gpusolve_assignment(str(tmp_path), edges, 3, regions),
                          np.array([0, 1, 1]))


def test_an_unfilled_id_slot_cannot_claim_the_default_region(tmp_path):
    """imp_region[].ID[] defaults to -1, which is also the ID of a node no
    region reaches. A counter that promises more IDs than the file gives must
    therefore not leave -1 in the tag list: the region would otherwise claim
    every node of the default region. The reference warns about the missing ID
    and the value is inert there; here it is dropped, and reported.
    """
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-num_imp_regions', '2',
                                        '-imp_region[1].num_IDs', '2',
                                        '-imp_region[1].ID[0]', '2']))
    assert mapper.region_tags('imp_region', 1) == [2]
    assert any('not all of them are given' in note for note in mapper.notes())
    assert mapper.nodal_region_tags() == {2}
    # and the nodes of the default region are still the default region's
    edges = [[0, 1, 1], [1, 2, 2]]
    domain = _mesh(str(tmp_path), edges, 3)
    domain.set_regionIDrule('carp_compatibility')
    domain.set_regionIDtags(mapper.nodal_region_tags())
    nodal_ids  = np.reshape(np.asarray(domain.point_region_ids()), (-1,)).astype(np.int64)
    assignment = mapper.tag_to_entry('imp_region', {1, 2, NODAL_DEFAULT_TAG})
    assert [assignment[int(tag)] for tag in nodal_ids] == [0, 1, 1]


def test_the_default_rule_is_untouched(tmp_path):
    """The equivalence is asked of the parameter-file rule only. A script that
    never set a rule must still get 'mostfrequent', where no node carries the
    sentinel, or every existing caller's numbers would move.
    """
    domain = _mesh(str(tmp_path), [[0, 1, 1], [1, 2, 2]], 3)
    assert domain.regionIDrule() == 'mostfrequent'
    assert domain.regionIDtags() is None
    assert np.array_equal(domain.point_region_ids(), np.array([1.0, 1.0, 2.0]))

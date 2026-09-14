#!/usr/bin/env python
"""
    Tier-1 unit tests for the external mesh format round trip
    (gpuSolve.IO.writers.CarpMeshWriter -> gpuSolve.IO.readers.CarpMeshReader,
    through Triangulation.exportCarpFormat / readMesh).

    The interesting case is the line element. Its type specifier in the `.elem`
    format is `Ln`, but this package wrote `Cx`, so a mesh exported with line
    elements could not be read by anything else that consumes that format.
    The writer now emits `Ln`; the reader accepts both, so meshes written by
    earlier versions still load.

    CPU-only, a handful of elements, no solve.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pickle

import numpy as np

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.IO.readers.carpmeshreader import elemDeCode
from gpuSolve.IO.writers.carpmeshwriter import elemCode


def _write_pkl(path: str) -> dict:
    """A four-node strip carrying both a line and two triangles."""
    mesh = {'Pts': np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                             [100.0, 100.0, 0.0], [0.0, 100.0, 0.0]]),
            'Elems': {'Edges': np.array([[0, 1, 1]], dtype=np.int32),
                      'Trias': np.array([[0, 1, 2, 1], [0, 2, 3, 2]], dtype=np.int32)},
            'Fibres': np.array([[1.0, 0.0, 0.0]] * 3)}
    with open(path, 'wb') as fout:
        pickle.dump(mesh, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return(mesh)


def test_line_elements_are_written_as_Ln(tmp_path):
    """The exported .elem uses the specifier the format defines, not `Cx`."""
    source = os.path.join(str(tmp_path), 'strip.pkl')
    _write_pkl(source)
    mesh = Triangulation()
    mesh.readMesh(source)
    prefix = os.path.join(str(tmp_path), 'strip')
    mesh.exportCarpFormat(prefix)
    with open('{}.elem'.format(prefix), 'r') as felem:
        body = felem.read()
    assert 'Ln ' in body
    assert 'Cx ' not in body
    assert 'Tr ' in body


def test_the_round_trip_preserves_the_mesh(tmp_path):
    """Written and read back, the mesh is the one that went in."""
    source = os.path.join(str(tmp_path), 'strip.pkl')
    original = _write_pkl(source)
    mesh = Triangulation()
    mesh.readMesh(source)
    prefix = os.path.join(str(tmp_path), 'strip')
    mesh.exportCarpFormat(prefix)

    back = Triangulation()
    back.readMesh(prefix)                       # no suffix: the three-file format
    np.testing.assert_allclose(back.Pts(), original['Pts'])
    np.testing.assert_array_equal(back.Elems()['Edges'], original['Elems']['Edges'])
    np.testing.assert_array_equal(back.Elems()['Trias'], original['Elems']['Trias'])
    np.testing.assert_allclose(back.Fibres(), original['Fibres'])


def test_the_legacy_spelling_still_reads(tmp_path):
    """A mesh written by an earlier version uses `Cx`; it must still load, or
    existing files on disk would stop working."""
    prefix = os.path.join(str(tmp_path), 'legacy')
    with open('{}.pts'.format(prefix), 'w') as fout:
        fout.write('2\n0.0 0.0 0.0\n100.0 0.0 0.0\n')
    with open('{}.elem'.format(prefix), 'w') as fout:
        fout.write('1\nCx 0 1 1\n')
    with open('{}.lon'.format(prefix), 'w') as fout:
        fout.write('1\n1.0 0.0 0.0\n')
    mesh = Triangulation()
    mesh.readMesh(prefix)
    np.testing.assert_array_equal(mesh.Elems()['Edges'], np.array([[0, 1, 1]]))


def test_both_specifiers_name_the_same_element():
    """The two spellings decode to one internal element type, and the writer
    now produces the one the format defines."""
    assert elemDeCode('Ln') == elemDeCode('Cx') == 'Edges'
    assert elemCode('Edges') == 'Ln'

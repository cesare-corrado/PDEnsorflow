#!/usr/bin/env python
"""
    Tier-1 unit tests for the `.vtx` vertex-file reader
    (gpuSolve.IO.readers.VtxReader).

    A `.vtx` file names a set of nodes explicitly: a count, an optional
    `intra` / `extra` keyword, then one 0-based node index per line. The two
    tolerances that make such a file portable are covered here: the keyword line
    is not counted and not required, and reading stops once the declared number
    of indices has been collected.

    Pure text handling, no mesh and no TensorFlow.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os

import numpy as np
import pytest

from gpuSolve.IO.readers import VtxReader


def _write(tmp_path, name: str, text: str) -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, 'w') as fout:
        fout.write(text)
    return(path)


def test_reads_a_file_with_the_domain_keyword(tmp_path):
    """The keyword line is recorded but does not count towards n."""
    reader = VtxReader()
    indices = reader.read(_write(tmp_path, 'e.vtx', '3\nextra\n7\n11\n42\n'))
    np.testing.assert_array_equal(indices, np.array([7, 11, 42]))
    assert reader.domain() == 'extra'
    assert reader.indices() is indices


def test_reads_a_file_without_the_domain_keyword(tmp_path):
    """The keyword is optional, so a bare count and list is valid."""
    reader = VtxReader()
    indices = reader.read(_write(tmp_path, 'e.vtx', '2\n0\n5\n'))
    np.testing.assert_array_equal(indices, np.array([0, 5]))
    assert reader.domain() is None


def test_comments_and_blank_lines_are_skipped(tmp_path):
    """A file may be annotated without changing what it names."""
    indices = VtxReader().read(_write(tmp_path, 'e.vtx',
                                      '# an electrode\n\n2\nintra\n\n3\n4\n'))
    np.testing.assert_array_equal(indices, np.array([3, 4]))


def test_reading_stops_at_the_declared_count(tmp_path):
    """Trailing content is ignored rather than being an error."""
    indices = VtxReader().read(_write(tmp_path, 'e.vtx', '2\n1\n2\n3\n4\n'))
    np.testing.assert_array_equal(indices, np.array([1, 2]))


def test_a_short_file_is_rejected(tmp_path):
    """Declaring more vertices than the file carries is an error: silently
    stimulating a smaller electrode than asked for would be worse."""
    with pytest.raises(ValueError) as excinfo:
        VtxReader().read(_write(tmp_path, 'e.vtx', '5\nintra\n1\n2\n'))
    assert 'declares 5' in str(excinfo.value)


def test_a_file_with_per_node_data_is_refused(tmp_path):
    """A second column is a different format whose values would be dropped."""
    with pytest.raises(ValueError) as excinfo:
        VtxReader().read(_write(tmp_path, 'e.vtx', '2\n1 0.5\n2 0.25\n'))
    assert 'more than one value per node' in str(excinfo.value)


def test_a_file_without_a_count_is_refused(tmp_path):
    """Without the count there is nothing to read."""
    with pytest.raises(ValueError) as excinfo:
        VtxReader().read(_write(tmp_path, 'e.vtx', 'intra\n'))
    assert 'no vertex count' in str(excinfo.value)

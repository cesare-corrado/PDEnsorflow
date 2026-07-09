#!/usr/bin/env python
"""
    Tier-1 unit test for the IGB write/read round-trip
    (gpuSolve.IO.writers.IGBWriter <-> gpuSolve.IO.readers.IGBReader).

    Writes a deterministic sequence of frames with IGBWriter, reads the file
    back with IGBReader and checks that the recovered array matches the
    original data exactly, together with the header-derived shape (nt, nx).

    Two parametrized cases are exercised:
      * a small mesh (a few hundred nodes, a handful of frames), and
      * a LARGE-nx mesh (nx = 63001): this is a regression guard for a fixed
        bug where IGBReader mis-parsed the header for large nx (the reader
        must read the full 1024-byte header rather than a truncated one) and
        therefore failed to round-trip large-nx data.

    CPU-only, fast and self-cleaning (the IGB file lives under pytest's
    tmp_path and is discarded automatically).

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf  # noqa: F401  (imported for parity with the sibling IO tests)
import pytest

from gpuSolve.IO.writers import IGBWriter
from gpuSolve.IO.readers import IGBReader


def _reference_data(nx: int, nt: int) -> np.ndarray:
    """Deterministic (nt, nx) reference array: the value at frame t, node i is
    a smooth function of both indices (sin(i) offset by 100*t), entirely
    reproducible (no RNG) so the round-tripped data is exactly checkable.
    """
    node  = np.arange(nx, dtype=np.float32)
    frame = np.arange(nt, dtype=np.float32)
    data  = np.sin(node)[np.newaxis, :] + 100.0 * frame[:, np.newaxis]
    return(data.astype(np.float32))


def _write_igb(fname: str, data: np.ndarray, nx: int, nt: int):
    """Writes the (nt, nx) array `data` to `fname` in IGB format, one frame
    (row) at a time through IGBWriter.imshow(), then finalises the file.
    """
    writer = IGBWriter({'fname': fname, 'Tend': float(nt - 1), 'nt': nt, 'nx': nx})
    for t in range(nt):
        writer.imshow(data[t, :])
    writer.wait()


@pytest.mark.parametrize('nx,nt', [(300, 5), (63001, 3)], ids=['small', 'large_nx'])
def test_igb_write_read_roundtrip(tmp_path, nx, nt):
    """IGBWriter -> IGBReader must reproduce the original data and header
    (nt, nx) exactly, including the large-nx case that exercises the full
    1024-byte header parsing path.
    """
    fname = str(tmp_path / 'roundtrip.igb')
    ref   = _reference_data(nx, nt)

    _write_igb(fname, ref, nx, nt)

    reader = IGBReader()
    reader.read(fname)
    result = reader.data()

    # header-derived shape/counts
    assert reader.nx() == nx, 'header nx mismatch'
    assert reader.nt() == nt, 'header nt mismatch'
    assert reader.ndiff() == 0, 'data size does not match nt*nx from the header'
    assert result.shape == (nt, nx), 'read-back array must be shaped (nt, nx)'

    # exact (float32) data match
    np.testing.assert_allclose(result, ref, rtol=1.0e-6, atol=1.0e-6)

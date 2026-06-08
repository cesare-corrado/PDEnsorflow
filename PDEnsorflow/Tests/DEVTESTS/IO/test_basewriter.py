#!/usr/bin/env python
"""
    DEVTEST for gpuSolve.IO.writers.BaseWriter

    It verifies the variable-size GPU container that buffers the solution and
    flushes it to disk either:
      * every_N      : every N received solutions, or
      * max_chunk_mb : when the buffered data exceeds the memory threshold.

    The test is fully autonomous: every temporary file/folder created during
    the run is removed in tearDown().

    Copyright 2022-2023 Cesare Corrado (cesare.corrado@kcl.ac.uk)
"""
import os
import sys
import shutil
import tempfile
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# make the test self-locating so it always exercises the local
# (edited) gpuSolve package rather than a globally installed copy
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np
import tensorflow as tf
from gpuSolve.IO.writers import BaseWriter


class TestBaseWriter(unittest.TestCase):
    """Unit tests for the BaseWriter chunked GPU container."""

    def setUp(self):
        """Create an isolated temporary directory for each test."""
        self._tmpdir = tempfile.mkdtemp(prefix='basewriter_test_')

    def tearDown(self):
        """Remove every temporary .npy/.igb file and folder created by the test."""
        if os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Test 1: iteration-based chunking (every_N)                         #
    # ------------------------------------------------------------------ #
    def test_every_N_chunking(self):
        """The writer must keep solutions in GPU memory and flush a chunk to
        disk only on the exact multiples of every_N; the final aggregated file
        must match the original solutions exactly.
        """
        n      = 10
        N      = 3
        nx     = 128
        fname  = os.path.join(self._tmpdir, 'every_n')

        writer = BaseWriter({'fname': fname, 'every_N': N})

        # n different dummy solutions (random GPU tensors)
        frames = [tf.random.uniform((nx,), dtype=tf.float32) for _ in range(n)]
        ref    = np.stack([f.numpy() for f in frames], axis=0)

        prev_chunks = 0
        for i, frame in enumerate(frames, start=1):
            writer.add_solution(frame)
            expected_chunks = i // N

            # the class dumps a chunk ONLY on exact multiples of every_N
            if i % N == 0:
                self.assertEqual(writer.nb_chunks(), prev_chunks + 1,
                                 'a chunk should be flushed at iteration {0}'.format(i))
            else:
                self.assertEqual(writer.nb_chunks(), prev_chunks,
                                 'no chunk should be flushed at iteration {0}'.format(i))
            prev_chunks = writer.nb_chunks()

            # un-flushed solutions are still held in (GPU) memory
            self.assertEqual(len(writer._buffer), i - expected_chunks * N)
            # the number of chunk files on disk matches the flush count
            self.assertEqual(len(writer._chunk_files), expected_chunks)
            for cf in writer._chunk_files:
                self.assertTrue(os.path.exists(cf))

        # remainder still buffered before the final aggregation
        self.assertEqual(len(writer._buffer), n % N)

        # final aggregation
        writer.finalize()

        # final file shape and data must match the original n solutions
        out = np.load(fname + '.npy')
        self.assertEqual(out.shape, (n, nx))
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

        # temporary chunks have been cleaned up
        self.assertEqual(writer._chunk_files, [])

    # ------------------------------------------------------------------ #
    # Test 2: memory-based chunking (max_chunk_mb)                       #
    # ------------------------------------------------------------------ #
    def test_max_chunk_mb_chunking(self):
        """With a tiny memory threshold, the writer must flush a chunk as soon
        as the accumulated tensors exceed max_chunk_mb.
        """
        max_chunk_mb = 1                       # 1 MB threshold
        nelem        = 200000                  # float32 -> 200000*4 = ~0.76 MB / frame
        fname        = os.path.join(self._tmpdir, 'max_mb')

        # every_N is None -> the memory trigger governs the flushing
        writer = BaseWriter({'fname': fname, 'max_chunk_mb': max_chunk_mb})
        self.assertIsNone(writer._every_N)

        frames = [tf.ones((nelem,), dtype=tf.float32) * float(k) for k in range(5)]
        ref    = np.stack([f.numpy() for f in frames], axis=0)

        # frame 0: ~0.76 MB < 1 MB  -> still in memory, no dump
        writer.add_solution(frames[0])
        self.assertEqual(writer.nb_chunks(), 0,
                         'no dump expected while below the memory threshold')

        # frame 1: ~1.52 MB >= 1 MB -> the accumulated buffer exceeds the
        # threshold, so a disk dump must be triggered
        writer.add_solution(frames[1])
        self.assertEqual(writer.nb_chunks(), 1,
                         'a dump must be triggered as soon as the threshold is exceeded')
        self.assertEqual(len(writer._chunk_files), 1)
        self.assertTrue(os.path.exists(writer._chunk_files[0]))
        # the GPU buffer has been freed after the flush
        self.assertEqual(len(writer._buffer), 0)

        # feed the remaining frames: another flush at frame 3 (pairs of frames)
        writer.add_solution(frames[2])
        self.assertEqual(writer.nb_chunks(), 1)
        writer.add_solution(frames[3])
        self.assertEqual(writer.nb_chunks(), 2)
        writer.add_solution(frames[4])
        self.assertEqual(writer.nb_chunks(), 2)

        # final aggregation: data integrity preserved across the memory chunks
        writer.finalize()
        out = np.load(fname + '.npy')
        self.assertEqual(out.shape, (len(frames), nelem))
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

#!/usr/bin/env python
"""
    Shared pytest configuration and fixtures for the PDEnsorflow CI test-suite.

    pytest discovers this file automatically for every test collected under
    Tests/CICD. It makes the in-tree gpuSolve package importable without a prior
    `pip install -e .` (so the suite always exercises the edited sources) and
    exposes the shared mesh/data directory as a fixture.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import sys
import pytest

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Prepend the in-tree package root (the inner PDEnsorflow folder that contains
# gpuSolve) so the local sources are always the ones under test. This file sits
# two levels below that root: PDEnsorflow/Tests/CICD/conftest.py .
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


@pytest.fixture(scope='session')
def data_dir() -> str:
    """Absolute path to the shared Tests/data directory (meshes, images)."""
    return(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data')))

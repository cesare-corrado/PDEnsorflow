#!/usr/bin/env python
"""
    Import smoke test for the gpuSolve package.

    Importing gpuSolve must succeed (this also exercises the LD_LIBRARY_PATH
    setup in gpuSolve/__init__.py) and report a well-formed version string.
    Running under pytest (a real script invocation) is robust to the
    re-exec in gpuSolve/__init__.py, unlike a bare `python -c "import gpuSolve"`.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import gpuSolve


def test_package_imports_and_reports_version():
    """gpuSolve imports and version() returns a dotted numeric string."""
    version = gpuSolve.version()
    assert isinstance(version, str)
    parts = version.split('.')
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts)

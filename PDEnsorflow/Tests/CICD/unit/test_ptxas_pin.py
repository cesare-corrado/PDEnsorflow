#!/usr/bin/env python
"""
    Tier-1 unit test for the XLA CUDA-data-dir pin in gpuSolve/__init__.py.

    Every ionic model compiles differentiate() with XLA, and XLA shells out to
    ptxas. It probes a list of candidate CUDA roots and takes the first one it
    accepts, so a CUDA toolkit installed system-wide can win over the one shipped
    with the pip CUDA wheels; releases 12.0 to 12.6.2 are rejected outright for a
    clamping miscompile and then no GPU kernel compiles at all. gpuSolve therefore
    points --xla_gpu_cuda_data_dir at the wheel, which is the first candidate
    probed. This test pins that selection logic; it needs no GPU, because the
    functions only look at the filesystem and at a mapping.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os

from gpuSolve import cuda_data_dir, pin_cuda_data_dir


def _make_wheel(root, with_libdevice: bool = True) -> str:
    """ _make_wheel(root, with_libdevice) builds a fake nvidia-cuda-nvcc-cu12 wheel
        layout under root and returns the cuda_nvcc directory
    """
    nvcc = os.path.join(root, 'nvidia', 'cuda_nvcc')
    os.makedirs(os.path.join(nvcc, 'bin'), exist_ok=True)
    ptxas = os.path.join(nvcc, 'bin', 'ptxas')
    with open(ptxas, 'w') as fout:
        fout.write('#!/bin/sh\n')
    os.chmod(ptxas, 0o755)
    if with_libdevice:
        os.makedirs(os.path.join(nvcc, 'nvvm', 'libdevice'), exist_ok=True)
    return(nvcc)


def test_cuda_data_dir_finds_the_wheel(tmp_path):
    root = tmp_path.as_posix()
    nvcc = _make_wheel(root)
    assert cuda_data_dir([root]) == nvcc


def test_cuda_data_dir_needs_both_ptxas_and_libdevice(tmp_path):
    # ptxas alone is not a CUDA root: XLA also links libdevice into the PTX, so a
    # half-installed wheel must be skipped rather than pinned.
    root = tmp_path.as_posix()
    _make_wheel(root, with_libdevice=False)
    assert cuda_data_dir([root]) == ''


def test_cuda_data_dir_returns_empty_when_nothing_is_installed(tmp_path):
    assert cuda_data_dir([tmp_path.as_posix()]) == ''


def test_pin_sets_the_flag_on_a_clean_environment(tmp_path):
    root = tmp_path.as_posix()
    nvcc = _make_wheel(root)
    env = {}
    flags = pin_cuda_data_dir(env, [root])
    assert flags == '--xla_gpu_cuda_data_dir={}'.format(nvcc)
    assert env['XLA_FLAGS'] == flags


def test_pin_keeps_the_other_flags_of_the_caller(tmp_path):
    root = tmp_path.as_posix()
    nvcc = _make_wheel(root)
    env = {'XLA_FLAGS': '--xla_dump_to=/tmp/dump'}
    flags = pin_cuda_data_dir(env, [root])
    assert flags == '--xla_dump_to=/tmp/dump --xla_gpu_cuda_data_dir={}'.format(nvcc)


def test_pin_is_inert_when_the_caller_chose_a_data_dir(tmp_path):
    # an explicit choice by the user wins: this is the escape hatch for a machine
    # whose system toolkit is the one that must be used.
    root = tmp_path.as_posix()
    _make_wheel(root)
    env = {'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/opt/cuda'}
    assert pin_cuda_data_dir(env, [root]) == '--xla_gpu_cuda_data_dir=/opt/cuda'
    assert env['XLA_FLAGS'] == '--xla_gpu_cuda_data_dir=/opt/cuda'


def test_pin_is_inert_without_a_wheel(tmp_path):
    env = {}
    assert pin_cuda_data_dir(env, [tmp_path.as_posix()]) == ''
    assert 'XLA_FLAGS' not in env

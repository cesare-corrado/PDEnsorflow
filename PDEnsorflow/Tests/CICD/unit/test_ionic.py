#!/usr/bin/env python
"""
    Tier-1 unit tests for the ionic cell models (gpuSolve.ionic). Every model
    subclasses IonicModel and exposes the same initialize/differentiate contract,
    so a single parametrised test exercises them all.

    Two families:
      * dimensionless -- transmembrane potential rescaled to [vmin, vmax] = [-80, 20]
        (MitchellSchaeffer2v, ModifiedMS2v, Fenton4v); rest is u = 0 -> V = vmin.
      * dimensional (physiological mV) with a resting potential V_init
        (CourtemancheRamirezNattel, TenTusscherPanfilov).

    Checks over all models: differentiate() returns a finite dU of U's shape, is
    deterministic from a fresh state, and the resting state is quasi-stable (it
    does not spontaneously depolarise -- |dV/dt| at rest is far below an upstroke,
    which is O(100) mV/ms). The dimensionless family additionally has its [0, 1]
    rescaling verified and an exact resting fixed point (dU = 0 at u = 0).

    CPU-only and fast; single-cell columns, no mesh.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import pytest

from gpuSolve.ionic.ms2v import MitchellSchaeffer2v
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.ionic.fenton4v import Fenton4v
from gpuSolve.ionic.courtemanche_ramirez_nattel import CourtemancheRamirezNattel
from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov

DIMENSIONLESS = [MitchellSchaeffer2v, ModifiedMS2v, Fenton4v]
PHYSIOLOGICAL = [CourtemancheRamirezNattel, TenTusscherPanfilov]
ALL_MODELS    = DIMENSIONLESS + PHYSIOLOGICAL

_DT        = 0.01                                # small step for the resting gate update
_N_NODES   = 4                                   # single-cell column (no mesh)
_REST_BOUND = 0.5                                # |dV/dt| at rest must be << an upstroke (O(100) mV/ms)


def _ids(models):
    return([m.__name__ for m in models])


def _resting_potential(model) -> float:
    """Resting transmembrane potential: V_init for the physiological models,
    vmin (u = 0) for the dimensionless ones."""
    v = model.get_parameter('V_init')            # plain float on the physiological models
    if v is None:
        v = model.get_parameter('vmin')          # tf.constant on the dimensionless models
    return(float(v))


def _at_rest(Model):
    """Build a model and a resting-potential column U; return (model, U)."""
    model = Model(dt=_DT, n_nodes=_N_NODES)
    U     = tf.Variable(_resting_potential(model) * tf.ones(shape=(_N_NODES, 1), dtype=tf.float32))
    model.initialize_state_variables(U)
    return(model, U)


@pytest.mark.parametrize('Model', ALL_MODELS, ids=_ids(ALL_MODELS))
def test_ionic_differentiate_contract(Model):
    """Every model returns a finite dU of U's shape, deterministically, and its
    resting state does not spontaneously depolarise."""
    model, U = _at_rest(Model)
    dU       = model.differentiate(U).numpy()

    assert dU.shape == tuple(U.shape), \
        '{0}: dU shape {1} != U {2}'.format(Model.__name__, dU.shape, tuple(U.shape))
    assert np.all(np.isfinite(dU)), '{0}: non-finite dU at rest'.format(Model.__name__)
    assert np.max(np.abs(dU)) < _REST_BOUND, \
        '{0}: rest not quasi-stable, max|dU|={1}'.format(Model.__name__, np.max(np.abs(dU)))

    # deterministic from a fresh instance / state
    model2, U2 = _at_rest(Model)
    np.testing.assert_array_equal(dU, model2.differentiate(U2).numpy())


@pytest.mark.parametrize('Model', DIMENSIONLESS, ids=_ids(DIMENSIONLESS))
def test_ionic_dimensionless_rescaling(Model):
    """The [-80, 20] rescaling maps vmin -> 0 and vmax -> 1, and rest (u = 0) is
    an exact fixed point of the potential."""
    model, U = _at_rest(Model)
    vmin     = float(model.get_parameter('vmin'))
    vmax     = float(model.get_parameter('vmax'))
    lo       = tf.constant(vmin * np.ones(shape=(_N_NODES, 1), dtype=np.float32))
    hi       = tf.constant(vmax * np.ones(shape=(_N_NODES, 1), dtype=np.float32))

    np.testing.assert_allclose(model.to_dimensionless(lo).numpy(), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(model.to_dimensionless(hi).numpy(), 1.0, atol=1.0e-6)
    # rest (U = vmin, u = 0) is an exact fixed point of the transmembrane potential
    np.testing.assert_allclose(model.differentiate(U).numpy(), 0.0, atol=1.0e-6)

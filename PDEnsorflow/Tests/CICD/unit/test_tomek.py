#!/usr/bin/env python
"""
    Tier-1 unit tests for the ToR-ORd cell model (gpuSolve.ionic.tomek.Tomek)
    and its parameters. The generic contract (finite, deterministic, quasi-stable
    rest, declared state variables) is covered with the other models in
    test_ionic.py and test_savestate.py; this file pins what is specific to it:

      * the PARAMETERS. celltype accepts 0, 1 and 2 only: a cell-type name is
        read as ENDO by the single-cell reference tool, so anything else must
        fail loudly. A constant that is folded into the lookup tables cannot be
        changed afterwards, so setting one is refused rather than ignored.
      * the PER-NODE COMPOSITION. An effective conductance is
        Coef x cell-type factor x base, and a heterogeneous column must give each
        node exactly the trajectory of a uniform model of its own type. This
        covers the two versions of the Ito inactivation table for EPI cells.
      * the PHYSICS HOOKS a scar or border zone relies on: GNa = 0 on a node
        removes its upstroke and leaves its neighbour's untouched.
      * the UNITS and the V = 0 singularity: Cai is held in mM, and the GHK
        terms, 0/0 at 0 mV, take their exact limit within 1e-6 mV of it.
      * the FORWARD-EULER option at high potentials, where tau is tiny: a gate
        at its steady state must stay there, as in the reference.
      * the IKr MARKOV CHAIN in the default mode: one step is exp(dt Q(V)),
        checked against an eigen-decomposition, and under a shock potential it
        stays a probability distribution (forward Euler, the reference scheme,
        jumps between the clamps there).

    The agreement with the reference implementation over full beats is a GPU
    regression (Tests/CICD/nightly/test_tomek_regression.py).

    CPU-only and small (a few nodes, at most 10 ms at dt = 0.02 ms). Tests whose
    subject does not depend on the integration scheme use forward Euler, whose
    tables build in a fraction of a second; the default schemes (Rush-Larsen
    gates, matrix-exponential IKr chain) are tested where they are the subject,
    and through Tomek's default in test_ionic.py, test_savestate.py and the
    parameter-file tests.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pytest
import tensorflow as tf

from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.ionic.tomek import Tomek

_DT        = 0.02                                 # ms
_STIM      = 60.0                                 # mV/ms, as the reference single-cell tool
_STIM_DUR  = 1.0                                  # ms
_GNA       = 11.7802                              # mS/uF, the model default


def _model(n_nodes: int, parameters: dict = None, rush_larsen: bool = False) -> tuple:
    """Build a model with the given parameters, at rest; return (model, U).
    Forward Euler by default, although the model's own default is the
    exponential schemes: building the default table (the IKr step matrix on
    200001 grid points) takes about 2 s per model, and what most tests here
    check (parameters, per-node composition, units, singularities) does not
    depend on the scheme. The tests of the default schemes ask for them."""
    model = Tomek(dt=_DT, n_nodes=n_nodes)
    model.set_use_rush_larsen(rush_larsen)
    for pname, pvalue in (parameters or {}).items():
        model.set_parameter(pname, pvalue)
    U = tf.Variable(np.full((n_nodes, 1), model.get_parameter('V_init'), dtype=np.float32))
    model.initialize_state_variables(U)
    return(model, U)


def _pace(model, U, duration: float) -> np.ndarray:
    """Stimulate for _STIM_DUR, run to duration; return the peak potential per node."""
    peak = np.full(U.shape[0], -np.inf)
    for step in range(int(round(duration / _DT))):
        if step < int(round(_STIM_DUR / _DT)):
            U.assign_add(_DT * _STIM * tf.ones_like(U))
        U.assign_add(_DT * model.differentiate(U))
        peak = np.maximum(peak, U.numpy().ravel())
    return(peak)


# ---- parameters --------------------------------------------------------------
@pytest.mark.parametrize('value', [3.0, -1.0, 0.5, np.array([[0.0], [1.0], [4.0]])])
def test_celltype_accepts_only_endo_epi_mcell(value):
    """celltype is an integer flag: 0 ENDO, 1 EPI, 2 MCELL, and nothing else."""
    with pytest.raises(ValueError, match='celltype'):
        Tomek(dt=_DT).set_parameter('celltype', value)


def test_a_table_constant_cannot_be_set():
    """KNa3 is folded into the voltage table; changing it later would be inert."""
    with pytest.raises(ValueError, match='not a tunable parameter'):
        Tomek(dt=_DT).set_parameter('KNa3', 90.0)


def test_extracellular_concentrations_take_one_value_for_the_tissue():
    """Raising Ko to 10 mM moves EK from about -89 to about -71 mV, so IK1 turns
    inward at the resting potential: about +16 mV/ms on the first step, against
    about 0 at 5 mM. A column of equal values (what the front end passes) is
    accepted; different values, or a non-positive one, are refused."""
    _rest, U_rest = _model(2)
    assert np.max(np.abs(_rest.differentiate(U_rest).numpy())) < 0.5
    model, U = _model(2, {'Ko': np.array([[10.0], [10.0]])})
    assert float(model.get_parameter('Ko')) == 10.0
    assert np.all(model.differentiate(U).numpy() > 5.0)
    with pytest.raises(ValueError, match='one value for all the nodes'):
        Tomek(dt=_DT).set_parameter('Nao', np.array([[140.0], [145.0]]))
    with pytest.raises(ValueError, match='must be positive'):
        Tomek(dt=_DT).set_parameter('Cao', 0.0)


def test_changing_nao_after_initialisation_rebuilds_the_table():
    """Nao enters the voltage table: set after initialisation, it must give the
    same step as a model built with it."""
    before, U_before = _model(1)
    before.set_parameter('Nao', 130.0)
    built, U_built = _model(1, {'Nao': 130.0})
    np.testing.assert_allclose(before.differentiate(U_before).numpy(),
                               built.differentiate(U_built).numpy(), rtol=1.0e-12)


def test_effective_conductance_is_coefficient_times_celltype_times_base():
    """GKr on ENDO, EPI, MCELL nodes with per-node bases and a global block."""
    base  = np.array([[0.0321], [0.0200], [0.0100]])
    model, _U = _model(3, {'celltype': np.array([[0.0], [1.0], [2.0]]),
                           'GKr_b': base, 'CoefKr': 0.5})
    np.testing.assert_allclose(model.effective_parameter('GKr').numpy(),
                               0.5 * np.array([1.0, 1.3, 0.8]) * base.ravel(), rtol=1.0e-14)
    # CoefCaL acts on the permeability, before the Na and K fractions are taken
    model.set_parameter('CoefCaL', 0.25)
    np.testing.assert_allclose(model.effective_parameter('PCa').numpy(),
                               0.25 * np.array([1.0, 1.2, 2.0]) * 8.3757e-05, rtol=1.0e-14)


def test_a_heterogeneous_column_matches_uniform_models():
    """Each node of an ENDO/EPI/MCELL column follows its own uniform model."""
    mixed, U_mixed = _model(3, {'celltype': np.array([[0.0], [1.0], [2.0]])})
    _pace(mixed, U_mixed, 4.0)
    mixed_states = mixed.get_state_variables()
    for node, celltype in enumerate([0.0, 1.0, 2.0]):
        alone, U_alone = _model(1, {'celltype': celltype})
        _pace(alone, U_alone, 4.0)
        np.testing.assert_allclose(U_mixed.numpy()[node], U_alone.numpy()[0], rtol=0.0, atol=1.0e-4)
        for name, values in alone.get_state_variables().items():
            np.testing.assert_allclose(mixed_states[name][node], values[0], rtol=1.0e-9, atol=1.0e-15,
                                       err_msg='celltype {} variable {}'.format(celltype, name))


def test_epi_nodes_use_the_scaled_ito_inactivation():
    """One forward Euler step of iF at -40 mV on an ENDO and an EPI node, against
    the model equations. EPI scales the time constant by delta_epi(V), which a
    per-node scalar cannot express; a column mix-up would give both nodes the
    same update, and the uniform-model comparison above could not see it."""
    model, _U = _model(2, {'celltype': np.array([[0.0], [1.0]])}, rush_larsen=False)
    V   = -40.0
    U   = tf.Variable(np.full((2, 1), V, dtype=np.float32))
    iF0 = model.get_state_variables()['iF']
    model.differentiate(U)
    iss       = 1.0 / (1.0 + np.exp((V + 43.94) / 5.711))
    tiF_b     = 4.562 + 1.0 / (0.3933 * np.exp(-(V + 100.0) / 100.0) + 0.08004 * np.exp((V + 50.0) / 16.59))
    delta_epi = 1.0 - 0.95 / (1.0 + np.exp((V + 70.0) / 5.0))
    tau       = np.array([tiF_b, tiF_b * delta_epi])
    expected  = iF0 + _DT * (iss - iF0) / tau
    np.testing.assert_allclose(model.get_state_variables()['iF'], expected, rtol=1.0e-6)
    assert abs(expected[1] - expected[0]) > 1.0e-3 * abs(expected[0] - iF0[0])


# ---- the IKr Markov chain ------------------------------------------------------
_IKR = ('C3', 'C2', 'C1', 'O', 'I')


def _ikr_generator(V: float) -> np.ndarray:
    """Q(V) of the IKr chain (dx/dt = Q x, states _IKR), from the model equations."""
    x   = V * 96485.0 / (8314.0 * 310.0)
    a   = 0.1161 * np.exp(0.2990 * x)
    b   = 0.2442 * np.exp(-1.604 * x)
    a1, b1 = 0.154375, 0.1911
    a2  = 0.0578 * np.exp(0.9710 * x)
    b2  = 0.349e-3 * np.exp(-1.062 * x)
    ai  = 0.2533 * np.exp(0.5953 * x)
    bi  = 0.06525 * np.exp(-0.8209 * x)
    aci = 0.52e-4 * np.exp(1.525 * x)
    bic = (b2 * bi * aci) / (a2 * ai)
    return(np.array([[-a,  b,         0.0,               0.0,         0.0],
                     [a,   -(b + a1), b1,                0.0,         0.0],
                     [0.0, a1,        -(b1 + a2 + aci),  b2,          bic],
                     [0.0, 0.0,       a2,                -(b2 + ai),  bi],
                     [0.0, 0.0,       aci,               ai,          -(bic + bi)]]))


@pytest.mark.parametrize('V', [-80.0, 20.0, 150.0])
def test_one_ikr_step_is_the_matrix_exponential(V):
    """At a potential where the chain is not stiff, one default step moves the
    IKr states by exp(dt Q(V)), computed here by eigen-decomposition."""
    model, _U = _model(1, rush_larsen=True)
    x0 = np.array([model.get_state_variables()[name][0] for name in _IKR])
    model.differentiate(tf.Variable([[V]], dtype=tf.float64))
    lam, vec = np.linalg.eig(_DT * _ikr_generator(V))
    step = np.real(vec @ np.diag(np.exp(lam)) @ np.linalg.inv(vec))
    after = np.array([model.get_state_variables()[name][0] for name in _IKR])
    np.testing.assert_allclose(after, step @ x0, rtol=1.0e-7, atol=1.0e-15)


def test_the_ikr_chain_stays_a_distribution_under_a_shock_potential():
    """At +860 mV the generator has eigenvalues near -1e17 /ms. In the default
    mode the states stay nonnegative, keep their total, and O settles
    monotonically near 1e-3; forward Euler would jump between 0 and 1."""
    model, _U = _model(1, rush_larsen=True)
    total = sum(model.get_state_variables()[name][0] for name in _IKR)
    U = tf.Variable([[860.0]], dtype=tf.float64)
    O = []
    for _step in range(100):
        model.differentiate(U)
        states = model.get_state_variables()
        x = np.array([states[name][0] for name in _IKR])
        assert np.all(x >= 0.0)
        assert x.sum() == pytest.approx(total, rel=1.0e-12)
        O.append(x[_IKR.index('O')])
    assert max(O) < 1.0e-2
    increments = np.diff(O)
    assert np.all(increments[1:] * increments[:-1] >= 0.0), 'O oscillates'


# ---- physics hooks -------------------------------------------------------------
@pytest.mark.parametrize('rush_larsen', [False, True], ids=['forward_euler', 'rush_larsen'])
def test_zero_sodium_conductance_removes_the_upstroke(rush_larsen):
    """A scar-like node (GNa = 0) is only lifted by the stimulus itself."""
    model, U = _model(2, {'GNa': np.array([[_GNA], [0.0]])}, rush_larsen)
    peak = _pace(model, U, 10.0)
    assert peak[0] > 20.0, 'normal node did not fire: peak {:.1f} mV'.format(peak[0])
    assert peak[1] < -10.0, 'GNa = 0 node fired: peak {:.1f} mV'.format(peak[1])


def test_calcium_is_held_in_millimolar():
    """Cai is stored in the unit of the equations (8.1583e-5 mM at rest)."""
    model, _U = _model(1)
    assert model.get_state_variables()['Cai'][0] == pytest.approx(8.1583e-05, rel=1.0e-12)


def test_ghk_terms_take_their_limit_near_zero_potential():
    """The GHK flux terms are 0/0 at 0 mV. At 0 and within the 1e-6 mV band
    around it the model must return the exact limit: finite, and equal to the
    direct formula 1e-3 mV away (the mean of the two sides cancels the slope).
    Every node starts from the same state, so dU differs only through V."""
    inside  = [0.0, 5.0e-7, -5.0e-7, 1.0e-9, -1.0e-9]
    outside = [-1.0e-3, 1.0e-3]
    model, _U = _model(len(inside) + len(outside))
    U  = tf.Variable(np.reshape(np.array(inside + outside), (-1, 1)), dtype=tf.float64)
    dU = np.reshape(model.differentiate(U).numpy(), (-1,))
    assert np.all(np.isfinite(dU))
    for name, values in model.get_state_variables().items():
        assert np.all(np.isfinite(values)), name
    np.testing.assert_allclose(dU[:len(inside)], np.mean(dU[len(inside):]), rtol=1.0e-6)


@pytest.mark.parametrize('V', [300.0, 330.0, 500.0])
def test_forward_euler_keeps_a_saturated_gate_at_high_potential(V):
    """Above +300 mV the steady states of m and mL are exactly 1 and tm is
    below 1e-16 ms. The reference advances m + dt (m_inf - m)/tm, which leaves
    m = 1 unchanged; written as A + B m with A = dt m_inf/tm and B = 1 - dt/tm
    the two terms cancel and m came out as 0 at +330 mV."""
    model, _U = _model(1, rush_larsen=False)
    states = model.get_state_variables()
    states['m']  = np.ones(1)
    states['mL'] = np.ones(1)
    model.set_state_variables(states)
    model.differentiate(tf.Variable([[V]], dtype=tf.float64))
    after = model.get_state_variables()
    assert after['m'][0] == 1.0
    assert after['mL'][0] == 1.0


# ---- the parameter-file front end ---------------------------------------------
def test_parameter_file_selects_the_model_and_maps_region_parameters():
    """im = Tomek selects the class; celltype and GNa can differ by region."""
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(['-num_imp_regions', '2',
                                        '-imp_region[0].im', 'Tomek',
                                        '-imp_region[0].num_IDs', '1', '-imp_region[0].ID[0]', '1',
                                        '-imp_region[1].im', 'Tomek',
                                        '-imp_region[1].num_IDs', '1', '-imp_region[1].ID[0]', '2',
                                        '-imp_region[1].im_param', 'celltype=1,GNa=0']))
    assert mapper.ionic_model_class() is Tomek
    maps = mapper.ionic_parameter_maps(Tomek(dt=_DT), {1, 2})
    assert maps['celltype'] == {1: 0.0, 2: 1.0}
    assert maps['GNa'] == {1: pytest.approx(_GNA), 2: 0.0}


def test_cell_type_constructor_argument_sets_the_uniform_type():
    """cell_type names the type the whole tissue starts from; the reference
    model file declares ENDO as the default."""
    assert Tomek(dt=_DT).cell_type() == 'ENDO'
    assert float(Tomek(dt=_DT).get_parameter('celltype').numpy()) == 0.0
    assert float(Tomek(dt=_DT, cell_type='MCELL').get_parameter('celltype').numpy()) == 2.0
    with pytest.raises(ValueError, match='cell_type'):
        Tomek(dt=_DT, cell_type='MID')


def test_cell_type_default_returns_base_values():
    """Only celltype depends on the cell type: every tunable conductance is a
    base value, and the type factors are applied on top of it."""
    model = Tomek(dt=_DT)
    for cell_type, type_id in (('ENDO', 0.0), ('EPI', 1.0), ('MCELL', 2.0)):
        assert model.cell_type_default('celltype', cell_type) == type_id
        assert model.cell_type_default('GKs_b', cell_type) == pytest.approx(0.0011)
        assert model.cell_type_default('GNa', cell_type) == pytest.approx(11.7802)

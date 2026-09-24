#!/usr/bin/env python
"""
    Tier-1 unit tests for the parameter mapping
    (gpuSolve.carp_compatibility.parametermapper).

    This is where the two vocabularies meet, so the tests pin the three things
    that are easy to get silently wrong:

      * the DEFAULTS. A file that omits a key must mean what the same file
        means to the simulator whose format it is, so the defaults are the
        reference's (dt 5 us, g_il 0.174 S/m, cellSurfVolRatio 0.14 um^-1 ...).
        The one deliberate exception is a cell parameter no `im_param` names,
        which keeps the gpuSolve class default so that a parameter file and a
        hand-written script agree.
      * the UNIT CONVERSION. Conductivities arrive in S/m and coordinates in
        micrometres, so sigma = 1.0e5 * g * g_mult / (beta * volFrac) lands in
        um^2/ms. With the reference's own defaults this is 1.24e5 um^2/ms,
        the textbook 1.24e-3 cm^2/ms.
      * the CONDUCTIVITY RULE. bidm_eqv_mono defaults to 1, which makes the
        monodomain conductivity the half harmonic mean of the intracellular and
        extracellular values rather than the intracellular one alone. A file
        that says nothing therefore does NOT get g_il, and a mapper that
        assumed it would be about 20% out with nothing to show for it.

    CPU-only, no mesh, no solve.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import pytest

from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.parametermapper import expand_idset
from gpuSolve.carp_compatibility.parametermapper import parse_im_param
from gpuSolve.carp_compatibility.parametermapper import apply_param_mod
from gpuSolve.force_terms import Stimulus
from gpuSolve.ionic.mms2v import ModifiedMS2v
from gpuSolve.ionic.ms2v import MitchellSchaeffer2v

_TAGS = {1, 2, 3}


def _mapper(argv: list) -> ParameterMapper:
    """Resolve a command line straight into a mapper."""
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


# ---- tag lists -------------------------------------------------------------
@pytest.mark.parametrize('text,expected', [
    ('1', [1]),
    ('1:3', [1, 2, 3]),
    ('1:3,7', [1, 2, 3, 7]),
    ('100:2:106', [100, 102, 104, 106]),
    ('1:3 7', [1, 2, 3, 7]),
])
def test_expand_idset(text, expected):
    """Ranges are inclusive at both ends and may carry a stride."""
    assert expand_idset(text) == expected


def test_indexed_and_aggregate_tag_lists_agree():
    """The legacy indexed form and the aggregate form select the same tags."""
    indexed = _mapper(['-num_gregions', '1', '-gregion[0].num_IDs', '3',
                       '-gregion[0].ID[0]', '1', '-gregion[0].ID[1]', '2',
                       '-gregion[0].ID[2]', '3'])
    aggregate = _mapper(['-num_gregions', '1', '-gregion[0].ID', '1:3'])
    assert indexed.region_tags('gregion', 0) == aggregate.region_tags('gregion', 0) == [1, 2, 3]


# ---- defaults and units ----------------------------------------------------
def test_defaults_are_the_reference_ones():
    """An empty command line resolves to the documented defaults, with dt
    converted from microseconds to the milliseconds gpuSolve works in."""
    config = _mapper([]).solver_config()
    assert config['mesh_file_name'] == 'project'
    assert config['dt'] == pytest.approx(0.005)          # 5 us
    assert config['Tend'] == pytest.approx(100.0)        # ms
    assert config['dt_per_plot'] == 600                  # spacedt 3 ms / dt
    # the one solver default that is not the reference's: renumbering is on
    assert config['use_renumbering'] is True


def test_conductivity_conversion_with_the_default_rule():
    """bidm_eqv_mono defaults to 1: the conductivity is the half harmonic mean
    of the two domains, NOT the intracellular value."""
    maps  = _mapper([]).element_property_maps(_TAGS)
    g_i, g_e = 0.174, 0.625
    expected = 1.0e5 * (g_i * g_e) / (g_i + g_e)
    assert maps['sigma_l'][1] == pytest.approx(expected)
    # and beta is cellSurfVolRatio * volFrac
    assert maps['beta'][1] == pytest.approx(0.14)


def test_conductivity_conversion_without_the_equivalence():
    """With bidm_eqv_mono = 0 the intracellular value is used directly, and
    g_mult scales it. 1 S/m becomes 1.0e5 um^2/ms, i.e. 1.0e-3 cm^2/ms."""
    maps = _mapper(['-bidm_eqv_mono', '0', '-gregion[0].g_il', '1.0',
                    '-gregion[0].g_it', '0.25', '-gregion[0].g_mult', '2.0']).element_property_maps(_TAGS)
    assert maps['sigma_l'][1] == pytest.approx(2.0e5)
    assert maps['sigma_t'][1] == pytest.approx(0.5e5)


def test_beta_folds_in_volfrac():
    """beta and volFrac only ever appear multiplied, so they are one property."""
    maps = _mapper(['-imp_region[0].cellSurfVolRatio', '0.2',
                    '-imp_region[0].volFrac', '0.5']).element_property_maps(_TAGS)
    assert maps['beta'][1] == pytest.approx(0.1)


# ---- which region governs which tag ----------------------------------------
def test_a_region_without_ids_claims_every_tag():
    """A region that lists no tag is assigned implicitly to all of them."""
    mapper = _mapper(['-bidm_eqv_mono', '0', '-gregion[0].g_il', '1.0'])
    maps   = mapper.element_property_maps(_TAGS)
    assert all(maps['sigma_l'][tag] == pytest.approx(1.0e5) for tag in _TAGS)


def test_a_tag_claimed_by_nobody_falls_back_to_the_defaults():
    """An explicit ID list claims only what it names; the rest take the
    documented defaults rather than stopping the run."""
    mapper = _mapper(['-bidm_eqv_mono', '0', '-num_gregions', '1',
                      '-gregion[0].ID', '1', '-gregion[0].g_il', '1.0'])
    maps   = mapper.element_property_maps(_TAGS)
    assert maps['sigma_l'][1] == pytest.approx(1.0e5)
    assert maps['sigma_l'][2] == pytest.approx(1.0e5 * 0.174)
    assert maps['sigma_l'][3] == pytest.approx(1.0e5 * 0.174)


def test_an_explicit_claim_beats_an_implicit_one():
    """gregion[1] names tag 3, so it governs it even though gregion[0] claims
    every tag implicitly."""
    mapper = _mapper(['-bidm_eqv_mono', '0', '-num_gregions', '2',
                      '-gregion[0].g_il', '1.0',
                      '-gregion[1].g_il', '2.0', '-gregion[1].ID', '3'])
    maps = mapper.element_property_maps(_TAGS)
    assert maps['sigma_l'][1] == pytest.approx(1.0e5)
    assert maps['sigma_l'][3] == pytest.approx(2.0e5)


def test_a_tag_that_is_not_in_the_mesh_is_reported_and_the_run_continues():
    """Listing a tag the mesh does not have is a note, not a failure."""
    mapper = _mapper(['-num_gregions', '1', '-gregion[0].ID', '1,99'])
    mapper.element_property_maps(_TAGS)
    assert any('99' in note for note in mapper.notes())


# ---- cell models -----------------------------------------------------------
def test_im_param_aliases():
    """The cell-parameter names that differ are translated; the rest pass."""
    assert parse_im_param('V_gate=0.1,a_crit=0.2,tau_in=0.3') == {
        'u_gate': ('=', 0.1, False),
        'u_crit': ('=', 0.2, False),
        'tau_in': ('=', 0.3, False)}


@pytest.mark.parametrize('item,expected', [
    ('tau_in=0.3',  ('=', 0.3,  False)),
    ('tau_in*0.3',  ('*', 0.3,  False)),
    ('tau_in/2',    ('/', 2.0,  False)),
    ('tau_in+0.05', ('+', 0.05, False)),
    ('tau_in-0.05', ('-', 0.05, False)),
    ('tau_in-10%',  ('-', 10.0, True)),
    ('tau_in = 0.3', ('=', 0.3, False)),
    ('tau_in=-0.3', ('=', -0.3, False)),
])
def test_im_param_modifier_forms(item, expected):
    """The name is cut at the first operator, and `%` is kept as a flag."""
    assert parse_im_param(item) == {'tau_in': expected}


@pytest.mark.parametrize('base,item,expected', [
    (0.5, 'tau_in=0.3',  0.3),
    (0.5, 'tau_in*0.3',  0.15),
    (0.5, 'tau_in/2',    0.25),
    (0.5, 'tau_in+0.25', 0.75),
    (0.5, 'tau_in-0.25', 0.25),
    (0.5, 'tau_in-10%',  0.45),
    (0.5, 'tau_in=10%',  0.05),
])
def test_apply_param_mod(base, item, expected):
    """A modifier resolves against the cell model default."""
    modifier = parse_im_param(item)['tau_in']
    assert apply_param_mod(base, modifier) == pytest.approx(expected)


@pytest.mark.parametrize('text', ['tau_in', 'tau_in*', 'tau_in*abc', '*0.3'])
def test_im_param_rejects_malformed(text):
    """A malformed item is an error, not a silently ignored default."""
    with pytest.raises(ValueError):
        parse_im_param(text)


def test_im_param_modifier_scales_the_model_default():
    """tau_in*0.5 halves the gpuSolve default of the selected model."""
    mapper = _mapper(['-imp_region[0].im', 'mMS',
                      '-imp_region[0].im_param', 'tau_in*0.5',
                      '-imp_region[0].ID', '1'])
    model   = mapper.ionic_model_class()()
    default = float(model.get_parameter('tau_in'))
    maps    = mapper.ionic_parameter_maps(model, {1})
    assert maps['tau_in'][1] == pytest.approx(0.5 * default)


@pytest.mark.parametrize('argv,expected', [
    (['-imp_region[0].im', 'mMS'], ModifiedMS2v),
    (['-imp_region[0].im', 'MitchellSchaeffer'], MitchellSchaeffer2v),
    (['-imp_region[0].im', 'MitchellSchaeffer',
      '-imp_region[0].im_param', 'a_crit=0.1'], ModifiedMS2v),
])
def test_cell_model_selection(argv, expected):
    """a_crit is what separates the plain model from the modified one."""
    assert _mapper(argv).ionic_model_class() is expected


def test_unnamed_cell_parameters_keep_the_class_default():
    """im_param names u_gate only, so tau_out keeps the gpuSolve default and a
    parameter file and a script agree wherever the file is silent."""
    mapper = _mapper(['-imp_region[0].im', 'mMS', '-imp_region[0].im_param', 'V_gate=0.05'])
    model  = ModifiedMS2v(dt=0.1)
    maps   = mapper.ionic_parameter_maps(model, _TAGS)
    assert set(maps.keys()) == {'u_gate'}
    assert maps['u_gate'][1] == pytest.approx(0.05)
    assert float(model.get_parameter('tau_out')) == pytest.approx(9.0)


def test_a_parameter_the_model_does_not_have_is_rejected():
    """A misspelled cell parameter stops the run instead of being inert."""
    mapper = _mapper(['-imp_region[0].im', 'mMS', '-imp_region[0].im_param', 'G_Na=1.0'])
    with pytest.raises(ValueError) as excinfo:
        mapper.ionic_parameter_maps(ModifiedMS2v(dt=0.1), _TAGS)
    assert 'G_Na' in str(excinfo.value)


# ---- solver settings, stimuli and diagnostics ------------------------------
@pytest.mark.parametrize('norm,absolute,relative', [
    ('0', 1.0e-6, 0.0), ('1', 1.0e-6, 0.0), ('2', 0.0, 1.0e-6), ('3', 1.0e-6, 1.0e-6)])
def test_cg_norm_arms_the_matching_stopping_test(norm, absolute, relative):
    """cg_norm_parab selects absolute, relative or both; the disarmed one is
    set to 0.0, the value that can never fire."""
    settings = _mapper(['-cg_tol_parab', '1.0e-6', '-cg_norm_parab', norm]).solver_settings()
    assert settings['toll'] == pytest.approx(absolute)
    assert settings['toll_rel'] == pytest.approx(relative)


def test_stimulus_protocol_defaults_are_derived_from_tend():
    """A protocol that says nothing is one pulse covering the simulation."""
    props, geometry = _mapper(['-tend', '50', '-num_stim', '1',
                               '-stim[0].pulse.strength', '60',
                               '-stim[0].elec.p1[0]', '1000']).stimuli()[0]
    assert props['tstart'] == pytest.approx(0.0)
    assert props['duration'] == pytest.approx(50.0)
    assert props['nstim'] == 1
    assert props['intensity'] == pytest.approx(60.0)
    assert geometry == {'p0': [0.0, 0.0, 0.0], 'p1': [1000.0, 0.0, 0.0]}


def test_a_vertex_file_defines_the_electrode_and_wins_over_a_box():
    """A non-empty elec.vtx_file names the stimulated nodes outright, so it
    takes precedence over any box, and the override is reported."""
    plain = _mapper(['-num_stim', '1', '-stim[0].elec.vtx_file', 'electrode.vtx'])
    _props, geometry = plain.stimuli()[0]
    assert geometry == {'vtx_file': 'electrode.vtx'}

    both = _mapper(['-num_stim', '1', '-stim[0].elec.vtx_file', 'electrode.vtx',
                    '-stim[0].elec.p1[0]', '1000'])
    _props, geometry = both.stimuli()[0]
    assert geometry == {'vtx_file': 'electrode.vtx'}
    assert any('vertex file' in note for note in both.notes())


def test_the_pacing_protocol_timing_is_preserved():
    """ptcl.start / duration / npls / bcl map one to one onto the Stimulus, and
    the built Stimulus fires when the protocol says: npls pulses of `duration`,
    the first at `start` and the rest every `bcl` after it, then nothing."""
    import numpy as np

    props, _box = _mapper(['-tend', '400', '-num_stim', '1',
                           '-stim[0].pulse.strength', '60',
                           '-stim[0].ptcl.start', '50',
                           '-stim[0].ptcl.duration', '2',
                           '-stim[0].ptcl.npls', '3',
                           '-stim[0].ptcl.bcl', '100',
                           '-stim[0].elec.p1[0]', '1000']).stimuli()[0]
    assert props['tstart'] == pytest.approx(50.0)
    assert props['duration'] == pytest.approx(2.0)
    assert props['nstim'] == 3
    assert props['period'] == pytest.approx(100.0)

    stimulus = Stimulus(props)
    stimulus.set_stimregion(np.ones(shape=(4,), dtype=bool))
    # a point inside each pulse, and points that must be quiet: before the
    # first, between two, and after the last one the protocol defines
    for on_time in (50.5, 150.5, 250.5):
        assert float(np.max(stimulus.stimApp(on_time).numpy())) == pytest.approx(60.0)
    for off_time in (49.0, 100.0, 200.0, 260.0, 350.5):
        assert float(np.max(stimulus.stimApp(off_time).numpy())) == pytest.approx(0.0)


def test_a_non_transmembrane_electrode_is_refused():
    """Treating an extracellular electrode as a transmembrane one would change
    the physics, so it is refused rather than approximated."""
    mapper = _mapper(['-num_stim', '1', '-stim[0].crct.type', '2'])
    with pytest.raises(ValueError) as excinfo:
        mapper.stimuli()
    assert 'transmembrane' in str(excinfo.value)


def test_unknown_keys_stop_the_run():
    """A key outside the registry is named and refused, so a typo is never
    silently inert."""
    with pytest.raises(ValueError) as excinfo:
        _mapper(['-no_such_parameter', '1'])
    assert 'no_such_parameter' in str(excinfo.value)


def test_a_counter_that_would_drop_an_entry_is_refused():
    """num_gregions = 1 with gregion[1] assigned would silently lose a region."""
    with pytest.raises(ValueError) as excinfo:
        _mapper(['-num_gregions', '1', '-gregion[1].g_il', '0.2'])
    assert 'num_gregions' in str(excinfo.value)


def test_notes_are_silent_when_the_request_matches_what_is_done():
    """mass_lumping = 0, bidomain = 0 and parab_solve = 1 (the theta method)
    are what gpuSolve does, so none is reported; the other values are."""
    notes = _mapper(['-mass_lumping', '0', '-bidomain', '0']).notes()
    assert not any('mass_lumping' in note for note in notes)
    assert not any('bidomain' in note for note in notes)
    assert not any('parab_solve' in note for note in notes)
    noisy = _mapper(['-mass_lumping', '1', '-bidomain', '1', '-parab_solve', '0']).notes()
    assert any('mass_lumping' in note for note in noisy)
    assert any('bidomain' in note for note in noisy)
    assert any('parab_solve' in note for note in noisy)


# ---- legacy stimulus[] keys ------------------------------------------------
def test_legacy_stimulus_maps_onto_stim():
    """stimulus[] gives the same (props, box) as stim[]: p0 = x0, p1 = x0 + xd."""
    legacy = _mapper(['-tend', '10', '-num_stim', '1',
                      '-stimulus[0].strength', '200', '-stimulus[0].duration', '1',
                      '-stimulus[0].start', '0.5', '-stimulus[0].npls', '3',
                      '-stimulus[0].bcl', '1000',
                      '-stimulus[0].x0', '47000', '-stimulus[0].xd', '1000',
                      '-stimulus[0].y0', '36000', '-stimulus[0].yd', '200',
                      '-stimulus[0].z0', '-21000', '-stimulus[0].zd', '1000'])
    modern = _mapper(['-tend', '10', '-num_stim', '1',
                      '-stim[0].pulse.strength', '200', '-stim[0].ptcl.duration', '1',
                      '-stim[0].ptcl.start', '0.5', '-stim[0].ptcl.npls', '3',
                      '-stim[0].ptcl.bcl', '1000',
                      '-stim[0].elec.p0[0]', '47000', '-stim[0].elec.p1[0]', '48000',
                      '-stim[0].elec.p0[1]', '36000', '-stim[0].elec.p1[1]', '36200',
                      '-stim[0].elec.p0[2]', '-21000', '-stim[0].elec.p1[2]', '-20000'])
    assert legacy.stimuli() == modern.stimuli()
    # the square-pulse approximation is reported, not hidden
    assert any('truncated exponential' in note for note in legacy.notes())


def test_unnamed_stimulus_takes_the_reference_label():
    """An unnamed stimulus is Stimulus_<i> in both families; a name is kept."""
    legacy = _mapper(['-tend', '10', '-num_stim', '2',
                      '-stimulus[0].strength', '5', '-stimulus[0].xd', '400',
                      '-stimulus[1].strength', '5', '-stimulus[1].name', 'S2'])
    assert [props['name'] for props, _geom in legacy.stimuli()] == ['Stimulus_0', 'S2']
    modern = _mapper(['-tend', '10', '-stim[0].pulse.strength', '5',
                      '-stim[0].elec.p1[0]', '400'])
    assert modern.stimuli()[0][0]['name'] == 'Stimulus_0'


def test_legacy_stimulus_centred_box_and_defaults():
    """ctr_def centres the box on (x0, y0, z0); an unset extent is 100 um."""
    props, geometry = _mapper(['-tend', '10', '-stimulus[0].strength', '5',
                               '-stimulus[0].ctr_def', '1', '-stimulus[0].x0', '1000',
                               '-stimulus[0].xd', '400']).stimuli()[0]
    assert geometry == {'p0': [800.0, -50.0, -50.0], 'p1': [1200.0, 50.0, 50.0]}
    assert props['nstim'] == 1 and props['duration'] == 10.0


def test_legacy_stimulus_rejects_other_types_and_mixing():
    """Only type 0 is applied, and the two families cannot be mixed."""
    with pytest.raises(ValueError, match='transmembrane'):
        _mapper(['-stimulus[0].stimtype', '1', '-stimulus[0].strength', '5']).stimuli()
    with pytest.raises(ValueError, match='one of the two families'):
        _mapper(['-num_stim', '2', '-stimulus[0].strength', '5',
                 '-stim[1].pulse.strength', '5']).stimuli()


def test_lats_and_meshformat_accepted_not_actuated():
    """LAT and mesh-format keys are accepted and reported as not acted upon."""
    mapper = _mapper(['-meshformat', '0', '-num_LATs', '1', '-lats[0].ID', 'LATs',
                      '-lats[0].all', '0', '-lats[0].measurand', '0',
                      '-lats[0].threshold', '-10', '-lats[0].mode', '0'])
    assert any('activation times are not computed' in note for note in mapper.notes())


# ---- im_param flags (cell type) --------------------------------------------
def _ttp_regions(flags: list) -> list:
    argv = ['-num_imp_regions', str(len(flags))]
    for index, flag in enumerate(flags):
        text = 'GKs*1.5' + (',flags={}'.format(flag) if flag else '')
        argv += ['-imp_region[{}].im'.format(index), 'tenTusscherPanfilov',
                 '-imp_region[{}].im_param'.format(index), text,
                 '-imp_region[{}].ID'.format(index), str(1 + index)]
    return(argv)


def test_flags_select_cell_type():
    """flags=ENDO in every region builds the ENDO model; no flags means EPI."""
    assert _mapper(_ttp_regions(['ENDO', 'ENDO'])).ionic_model_options() == {'cell_type': 'ENDO'}
    assert _mapper(_ttp_regions([None])).ionic_model_options() == {'cell_type': 'EPI'}
    # the flags item is not a parameter modifier
    assert parse_im_param('GKs*1.5,flags=ENDO') == {'GKs': ('*', 1.5, False)}


def test_flags_may_differ_by_region():
    """Mixed cell types are per node now: the constructor gets the model
    default, and each region's type travels in the celltype map."""
    assert _mapper(_ttp_regions(['ENDO', 'MCELL'])).ionic_model_options() == {'cell_type': 'EPI'}
    assert _mapper(_ttp_regions(['ENDO', None])).ionic_model_options() == {'cell_type': 'EPI'}


@pytest.mark.parametrize('flags,match', [
    (['EPI|ENDO'], 'not a cell type'),
    (['APEX'], 'not a cell type'),
])
def test_flags_rejected(flags, match):
    with pytest.raises(ValueError, match=match):
        _mapper(_ttp_regions(flags)).ionic_model_options()


def test_flags_on_model_without_cell_types():
    mapper = _mapper(['-imp_region[0].im', 'mMS', '-imp_region[0].im_param', 'flags=ENDO'])
    with pytest.raises(ValueError, match='has no cell types'):
        mapper.ionic_model_options()


def test_ttp_conductance_modifiers_resolve_against_cell_type():
    """GKr/GKs exist before the first step and scale the cell-type default:
    GKs is 0.392 for ENDO, 0.098 for MCELL, and the modifier scales that."""
    from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov
    for cell_type, gks in (('ENDO', 0.392), ('MCELL', 0.098)):
        argv = ['-imp_region[0].im', 'tenTusscherPanfilov', '-imp_region[0].im_param',
                'GKr*1.5,GKs*1.5,flags={}'.format(cell_type), '-imp_region[0].ID', '1']
        mapper = _mapper(argv)
        model  = mapper.ionic_model_class()(dt=0.02, **mapper.ionic_model_options())
        maps   = mapper.ionic_parameter_maps(model, {1, 2})
        assert maps['GKr'][1] == pytest.approx(1.5 * 0.153)
        assert maps['GKs'][1] == pytest.approx(1.5 * gks)
        # tag 2 has no region, so it keeps the model default
        assert maps['GKs'][2] == pytest.approx(gks)


def test_ttp_per_node_conductance_survives_initialisation():
    """A per-node GKr set before initialize_state_variables is kept, not
    overwritten by the default."""
    import numpy as np
    import tensorflow as tf
    from gpuSolve.ionic.ten_tusscher_panfilov import TenTusscherPanfilov
    model = TenTusscherPanfilov(dt=0.02, n_nodes=3)
    values = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    model.set_parameter('GKr', values)
    model.initialize_state_variables(tf.Variable(tf.zeros([3, 1], dtype=tf.float32)))
    np.testing.assert_allclose(model.get_parameter('GKr').numpy(), values)
    np.testing.assert_allclose(model.get_parameter('GKs').numpy(), np.full((3, 1), 0.392))


# ---- time scheme of the diffusion step -------------------------------------
@pytest.mark.parametrize('argv,theta', [
    ([], 0.5),                                          # parab_solve = 1, theta = 0.5
    (['-theta', '0.75'], 0.75),
    (['-theta', '1'], 1.0),                             # implicit Euler, accepted here
    (['-parab_solve', '0'], 1.0),                       # not implemented: implicit Euler
    (['-parab_solve', '2', '-theta', '0.5'], 1.0),
])
def test_the_diffusion_theta(argv, theta):
    assert _mapper(argv).solver_config()['theta'] == pytest.approx(theta)


def test_a_theta_outside_the_reference_range_is_noted_and_one_outside_0_1_refused():
    assert any('theta' in note for note in _mapper(['-theta', '1']).notes())
    assert not any('theta' in note for note in _mapper(['-theta', '0.5']).notes())
    for bad in ('0', '1.5'):
        with pytest.raises(ValueError, match='theta'):
            _mapper(['-theta', bad]).solver_config()

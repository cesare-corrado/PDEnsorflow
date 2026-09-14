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
    assert config['use_renumbering'] is False


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
        'u_gate': 0.1, 'u_crit': 0.2, 'tau_in': 0.3}


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
    """mass_lumping = 0 and bidomain = 0 are what gpuSolve does, so neither is
    reported; parab_solve never matches implicit Euler, so it always is."""
    notes = _mapper(['-mass_lumping', '0', '-bidomain', '0']).notes()
    assert not any('mass_lumping' in note for note in notes)
    assert not any('bidomain' in note for note in notes)
    assert any('parab_solve' in note for note in notes)
    noisy = _mapper(['-mass_lumping', '1', '-bidomain', '1']).notes()
    assert any('mass_lumping' in note for note in noisy)
    assert any('bidomain' in note for note in noisy)

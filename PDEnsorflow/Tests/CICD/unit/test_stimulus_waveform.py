#!/usr/bin/env python
"""
    Tier-1 unit tests for the shape of a stimulus pulse
    (gpuSolve.force_terms.Stimulus, and the shape the .par front end selects).

    A pulse that is a square where the reference simulator applies a truncated
    exponential delivers more charge than the parameter file asks for: about
    2.6% more for a 2 ms pulse. The tests therefore pin

      * the WAVEFORM, against the reference's own sampled pulse. The expected
        values below are read from a trace the reference wrote itself
        (-stim[0].pulse.dumpTrace 1, strength 60, duration 2 ms, dt 5 us),
        which is why they are quoted to the six decimals that file carries.
      * that the square pulse stays the DEFAULT of the class, so that a script
        written before the shape existed keeps its numerics, and
      * that the .par front end selects the truncated exponential, because that
        is the reference's default in both of its stimulus families.

    CPU-only, no mesh, no solve.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

import pytest

from gpuSolve.carp_compatibility.optionreader import OptionReader
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.force_terms import Stimulus

# One pulse of the reference, as it sampled it itself: (time since the start of
# the pulse in ms, waveform as a fraction of the strength). duration = 2 ms and
# the defaults tau_edge = 0.01 ms, tau_plateau = 1000 ms, d1 = 1, bias = 0.
# The first sample is exactly zero, the rise is over within a few tau_edge, the
# plateau creeps down as exp(-t/tau_plateau), and the trailing edge takes the
# last 5*tau_edge = 0.05 ms.
REFERENCE_PULSE = ((0.000, 0.000000),
                   (0.005, 0.393467),
                   (0.010, 0.632114),
                   (0.500, 0.999500),
                   (1.000, 0.999000),
                   (1.945, 0.998057),
                   (1.950, 0.998052),
                   (1.955, 0.605349),
                   (1.975, 0.081925),
                   (2.000, 0.006725))

PULSE_START : float = 1.0
PULSE_DURATION : float = 2.0


def _stimulus(shape: str) -> Stimulus:
    """One pulse of the reference trace, with the whole domain stimulated."""
    stimulus = Stimulus({'tstart': PULSE_START,
                         'nstim': 1,
                         'period': 100.0,
                         'duration': PULSE_DURATION,
                         'intensity': 60.0,
                         'shape': shape})
    stimulus.set_stimregion(np.ones(shape=(4,), dtype=bool))
    return(stimulus)


def _mapper(argv: list) -> ParameterMapper:
    """Resolve a command line straight into a mapper."""
    mapper = ParameterMapper()
    mapper.resolve(OptionReader().read(argv))
    return(mapper)


def test_the_truncated_exponential_matches_the_reference_pulse():
    """Every sample of the pulse the reference wrote out is reproduced, to the
    precision that trace file is written with."""
    stimulus = _stimulus('truncexp')
    for elapsed, expected in REFERENCE_PULSE:
        assert stimulus.waveform(PULSE_START + elapsed) == pytest.approx(expected, abs=1.0e-6)


def test_the_truncated_exponential_delivers_less_charge_than_the_square_pulse():
    """The point of the shape: the rise and the trailing edge cost charge that
    a square pulse of the same strength and duration does not lose."""
    shaped = _stimulus('truncexp')
    square = _stimulus('square')
    times = PULSE_START + np.arange(0.0, PULSE_DURATION, 0.005)
    charge = sum(shaped.waveform(t) for t in times) / len(times)
    assert all(square.waveform(t) == 1.0 for t in times)
    # 5*tau_edge of trailing edge plus the rise, out of 2 ms
    assert charge == pytest.approx(0.974, abs=1.0e-3)
    assert charge < 1.0


def test_the_waveform_is_continuous_at_the_trailing_edge():
    """The decay restarts from the plateau height, so the pulse has no jump
    where the two branches of the formula meet."""
    stimulus = _stimulus('truncexp')
    t_edge = PULSE_START + PULSE_DURATION - 5.0 * stimulus.tau_edge()
    below = stimulus.waveform(t_edge - 1.0e-9)
    above = stimulus.waveform(t_edge + 1.0e-9)
    assert above == pytest.approx(below, abs=1.0e-6)


def test_every_pulse_of_a_train_carries_the_same_shape():
    """The waveform is evaluated on the time since the start of the pulse the
    time falls in, so the third pulse of a train is shaped like the first."""
    stimulus = Stimulus({'tstart': 50.0, 'nstim': 3, 'period': 100.0,
                         'duration': PULSE_DURATION, 'intensity': 60.0,
                         'shape': 'truncexp'})
    stimulus.set_stimregion(np.ones(shape=(4,), dtype=bool))
    for elapsed, expected in REFERENCE_PULSE:
        for pulse in range(3):
            time = 50.0 + 100.0 * pulse + elapsed
            assert stimulus.waveform(time) == pytest.approx(expected, abs=1.0e-6)


def test_the_applied_stimulus_is_scaled_by_the_waveform():
    """stimApp and the call operator both carry the shape, so the forcing term
    the solver adds is the shaped pulse and not the full strength."""
    stimulus = _stimulus('truncexp')
    time = PULSE_START + 0.005
    expected = 60.0 * 0.393467
    assert float(np.max(stimulus.stimApp(time).numpy())) == pytest.approx(expected, rel=1.0e-5)
    assert float(np.max(stimulus(time).numpy())) == pytest.approx(expected, rel=1.0e-5)
    # the stimulus is still silent outside its protocol
    assert float(np.max(stimulus.stimApp(0.5).numpy())) == pytest.approx(0.0)


def test_the_square_pulse_is_the_default_and_is_unchanged_by_the_time():
    """The shape is opt-in: a Stimulus built as before is a square pulse, and
    passing the time to it changes nothing."""
    stimulus = _stimulus('square')
    assert stimulus.shape() == 'square'
    assert Stimulus().shape() == 'square'
    for elapsed, _expected in REFERENCE_PULSE:
        time = PULSE_START + elapsed
        assert stimulus.waveform(time) == 1.0
        assert float(np.max(stimulus(time).numpy())) == pytest.approx(60.0)
    assert float(np.max(stimulus().numpy())) == pytest.approx(60.0)


def test_a_shape_that_is_not_implemented_is_refused():
    """A sine or an arbitrary pulse read from a file would silently become a
    square one, so the name is refused when it is set."""
    with pytest.raises(ValueError):
        Stimulus({'shape': 'sine'})
    with pytest.raises(ValueError):
        Stimulus().set_shape('arbitrary')


def test_a_biphasic_pulse_is_refused():
    """d1 < 1 asks for a second phase that reverses the current, which is a
    defibrillation waveform and not a transmembrane forcing term."""
    with pytest.raises(ValueError) as excinfo:
        Stimulus({'shape': 'truncexp', 'd1': 0.5})
    assert 'biphasic' in str(excinfo.value)
    # inert while the square shape is selected: it carries no d1 at all
    Stimulus({'d1': 0.5})


def test_the_front_end_selects_the_truncated_exponential_by_default():
    """The reference defaults stim[].pulse.shape to the truncated exponential,
    and shapes every pulse of its legacy family that way, so a parameter file
    that says nothing about the shape asks for it."""
    for argv in (['-tend', '100', '-num_stim', '1',
                  '-stim[0].pulse.strength', '60',
                  '-stim[0].ptcl.start', '1',
                  '-stim[0].ptcl.duration', '2',
                  '-stim[0].elec.p1[0]', '1000'],
                 ['-tend', '100', '-num_stim', '1',
                  '-stimulus[0].strength', '60',
                  '-stimulus[0].start', '1',
                  '-stimulus[0].duration', '2',
                  '-stimulus[0].xd', '1000']):
        props, _geometry = _mapper(argv).stimuli()[0]
        assert props['shape'] == 'truncexp'
        assert props['tau_edge'] == pytest.approx(0.01)
        assert props['tau_plateau'] == pytest.approx(1000.0)
        assert props['d1'] == pytest.approx(1.0)
        assert props['bias'] == pytest.approx(0.0)
        stimulus = Stimulus(props)
        stimulus.set_stimregion(np.ones(shape=(4,), dtype=bool))
        for elapsed, expected in REFERENCE_PULSE:
            assert stimulus.waveform(1.0 + elapsed) == pytest.approx(expected, abs=1.0e-6)


def test_the_front_end_passes_the_pulse_parameters_through():
    """tau_edge, tau_plateau and bias are read, in both vocabularies: they
    change the charge, so a file that tunes them must be followed."""
    props, _geometry = _mapper(['-tend', '100', '-num_stim', '1',
                                '-stim[0].pulse.strength', '60',
                                '-stim[0].ptcl.duration', '2',
                                '-stim[0].pulse.tau_edge', '0.05',
                                '-stim[0].pulse.tau_plateau', '10',
                                '-stim[0].pulse.bias', '0.1',
                                '-stim[0].elec.p1[0]', '1000']).stimuli()[0]
    assert props['tau_edge'] == pytest.approx(0.05)
    assert props['tau_plateau'] == pytest.approx(10.0)
    assert props['bias'] == pytest.approx(0.1)
    legacy, _geometry = _mapper(['-tend', '100', '-num_stim', '1',
                                 '-stimulus[0].strength', '60',
                                 '-stimulus[0].duration', '2',
                                 '-stimulus[0].tau_edge', '0.05',
                                 '-stimulus[0].tau_plateau', '10',
                                 '-stimulus[0].xd', '1000']).stimuli()[0]
    assert legacy['tau_edge'] == pytest.approx(0.05)
    assert legacy['tau_plateau'] == pytest.approx(10.0)


def test_the_front_end_selects_the_square_pulse_when_the_file_asks_for_it():
    """shape 0 is the reference's square pulse, a constant of unit height."""
    props, _geometry = _mapper(['-tend', '100', '-num_stim', '1',
                                '-stim[0].pulse.strength', '60',
                                '-stim[0].ptcl.duration', '2',
                                '-stim[0].pulse.shape', '0',
                                '-stim[0].elec.p1[0]', '1000']).stimuli()[0]
    assert props['shape'] == 'square'
    assert Stimulus(props).waveform(0.5) == 1.0


def test_a_pulse_shape_that_is_not_sampled_is_refused():
    """A sine (2) or a pulse read from a file (3) would change the charge the
    stimulus delivers, so it is refused rather than approximated."""
    mapper = _mapper(['-tend', '100', '-num_stim', '1',
                      '-stim[0].pulse.strength', '60',
                      '-stim[0].ptcl.duration', '2',
                      '-stim[0].pulse.shape', '2',
                      '-stim[0].elec.p1[0]', '1000'])
    with pytest.raises(ValueError) as excinfo:
        mapper.stimuli()
    assert 'pulse.shape' in str(excinfo.value)

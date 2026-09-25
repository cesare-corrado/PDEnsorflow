#!/usr/bin/env python
"""
    SingleCellRunner: builds and runs the single-cell experiment described by a
    `singlecell` command line (SingleCellOptionReader), with the time
    stepping, stimulus timing and output timing of the reference single-cell
    tool (bench, limpet/src/bench.cc and numerics/timer_utils.cc):

      * time is counted in whole steps, d = lround(t / dt); the run covers
        d = 0 .. lround(duration / dt), both ends included;
      * each step, in this order: write the output (the state at the start of
        the step, with the Iion of the previous step), save the state file if
        its time has come, add the stimulus (V += stim_curr * dt), then
        advance the model at that V (V += dt * dU, dU = -Iion);
      * a stimulus lasts lround(stim_dur / dt) steps (at least one); a regular
        train starts at lround(stim_start / dt) and repeats every
        lround(bcl / dt) steps, numstim times (numstim <= 0: no stimulus);
      * the output starts at lround(start_out / dt) and repeats every
        lround(dt_out / dt) steps;
      * without --duration the run lasts stim_start + bcl (numstim - 1) +
        past_stim, or the last of --stim-times + past_stim, computed in single
        precision as bench does.

    The cell model, its cell type, its plugins and the --imp-par / --plug-par
    modifiers are resolved by the parameter-file front end (ParameterMapper),
    so both executables accept the same names and modifiers.

    Performance: one cell is too little work to fill a device, and the cost of
    a step is the fixed cost of the many small kernels of the model (about 50
    for Courtemanche), not their arithmetic. Two choices follow, both measured
    on Courtemanche, 1 s of activity at dt = 0.01 ms, on the CPU:
      * the steps run inside compiled XLA loops of up to CHUNK_STEPS steps that
        also collect the output, so Python runs once per chunk, not once per
        step or per output (2.10 s with one call per ms, 1.57 s per chunk);
      * the caller runs TensorFlow on one thread (0.97 s): the thread pool
        costs more than it gives on arrays of one element.
    The GPU is slower still for one cell (Tomek, 3 cells: 17-20 s per beat on
    an RTX A2000 against 11 s on the CPU), which is why the default target is
    the CPU.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""
import time
import numpy as np
import tensorflow as tf

from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.ionic.ionicmodelwithplugins import PLUGIN_SEPARATOR
from gpuSolve.ionic.tomek import Tomek
from gpuSolve.ionic.plugins.defib_ashihara_trayanova import DefibAshiharaTrayanova
from gpuSolve.IO.writers.svfilewriter import SvFileWriter
from gpuSolve.carp_compatibility.parametermapper import ParameterMapper
from gpuSolve.carp_compatibility.parametermapper import IONIC_MODELS
from gpuSolve.carp_compatibility.parametermapper import IONIC_PLUGINS
from gpuSolve.carp_compatibility.parametermapper import IONIC_CELL_TYPES as CELL_TYPE_FLAGS
from gpuSolve.carp_compatibility.parametermapper import PLUGIN_LIST_SEPARATOR
from gpuSolve.carp_compatibility.parametermapper import split_param_mod
from gpuSolve.carp_compatibility.parametermapper import IM_PARAM_FLAGS
from gpuSolve.carp_compatibility.parametermapper import PARAM_MOD_OPERATORS
from gpuSolve.carp_compatibility.singlecellwriter import SingleCellWriter
from gpuSolve.carp_compatibility.svlayouts import IMP_DATA_NAMES
from gpuSolve.carp_compatibility.svlayouts import sv_layout
from gpuSolve.carp_compatibility.svstatefile import SvStateFile


# Upper bound of the steps one compiled call advances. Large enough that the
# Python work between calls is negligible (1e5 steps is 1 s of activity at
# dt = 0.01 ms), small enough that the output a call collects stays small.
CHUNK_STEPS : int = 100000

# Dtype of the potential, by model class: the model's own precision, so a
# float64 model is not rounded to float32 at every step through V. Models not
# listed work in float32.
POTENTIAL_DTYPE = {Tomek: tf.float64}

def lround(value: float) -> int:
    """ lround(value) rounds half away from zero, as C's lround (Python's round
        rounds half to even, which would move a stimulus by one step)
    """
    return(int(np.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1))


class SingleCellRunner:
    """
    class SingleCellRunner: set_options(), build(), run().
    """

    def __init__(self, config: dict = None):
        self._options                  = None
        self._mapper : ParameterMapper = None
        self._model                    = None
        self._cell                     = None
        self._plugins : list           = []
        self._imp_name : str           = ''
        self._svfile : SvStateFile     = None
        self._dtype                    = tf.float32
        self._dt : float               = 0.01
        self._last_step : int          = 0
        self._stim_mask : np.ndarray   = None
        self._stim_charge : float      = 0.0
        self._out_first : int          = 0
        self._out_every : int          = 1
        self._save_step : int          = None
        self._U : tf.Variable          = None
        self._Iion : tf.Variable       = None
        self._dumps : list             = []
        self._writer : SingleCellWriter = None
        self._notes : list             = []
        self._warnings : list          = []
        self._max_records : int        = 1
        self._advance_recorded         = None
        self._advance_plain            = None
        self._current_row              = None
        self._elapsed : float          = 0.0
        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    # ---- accessors ------------------------------------------------------------
    def set_options(self, options):
        """ set_options(options) sets the SingleCellOptionReader of the run """
        self._options = options

    def model(self):
        """ model() returns the ionic model (with its plugins, when any) """
        return(self._model)

    def notes(self) -> list:
        """ notes() returns the lines the run reports once, at the start """
        return(self._notes)

    def warnings(self) -> list:
        """ warnings() returns the warnings to repeat at the start and at the
            end of the run (a state file that sets parameters differently)
        """
        return(self._warnings)

    def elapsed(self) -> float:
        """ elapsed() returns the wall time of run(), in seconds """
        return(self._elapsed)

    def last_step(self) -> int:
        """ last_step() returns the index of the last step of the run """
        return(self._last_step)

    def dt(self) -> float:
        """ dt() returns the time step, in ms """
        return(self._dt)

    # ---- construction ---------------------------------------------------------
    def build(self):
        """ build() resolves the timing, builds the model and its initial
            state, and opens the output. A bad input raises ValueError.
        """
        options = self._options
        self._dt = float(options.value('dt'))
        if not self._dt > 0.0:
            raise ValueError('--dt must be positive, got {}'.format(self._dt))
        self.__resolve_timing()
        self.__build_model()
        self.__initialize_state()
        self.__build_dumps()
        self.__compile()
        self.__initial_current()

    def __resolve_timing(self):
        """ the number of steps, the stimulated steps, the output steps and
            the save step, as bench's timers define them
        """
        options = self._options
        dt = self._dt
        times = None
        if options.given('stim-times'):
            times = self.__stim_times(options.value('stim-times'), options.value('DIA'))
        if options.given('duration'):
            duration = options.value('duration')
        elif times is not None:
            duration = times[-1] + options.value('past-stim')
        else:
            duration = (options.value('stim-start')
                        + options.value('bcl') * (options.value('numstim') - 1)
                        + options.value('past-stim'))
        # bench's determine_duration() returns a float
        duration = float(np.float32(duration))
        self._last_step = lround(duration / dt)
        if self._last_step < 0:
            # bench runs no step at all and says nothing
            raise ValueError('the duration of the run is {} ms, which is negative; give '
                             '--duration, or check --numstim and --past-stim'.format(duration))
        # the stimulus: start steps, then the stimulated steps
        span = max(1, lround(options.value('stim-dur') / dt))
        if times is not None:
            starts = np.unique(np.array([lround(value / dt) for value in times], dtype=np.int64))
        else:
            nstim = options.value('numstim')
            first = lround(options.value('stim-start') / dt)
            every = lround(options.value('bcl') / dt) if options.value('bcl') else 1
            starts = first + every * np.arange(max(nstim, 0), dtype=np.int64)
        mask = np.zeros(self._last_step + 1, dtype=bool)
        for start in starts.tolist():
            lo = max(start, 0)
            hi = min(start + span, self._last_step + 1)
            if lo < hi:
                mask[lo:hi] = True
        self._stim_mask = mask
        self._stim_charge = float(options.value('stim-curr')) * dt
        # the output
        self._out_every = lround(options.value('dt-out') / dt) if options.value('dt-out') else 1
        if self._out_every < 1:
            raise ValueError('--dt-out ({} ms) is shorter than half a time step ({} ms)'.format(
                options.value('dt-out'), dt))
        first = lround(options.value('start-out') / dt)
        while first < 0:
            first += self._out_every
        self._out_first = first
        # the state file: bench saves only when --save-ini-time is given
        self._save_step = None
        if options.given('save-ini-time'):
            self._save_step = lround(options.value('save-ini-time') / dt)
            if self._save_step > self._last_step or self._save_step < 0:
                self._notes.append('--save-ini-time {} ms is outside the run (0 to {} ms): no state '
                                   'file is written'.format(options.value('save-ini-time'), duration))
                self._save_step = None
        elif options.given('save-ini-file'):
            self._notes.append('--save-ini-file is given without --save-ini-time: as in bench, no '
                               'state file is written')
        if not options.value('no-trace'):
            self._notes.append('the ionic model trace (Trace_0.dat) is not written by singlecell yet')

    def __stim_times(self, text: str, diastolic: bool) -> list:
        """ the --stim-times list, in ms, sorted; with --DIA each value after
            the first is an interval added to the previous time, as in bench
        """
        values : list = []
        for token in text.split(','):
            try:
                values.append(float(token))
            except ValueError:
                raise ValueError('Error in stimulus timing list: "{}"'.format(text))
            if diastolic and len(values) > 1:
                values[-1] += values[-2]
        return(sorted(values))

    def __build_model(self):
        """ the cell model and its plugins, tuned by --imp-par and --plug-par,
            resolved as a one-region parameter file would be
        """
        options = self._options
        self._imp_name = options.value('imp')
        store = {'num_imp_regions':            '1',
                 'imp_region[0].im':            self._imp_name,
                 'imp_region[0].im_param':      options.value('imp-par'),
                 'imp_region[0].plugins':       options.value('plug-in'),
                 'imp_region[0].plug_param':    options.value('plug-par')}
        self._mapper = ParameterMapper()
        self._mapper.resolve(store)
        modelclass = self._mapper.ionic_model_class()
        self._cell = modelclass(dt=self._dt, **self._mapper.ionic_model_options())
        self._dtype = POTENTIAL_DTYPE.get(modelclass, tf.float32)
        pluginclasses = self._mapper.ionic_plugin_classes()
        self._plugins = [pluginclass(dt=self._dt) for pluginclass in pluginclasses]
        if len(self._plugins) > 0:
            # a run without plugins keeps the bare model, as the tissue front
            # end does, so its numerics are the model's own
            wrapper = IonicModelWithPlugins(dt=self._dt)
            wrapper.set_model(self._cell)
            for plugin in self._plugins:
                wrapper.add_plugin(plugin)
            self._model = wrapper
        else:
            self._model = self._cell
        if options.value('reference-scheme'):
            self.__use_reference_scheme()
        # the modifiers. One region, one tag: the maps hold one value each.
        maps = self._mapper.ionic_parameter_maps(self._model, {0})
        if len(self._plugins) > 0:
            maps.update(self._mapper.plugin_parameter_maps(self._model, {0}))
        for pname, pmap in maps.items():
            self._model.set_parameter(pname, self.__typed(pname, pmap[0]))
        # the state files of this run: the model, and each plugin under the name
        # --plug-in selected it by, which is the section name of a plugin the
        # reference does not have
        names = [name.strip() for name in options.value('plug-in').split(PLUGIN_LIST_SEPARATOR)
                 if len(name.strip()) > 0]
        self._svfile = SvStateFile({'model': self._model, 'imp_name': self._imp_name,
                                    'plugins': list(zip(self._plugins, names))})
        self.__print_model(maps)

    def __use_reference_scheme(self):
        """ --reference-scheme: the reference's integration where gpuSolve's
            default differs
        """
        if isinstance(self._cell, Tomek):
            self._cell.set_use_rush_larsen(False)
        for plugin in self._plugins:
            if isinstance(plugin, DefibAshiharaTrayanova):
                plugin.set_use_reference_form(True)

    def __typed(self, pname: str, value: float):
        """ value in the type of the model's own default for pname: a Python
            float stays a Python float (the base set_parameter makes it a
            float32 constant, the precision of the models that hold floats), a
            tensor default gives a 0-d array of its dtype
        """
        reference = self._model.get_parameter(pname)
        if isinstance(reference, (int, float)):
            return(float(value))
        if isinstance(reference, (tf.Tensor, tf.Variable)):
            return(np.asarray(value, dtype=reference.dtype.as_numpy_dtype))
        return(np.asarray(value, dtype=np.asarray(reference).dtype))

    def __print_model(self, maps: dict):
        """ the model banner, as bench prints it: the model, then each
            modified parameter with its modifier and resolved value
        """
        print('\nIonic model: {}'.format(self._imp_name), flush=True)
        modifiers = self.__modifier_texts(self._options.value('imp-par'))
        for pname, text in modifiers.items():
            if pname in maps:
                print('\t{:20s} modifier: {:15s} value: {:g}'.format(pname, text, maps[pname][0]))
        if len(self._plugins) > 0:
            names  = [name.strip() for name in self._options.value('plug-in').split(PLUGIN_LIST_SEPARATOR)
                      if len(name.strip()) > 0]
            lists  = self._options.value('plug-par').split(PLUGIN_LIST_SEPARATOR)
            for index, name in enumerate(names):
                print('Plug-in: {}'.format(name))
                text = lists[index] if index < len(lists) else ''
                classname = IONIC_PLUGINS[name].__name__
                for pname, modifier in self.__modifier_texts(text, {}).items():
                    full = '{}{}{}'.format(classname, PLUGIN_SEPARATOR, pname)
                    if full in maps:
                        print('\t{:20s} modifier: {:15s} value: {:g}'.format(pname, modifier, maps[full][0]))
        print('', flush=True)

    def __modifier_texts(self, text: str, aliases: dict = None) -> dict:
        """ {gpuSolve parameter name: modifier as written} for an --imp-par or
            --plug-par list
        """
        texts : dict = {}
        for chunk in text.split(','):
            item = chunk.strip().replace(' ', '')
            if len(item) == 0 or item.startswith(IM_PARAM_FLAGS):
                continue
            pname, _modifier = split_param_mod(item, aliases)
            cut = next(position for position, char in enumerate(item) if char in PARAM_MOD_OPERATORS)
            texts[pname] = item[cut:]
        return(texts)

    def __initialize_state(self):
        """ the potential, the model state at rest, and the state file of
            --read-ini-file when one is given
        """
        resting = self._model.get_parameter('V_init')
        if resting is None:
            resting = self._model.get_parameter('vmin')
        V0 = float(np.reshape(np.asarray(resting, dtype=np.float64), (-1,))[0])
        self._U = tf.Variable(np.full((1, 1), V0), dtype=self._dtype, name='U')
        self._Iion = tf.Variable(0.0, dtype=tf.float64, name='Iion')
        self._model.initialize_state_variables(self._U)
        if self._options.given('read-ini-file'):
            self.__read_state_file(self._options.value('read-ini-file'))

    def __sections(self) -> list:
        """ [(section name, entries, prefix)] for the model, then each plugin;
            prefix is how the run names that object's states and parameters
        """
        return(self._svfile.sections())

    def __read_state_file(self, fname: str):
        """ loads the states of a state file; its parameter entries are not
            applied (the run's parameters come from the defaults and the
            modifiers), and each one that differs is reported
        """
        svfile = self._svfile
        svfile.read(fname)
        self._warnings.extend(svfile.mismatches())
        self._model.set_state_variables({name: np.array([value])
                                         for name, value in svfile.states().items()})
        self._U.assign(np.full((1, 1), svfile.Vm()))
        self._notes.append('initial state read from {}'.format(fname))

    def __build_dumps(self):
        """ the state entries written at every output: all of them with -v,
            the -u list otherwise, and the writer that holds them
        """
        options = self._options
        fout = options.value('fout')
        sections = self.__sections()
        self._dumps = []
        if options.value('validate'):
            # bench names the files <fout>_<model or plugin>.<entry>
            for section, entries, prefix in sections:
                for entry in entries:
                    self._dumps.append(('{}_{}.{}'.format(fout, section, entry[0]), prefix) + entry)
        elif len(options.value('imp-sv-dump')) > 0:
            section, entries, prefix = sections[0]
            by_name = {entry[0]: entry for entry in entries}
            base = fout if options.given('fout') else '{}_{}'.format(fout, self._imp_name)
            for name in options.value('imp-sv-dump').split(','):
                name = name.strip()
                if name not in by_name:
                    # bench skips an unknown name without a message
                    raise ValueError('--imp-sv-dump: {} has no entry "{}"; it has {} (the list is '
                                     'comma separated)'.format(section, name, ', '.join(by_name.keys())))
                self._dumps.append(('{}.{}'.format(base, name), prefix) + by_name[name])
        self._writer = SingleCellWriter({'fout': fout, 'fout_given': options.given('fout'),
                                         'binary': options.value('bin'),
                                         'validate': options.value('validate')})
        self._writer.set_dumps([(dump[0], dump[6]) for dump in self._dumps])

    # ---- the compiled stepping loops ------------------------------------------
    def __compile(self):
        """ the two compiled loops: one that collects an output row every
            _out_every steps, one that only advances. Their argument shapes are
            fixed for the run, so each is compiled once.
        """
        self._max_records = max(1, CHUNK_STEPS // self._out_every)
        model   = self._model
        U       = self._U
        Iion    = self._Iion
        dt      = tf.constant(self._dt, dtype=self._dtype)
        charge  = tf.constant(self._stim_charge, dtype=self._dtype)
        every   = self._out_every
        nrec    = self._max_records
        columns = self.__dump_columns()

        def step(stimulated):
            # the stimulus is added as 0 or charge: V + 0 is V exactly, and it
            # avoids a branch in the loop
            U.assign(U + charge * tf.cast(stimulated, self._dtype))
            dU = model.differentiate(U)
            U.assign_add(dt * dU)
            Iion.assign(-tf.cast(tf.reshape(dU, [-1])[0], tf.float64))

        def row():
            values = [tf.cast(tf.reshape(U, [-1])[0], tf.float64), Iion.read_value()]
            values += [column() for column in columns]
            return(tf.stack(values))

        @tf.function(jit_compile=True)
        def advance_recorded(nrows, mask):
            rows = tf.TensorArray(tf.float64, size=nrec, element_shape=[2 + len(columns)])
            for j in tf.range(nrows):
                rows = rows.write(j, row())
                for k in tf.range(every):
                    step(mask[j * every + k])
            return(rows.stack())

        @tf.function(jit_compile=True)
        def advance_plain(nsteps, mask):
            for k in tf.range(nsteps):
                step(mask[k])
            return(tf.constant(0))

        self._advance_recorded = advance_recorded
        self._advance_plain    = advance_plain
        # one row without stepping, for an output that falls just before a
        # state file; rare, so a plain graph is enough
        self._current_row      = tf.function(row)

    def __dump_columns(self) -> list:
        """ one function per dumped entry, giving its value (float64, in the
            reference's unit) inside the compiled loop
        """
        columns : list = []
        for dump in self._dumps:
            _base, prefix, _name, kind, source, scale, _gate = dump
            if kind == 'state':
                variable = self._model.state_variable(prefix + source)
                columns.append(lambda v=variable, s=scale: tf.cast(tf.reshape(v, [-1])[0], tf.float64) * s)
            else:
                value = self._svfile.entry_value(kind, prefix, source) * scale
                columns.append(lambda c=value: tf.constant(c, dtype=tf.float64))
        return(columns)

    def __initial_current(self):
        """ Iion at the initial state, for the first output row (bench computes
            it when the model is initialised). The model can only give it by
            taking a step, so the state is saved, the step taken and the state
            put back.
        """
        saved = self._model.get_state_variables()
        V0 = self._U.numpy()
        dU = self._model.differentiate(self._U)
        self._model.set_state_variables(saved)
        self._U.assign(V0)
        self._Iion.assign(-float(np.reshape(dU.numpy(), (-1,))[0]))

    # ---- running ----------------------------------------------------------------
    def run(self):
        """ run() advances the cell to the end of the run and writes the output """
        self._writer.open()
        start = time.time()
        step = 0
        last = self._last_step
        span = self._max_records * self._out_every
        while step <= last:
            if self._save_step is not None and step == self._save_step:
                self.__write_state_file()
            pending = self._save_step if self._save_step is not None and self._save_step > step else None
            next_out = self.__next_output(step)
            if next_out is None and pending is None:
                # nothing observable is left: the steps up to the end of the
                # run would change no output
                break
            if next_out is None or next_out > step:
                target = min(value for value in (next_out, pending) if value is not None)
                self.__plain(step, target - step, span)
                step = target
                continue
            # at an output step: as many rows as fit in the run and in a chunk,
            # each followed by its _out_every steps, without stepping past a
            # pending state file
            nrows = min((last - step) // self._out_every + 1, self._max_records)
            if pending is not None:
                nrows = min(nrows, (pending - step) // self._out_every)
            if nrows == 0:
                # the state file falls inside this output interval
                rows = self._current_row().numpy()[np.newaxis, :]
                self.__write_rows(step, rows)
                self.__plain(step, pending - step, span)
                step = pending
                continue
            mask = self.__mask(step, span)
            rows = self._advance_recorded(tf.constant(nrows, dtype=tf.int32), mask).numpy()[:nrows]
            self.__write_rows(step, rows)
            step += nrows * self._out_every
        self._writer.close()
        self._elapsed = time.time() - start

    def __write_rows(self, step: int, rows: np.ndarray):
        """ hands rows (one per output, the first at step) to the writer """
        times = (step + self._out_every * np.arange(rows.shape[0])) * self._dt
        self._writer.write_block(times, rows[:, 0], rows[:, 1],
                                 rows[:, 2:] if len(self._dumps) > 0 else None)

    def __next_output(self, step: int) -> int:
        """ the first output step at or after step; None when none is left """
        if step <= self._out_first:
            candidate = self._out_first
        else:
            offset = (step - self._out_first + self._out_every - 1) // self._out_every
            candidate = self._out_first + offset * self._out_every
        return(candidate if candidate <= self._last_step else None)

    def __plain(self, step: int, nsteps: int, span: int):
        """ advances nsteps steps from step without output, in chunks """
        while nsteps > 0:
            count = min(nsteps, span)
            self._advance_plain(tf.constant(count, dtype=tf.int32), self.__mask(step, span))
            step += count
            nsteps -= count

    def __mask(self, step: int, span: int) -> tf.Tensor:
        """ the stimulus switch of steps step .. step + span - 1 (False past
            the end of the run), as a fixed-length tensor
        """
        mask = np.zeros(span, dtype=bool)
        piece = self._stim_mask[step:step + span]
        mask[:piece.shape[0]] = piece
        return(tf.constant(mask))

    def __write_state_file(self):
        """ --save-ini-file at --save-ini-time: the full layout of the
            reference, with the parameters the run uses
        """
        states = self._model.get_state_variables()
        sections : list = []
        for section, entries, prefix in self.__sections():
            values : list = []
            for name, kind, source, scale, _gate in entries:
                if kind == 'state':
                    values.append((name, float(states[prefix + source][0]) * scale))
                else:
                    values.append((name, self._svfile.entry_value(kind, prefix, source) * scale))
            sections.append((section, values))
        known = {'Vm': float(np.reshape(self._U.numpy(), (-1,))[0]), 'Iion': float(self._Iion.numpy())}
        writer = SvFileWriter({'fname': self._options.value('save-ini-file')})
        writer.write([(name, known.get(name)) for name in IMP_DATA_NAMES], sections)
        print('state saved to {} at t = {:g} ms'.format(writer.fname(), self._save_step * self._dt),
              flush=True)


def list_models() -> str:
    """ list_models() returns the --list-imps text, laid out as bench's """
    lines = ['Ionic models: ']
    lines += ['\t{}'.format(name) for name in sorted(IONIC_MODELS.keys(), key=str.lower)]
    lines.append('Plug-ins: ')
    lines += ['\t{}'.format(name) for name in sorted(IONIC_PLUGINS.keys(), key=str.lower)]
    return('\n'.join(lines))


def plugin_outputs() -> str:
    """ plugin_outputs() returns the --plugin-outputs text: the plugins and the
        quantities they export (none, in gpuSolve)
    """
    return('\n'.join('\t{}\t'.format(name) for name in sorted(IONIC_PLUGINS.keys(), key=str.lower)))


def imp_info(imp: str, plug_in: str) -> str:
    """ imp_info(imp, plug_in) returns the --imp-info text: for the model and
        each plugin, the tunable parameters with their defaults and the state
        entries, under the names -u and the state files use
    """
    store = {'num_imp_regions': '1', 'imp_region[0].im': imp, 'imp_region[0].plugins': plug_in}
    mapper = ParameterMapper()
    mapper.resolve(store)
    cell = mapper.ionic_model_class()(dt=0.01, **mapper.ionic_model_options())
    objects = [(imp, cell)]
    names = [name.strip() for name in plug_in.split(PLUGIN_LIST_SEPARATOR) if len(name.strip()) > 0]
    for pluginclass, name in zip(mapper.ionic_plugin_classes(), names):
        objects.append((name, pluginclass(dt=0.01)))
    lines : list = []
    for name, model in objects:
        lines.append('Name: {} ({})'.format(name, type(model).__name__))
        lines.append('\tParameters:')
        for pname in model.tunable_parameter_names():
            value = np.reshape(np.asarray(model.get_parameter(pname), dtype=np.float64), (-1,))[0]
            lines.append('\t{:>32s}\t{:g}'.format(pname, value))
        if name in CELL_TYPE_FLAGS:
            lines.append('\t{:>32s}\t{}'.format('flags', '|'.join(CELL_TYPE_FLAGS[name])))
        lines.append('\tState variables:')
        _section, entries = sv_layout(model, name)
        for entry in entries:
            if entry[1] == 'state':
                lines.append('\t\t{:>20s}'.format(entry[0]))
        lines.append('')
    return('\n'.join(lines))

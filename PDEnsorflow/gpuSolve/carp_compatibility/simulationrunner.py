#!/usr/bin/env python
"""
    SimulationRunner: builds the solver described by a resolved parameter set,
    owns the time loop and writes the results.

    This is the eight-step sequence the example scripts under Tests/FEM all
    perform by hand (load, materials, assemble, initial condition, stimuli,
    finalize, step, write), with the constants coming from a ParameterMapper
    instead of being written into the script.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import os
import time

import numpy as np

from gpuSolve.physics import HeatSolver
from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import mass_beta_weight
from gpuSolve.physics import Prepacer
from gpuSolve.physics import LatDetector
from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.IO.readers import VtxReader
from gpuSolve.IO.readers import StateReader
from gpuSolve.IO.writers import IGBWriter
from gpuSolve.IO.writers import StateWriter
from gpuSolve.IO.readers.roereader import ROE_MAGIC
from gpuSolve.IO.writers.roewriter import ROE_EXTENSION
from gpuSolve.IO.writers.roewriter import RoeWriter
from gpuSolve.IO.writers.roewriter import checkpoint_file_name
from gpuSolve.IO.writers.roewriter import tsav_file_name
from gpuSolve.carp_compatibility.parametermapper import IONIC_PLUGINS
from gpuSolve.carp_compatibility.roestatefile import ROE_PARAMETER_RTOL
from gpuSolve.carp_compatibility.roestatefile import RoeStateFile
from gpuSolve.carp_compatibility.svstatefile import SvStateFile


# Extension of gpuSolve's own checkpoint files (state_format = pkl). The
# reference's binary state files end in ROE_EXTENSION; start_statef tells the
# two apart by their first bytes.
STATE_FILE_EXTENSION : str = '.pkl'

# Extension of a mesh stored as one binary file (Triangulation.saveMesh); a
# mesh without it is read from the three text files <meshname>.pts/.elem/.lon.
MESH_BINARY_EXTENSION : str = '.pkl'


class SimulationRunner:
    """
    class SimulationRunner: runs the simulation described by a ParameterMapper.
    build() must be called before run().
    """

    def __init__(self, config: dict = None):
        self._mapper                 = None
        self._model                  = None
        self._ionic                  = None
        self._writer : IGBWriter     = None
        self._outdir : str           = None
        self._nframes : int          = 0
        self._elapsed : float        = 0.0
        self._verbose : bool         = True
        self._first_step : int       = 0
        self._state_writer : StateWriter = None
        self._pending_saves : list   = None
        self._savestate : dict       = None
        self._chkpt_steps : list     = None
        self._state_format : str     = 'pkl'
        self._roe_writer : RoeStateFile = None
        self._saved_names : set      = None
        self._start_kind : str       = ''
        self._start_path : str       = ''
        self._roe_start : RoeStateFile = None
        self._lats : list            = None
        self._element_maps : dict    = None
        self._beta_ref : float       = None

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    # ---- accessors ----------------------------------------------------------
    def mapper(self):
        """ mapper() returns the ParameterMapper the run is built from """
        return(self._mapper)

    def set_mapper(self, mapper):
        """ set_mapper(mapper) assigns the resolved ParameterMapper to build from """
        self._mapper = mapper

    def model(self):
        """ model() returns the solver that was built; None before build() """
        return(self._model)

    def outdir(self) -> str:
        """ outdir() returns the output directory of the run """
        return(self._outdir)

    def lat_detectors(self) -> list:
        """ lat_detectors() returns the LatDetector list the parameters asked
            for; empty when no num_LATs / lats[] entry is actuated, None before
            build()
        """
        return(self._lats)

    def elapsed(self) -> float:
        """ elapsed() returns the wall-clock duration of the time loop in seconds """
        return(self._elapsed)

    def nframes(self) -> int:
        """ nframes() returns the number of frames written to the output file """
        return(self._nframes)

    # ---- setup --------------------------------------------------------------
    def build(self):
        """ build() creates the solver, assigns the materials and the stimuli,
            assembles the matrices and opens the output file
        """
        try:
            # The kernels run as traced functions and the cell model's step is
            # compiled with XLA, so eager execution is deliberately not forced
            # here: forcing it runs every kernel operation by operation, and
            # made a step of the Tomek model 27 times slower on 63001 nodes.
            self.__build_solver()
            # before anything reads the nodal region IDs: the rule and the
            # claimed tags change what those IDs are
            self.__configure_region_rule()
            # the start state is read now: a binary state file carries cell
            # parameters, which must reach the model with the others
            self.__read_start_state()
            self.__assign_materials()
            self.__assign_cell_parameters()
            self.__apply_stored_parameters()
            # after the materials, because their values are what the cache key
            # has to agree on, and before the assembly, which consumes it
            self.__configure_matrix_cache()
            self._model.assemble_matrices()
            self.__configure_linear_solver()
            self.__set_initial_condition()
            self.__init_region_states()
            self.__add_stimuli()
            # prepacing before the checkpoint is restored, and skipped when one
            # is: a checkpoint already carries a conditioned state, and pacing
            # into it and then overwriting it would only cost time
            self.__prepace()
            self.__restore_state()
            self.__open_output()
            self.__add_lat_detectors()
            self.__schedule_saves()
            self._model.finalize_for_run()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def run(self):
        """ run() advances the solution to Tend, recording a frame every
            dt_per_plot steps and saving the states the parameters ask for,
            and closes the output file. A restarted run continues from the step
            the checkpoint was saved at.
        """
        try:
            settings = self._mapper.output_settings()
            timedt   = settings['timedt']
            model    = self._model
            self._writer.imshow(model.U())
            frames   = 1
            # ctime starts from the solver's own clock (0 for a fresh run, the
            # saved time on a restart) and is still advanced by accumulating dt.
            # A checkpoint stores the accumulated value itself, so a restarted
            # run sees exactly the same sequence of times as an uninterrupted one.
            ctime    = model.ctime()
            report   = timedt * (np.floor(ctime / timedt) + 1.0)
            self.__save_due_states(self._first_step)
            then     = time.time()
            # the step counter is global, so frames are recorded on the same
            # steps as in an uninterrupted run. Step istep advances the solution
            # from istep*dt to (istep+1)*dt, so the frame test is on istep+1:
            # frames then fall at t = k*spacedt (0, spacedt, ..., tend), the
            # times the reference writes. Testing istep put every frame one
            # step late (dt, spacedt + dt, ...) and dropped the one at tend.
            for istep in range(self._first_step, model.nt()):
                ctime += model.dt()
                model.step(ctime)
                if len(self._lats) > 0:
                    # the potential is fetched once and shared: every detector
                    # sees the same step, and the fetch is skipped entirely
                    # when no activation monitoring was asked for. What
                    # check() returns is deliberately not read: the file at the
                    # end of the run is what the parameters asked for
                    potential = model.U()
                    for detector in self._lats:
                        detector.check(potential, ctime)
                if (istep + 1) % model.dt_per_plot() == 0:
                    self._writer.imshow(model.U())
                    frames += 1
                self.__save_due_states(istep + 1)
                if self._verbose and ctime >= report:
                    print('  t = {:9.3f} ms  ({:5.1f}%)'.format(
                        ctime, 100.0 * ctime / model.Tend()), flush=True)
                    report += timedt
            self._elapsed = time.time() - then
            self._nframes = frames
            if self._verbose:
                model.solver().summary()
                print('solution, elapsed: {:f} sec'.format(self._elapsed), flush=True)
            self.__write_lats()
            self._writer.wait()
            self._writer = None
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    # ---- internals ----------------------------------------------------------
    def __check_mesh(self, meshname: str):
        """ reports a missing mesh by name before the solver tries to read it.
            The reader raises a bare FileNotFoundError that names neither the
            file nor the parameter it came from, and `meshname` has a default,
            so the commonest first mistake would otherwise be the least legible.
        """
        expected = self.__mesh_files(meshname)
        missing = [path for path in expected if not os.path.isfile(path)]
        if len(missing) > 0:
            raise ValueError('meshname = "{}": cannot find {}. The mesh is read from '
                             '<meshname>.pts/.elem/.lon relative to the working directory, '
                             'or from a single .pkl file'.format(
                                 meshname, ', '.join(missing)))

    def __build_solver(self):
        """ instantiates the cell model and the solver; the mesh is read by the
            solver constructor from the config
        """
        config     = self._mapper.solver_config()
        self.__check_mesh(config['mesh_file_name'])
        modelclass = self._mapper.ionic_model_class()
        if modelclass is None:
            # no cell model named: this is a pure diffusion (heat) problem
            self._ionic = None
            self._model = HeatSolver(config)
        else:
            self._ionic = self.__new_ionic_model(config['dt'])
            self._model = MonodomainSolver(self._ionic, config)

    def __new_ionic_model(self, dt: float):
        """ builds a cell model of the run's class, with its plugins, at time
            step dt. It is a method rather than three lines inside
            __build_solver() because the prepacer needs a second, independent
            instance of exactly the same model: it gives its cells their
            parameters before initialising them, which a model that folds the
            parameters into a lookup table cannot be asked to do twice.
        """
        modelclass = self._mapper.ionic_model_class()
        # the options (a cell type selected by im_param flags) are constructor
        # arguments: the type fixes the parameter defaults the im_param
        # modifiers are then resolved against
        model   = modelclass(dt=dt, **self._mapper.ionic_model_options())
        plugins = self._mapper.ionic_plugin_classes()
        if len(plugins) > 0:
            # the plugins wrap the model; a run without plugins keeps the
            # bare model, so its numerics and checkpoints are unchanged
            wrapper = IonicModelWithPlugins(dt=dt)
            wrapper.set_model(model)
            for pluginclass in plugins:
                wrapper.add_plugin(pluginclass(dt=dt))
            model = wrapper
        return(model)

    def __init_region_states(self):
        """ imp_region[].im_sv_init: the state held in a single-cell state file,
            copied onto every node of the ionic region that names the file. It
            is what the reference does when it sets its cell models up
            ("read in single cell state vector and spread it out over the entire
            region", physics/ionics.cc), and in the same place in the sequence:
            after the models are tuned, before prepacing (which then paces the
            cells from this state) and before a checkpoint is restored (which
            replaces the whole state anyway).
        """
        if self._ionic is None:
            return
        entries = self._mapper.state_init_files(self.__mesh_tags())
        if len(entries) == 0:
            return
        # the nodal arrays are read and written through the checkpoint, as the
        # prepacer does: a checkpoint is in the user's node order whether or not
        # the solver has been finalized, which is the order the region ids are in
        checkpoint = self._model.checkpoint()
        npt    = int(checkpoint['num_nodes'])
        Vm     = np.reshape(np.array(checkpoint['Vm'], dtype=np.float64), (-1,))
        states = {name: np.reshape(np.array(values, dtype=np.float64), (-1,))
                  for name, values in checkpoint['state_variables'].items()}
        region_of_node = np.reshape(np.asarray(self._model.domain().point_region_ids()), (-1,))
        for entry in entries:
            # entry['tags'] are node region IDs, not element tags: the mapper
            # has already resolved them through the nodal map, so imp_region[0]
            # carries the ID of the nodes no other region claims
            nodes = np.where(np.isin(region_of_node, entry['tags']))[0]
            if nodes.size == 0:
                # the tags exist in the element list, but every node they touch
                # was won by another region (a node takes the highest tag among
                # those the imp_regions claim, see __configure_region_rule)
                self._mapper.add_note('imp_region[{}].im_sv_init = "{}" is ignored: no node '
                                      'belongs to the region'.format(entry['index'], entry['file']))
                continue
            svfile = SvStateFile({'model': self._ionic,
                                  'imp_name': self._mapper.ionic_model_name(),
                                  'plugins': self.__region_plugins(entry['plugins']),
                                  'nodes': nodes})
            svfile.read(entry['file'])
            Vm[nodes] = svfile.Vm()
            for name, value in svfile.states().items():
                if states[name].size != npt:
                    raise ValueError('imp_region[{}].im_sv_init: the state variable {} holds {} '
                                     'value(s) for {} nodes, so it cannot be set region by '
                                     'region'.format(entry['index'], name, states[name].size, npt))
                states[name][nodes] = value
            for message in svfile.mismatches():
                self._mapper.add_note('imp_region[{}]: {}'.format(entry['index'], message))
            if self._verbose:
                print('imp_region[{}]: initial state of {} node(s) read from {}'.format(
                    entry['index'], nodes.size, entry['file']), flush=True)
        self._model.restore_checkpoint({'ionic_model': self._model.checkpoint_model_name(),
                                        'time': checkpoint['time'],
                                        'num_nodes': npt,
                                        'Vm': Vm,
                                        'state_variables': states})

    def __region_plugins(self, names: list) -> list:
        """ __region_plugins(names) returns [(plugin object, name)] for the
            plugins one region lists, in that order. The run's model holds every
            region's plugins together, while a region's state file holds only
            the sections of its own.
        """
        if not isinstance(self._ionic, IonicModelWithPlugins):
            return([])
        attached = {type(plugin).__name__: plugin for plugin in self._ionic.plugins()}
        return([(attached[IONIC_PLUGINS[name].__name__], name) for name in names])

    def __prepace(self):
        """ paces single cells and distributes their state over the mesh, when
            the prepacing parameters ask for it. It runs after the initial
            condition (the cell model must hold its state variables) and before
            finalize_for_run() (it works in the user's node order).
        """
        settings = self._mapper.prepacing_settings()
        prepacer = Prepacer({key: value for key, value in settings.items()
                             if key != 'lats_file'})
        if self._ionic is None or not prepacer.enabled():
            return
        if len(settings['lats_file']) == 0:
            # the mapper has already noted why; prepacing without activation
            # times has no meaning, so it is simply not done
            return
        if len(self._mapper.savestate_settings()['start_statef']) > 0:
            self._mapper.add_note('prepacing_bcl = {} is set and so is start_statef: the '
                                  'checkpoint already carries a conditioned state, so no '
                                  'prepacing is done'.format(settings['bcl']))
            return
        npt = self._model.domain().Pts().shape[0]
        prepacer.read_lats(settings['lats_file'], npt)
        prepacer.set_model_factory(lambda: self.__new_ionic_model(settings['dt']))
        prepacer.prepace(self._model)

    def __configure_region_rule(self):
        """ tells the mesh how to give a node one region ID when the elements
            it belongs to carry several.
            A parameter-file run asks for the reference assignment, the highest
            tag at the node among those an imp_region claims, so that the cell
            model a node runs is the one the reference would give it. A node no
            region claims keeps the no-region sentinel and takes imp_region[0],
            as it does there.
            The array holds a TAG (or the sentinel), not the index of the region
            that owns it; ParameterMapper.tag_to_entry() is what turns one into
            the other, and everything that needs the region must go through it.
            One consumer deliberately does not: the prepacer partitions nodes by
            the raw value, so a region claiming two tags is paced as two cells
            rather than one. That is a finer partition of the same regions, and
            the cell parameters are equal inside each part, so the state it
            distributes is the same; it just does more work than the reference.
        """
        domain = self._model.domain()
        domain.set_regionIDrule('carp_compatibility')
        domain.set_regionIDtags(self._mapper.nodal_region_tags())

    def __mesh_tags(self) -> set:
        """ the set of element tags that actually occur in the mesh """
        tags : set = set()
        for _elemtype, elements in self._model.domain().Elems().items():
            if elements is not None and elements.shape[0] > 0:
                tags |= set(int(tag) for tag in np.unique(elements[:, -1]))
        return(tags)

    def __assign_materials(self):
        """ registers the conductivities and beta, and the two material
            functions the assembler calls.
            beta is registered twice over: as the per-region map, and as the
            single reference value beta_ref that the conductivity is divided by.
            The equation may be divided by a constant but not by a function of
            space, so only the reference leaves the membrane side; the rest of
            beta stays there, as the coefficient beta/beta_ref of the mass
            integral (see gpuSolve.physics.diffusion_tensor). Where beta is
            uniform the coefficient is exactly 1 and the two matrices are the
            ones every earlier version assembled.
        """
        maps = self._mapper.element_property_maps(self.__mesh_tags())
        # kept, because they are also the part of the matrix cache key that
        # describes the physics, and recomputing them there could drift
        self._element_maps = maps
        self._beta_ref     = self._mapper.reference_beta(maps['beta'])
        for pname in ('sigma_l', 'sigma_t', 'beta'):
            self._model.add_element_material_property(pname, 'region', maps[pname])
        self._model.add_element_material_property('beta_ref', 'uniform', self._beta_ref)
        self._model.add_material_function('mass', mass_beta_weight)
        self._model.add_material_function('stiffness', conductivity_tensor)

    def __configure_matrix_cache(self):
        """ names the matrix cache file and builds the key that a cached file
            must carry to be used for this run.
            The key holds everything the mass and stiffness matrices depend on,
            and nothing else. It is built from the conductivity maps that the
            mapper produced rather than from the parameter keys they came from,
            so that no contributor to them can be left out of the key and so
            that the key survives a change in how those parameters are spelled.
            dt, theta, the stimuli, the cell model and the duration are
            deliberately absent: they do not enter the two matrices, and a key
            that held dt would reject the cache in exactly the study, a
            convergence study, that varies it.
        """
        fname = self._mapper.matrix_cache_file()
        if len(fname) == 0:
            return
        cache = self._model.matrix_cache()
        mesh  = {path: cache.file_fingerprint(path)
                 for path in self.__mesh_files(self._mapper.value('meshname'))}
        key   = {'mesh': mesh,
                 'sigma_l': self._element_maps['sigma_l'],
                 'sigma_t': self._element_maps['sigma_t'],
                 'beta': self._element_maps['beta'],
                 # the mass matrix now carries beta/beta_ref, so the reference
                 # is part of what the two matrices depend on
                 'beta_ref': self._beta_ref,
                 'renumbering': int(self._mapper.value('renumbering') != 0),
                 'gpuSolve_version': cache.version_string()}
        self._model.set_matrix_cache(fname, key)

    def __mesh_files(self, meshname: str) -> list:
        """ the files the mesh is read from: the single binary file, or the
            three text files of the other format
        """
        if meshname.endswith(MESH_BINARY_EXTENSION):
            return([meshname])
        return(['{}{}'.format(meshname, suffix) for suffix in ('.pts', '.elem', '.lon')])

    def __assign_cell_parameters(self):
        """ pushes the cell parameters named by im_param into the cell model,
            and those named by plug_param, with the per-region switches, into
            its plugins. This must happen before the first differentiate() call,
            because the rescaling runs inside a tf.function that captures the
            parameters when it is first traced, and before the initial
            condition, from which the plugins compute their initial state.
        """
        if self._ionic is None:
            return
        maps = self._mapper.ionic_parameter_maps(self._ionic, self.__mesh_tags())
        if isinstance(self._ionic, IonicModelWithPlugins):
            maps.update(self._mapper.plugin_parameter_maps(self._ionic, self.__mesh_tags()))
        for pname, pmap in maps.items():
            self._model.add_nodal_material_property(pname, 'region', pmap)
        self._model.assign_nodal_properties()

    def __configure_linear_solver(self):
        """ arms the stopping tests of the conjugate-gradient solver """
        settings = self._mapper.solver_settings()
        solver   = self._model.solver()
        solver.set_toll(settings['toll'])
        solver.set_toll_rel(settings['toll_rel'])
        solver.set_maxiter(settings['maxiter'])

    def __set_initial_condition(self):
        """ starts every node at the resting potential of the cell model, which
            may itself be a per-node column when a region retuned it
        """
        npt = self._model.domain().Pts().shape[0]
        if self._ionic is None:
            self._model.set_initial_condition()
            return
        resting = self._ionic.get_parameter('V_init')
        if resting is None:
            resting = self._ionic.get_parameter('vmin')
        values = np.array(resting)
        if values.ndim == 0:
            U0 = np.full(shape=(npt, 1), fill_value=float(values), dtype=np.float32)
        else:
            U0 = values.reshape(npt, 1).astype(np.float32)
        self._model.set_initial_condition(U0)

    def __read_start_state(self):
        """ reads the state named by start_statef, if any, before the cell
            parameters are assigned: a binary state file carries the parameter
            fields of the reference's state structures, which the reference
            restores along with the state, so they have to reach the model with
            the other parameters (see __apply_stored_parameters). The kind of
            file is told by its first bytes, and its name must agree.
        """
        self._savestate  = self._mapper.savestate_settings()
        self._start_kind = ''
        self._start_path = ''
        self._roe_start  = None
        fname = self._savestate['start_statef']
        if len(fname) == 0:
            return
        path = self.__find_state_file(fname)
        with open(path, 'rb') as fstate:
            opening = fstate.read(4 + len(ROE_MAGIC))
        is_roe = opening[4:] == ROE_MAGIC.encode('ascii')
        if is_roe and path.endswith(STATE_FILE_EXTENSION):
            raise ValueError('start_statef = "{}": {} is a binary state file (.roe), not a {} '
                             'checkpoint'.format(fname, path, STATE_FILE_EXTENSION))
        if not is_roe and path.endswith(ROE_EXTENSION):
            raise ValueError('start_statef = "{}": {} is not a binary state file, although it '
                             'is named {}'.format(fname, path, ROE_EXTENSION))
        self._start_path = path
        if not is_roe:
            self._start_kind = 'pkl'
            return
        if self._ionic is None:
            raise ValueError('start_statef = "{}": a binary state file holds the state of cell '
                             'models, and this run has none'.format(fname))
        self._start_kind = 'roe'
        self._roe_start  = self.__roe_state_file(1)
        self._roe_start.read(path)
        for message in self._roe_start.mismatches():
            self.__warn(message)

    def __apply_stored_parameters(self):
        """ gives the cells the parameters stored in the binary state file,
            where they differ from the parameter file's. The reference copies
            each record back whole, parameter fields included, so a restarted
            run of the reference uses the stored values; doing the same keeps
            the two restarts the same physics. Every difference is printed as a
            warning: a parameter-file change that does not take effect on a
            restart is a trap, and it must not be a silent one.
        """
        if self._roe_start is None:
            return
        changed = 0
        for pname, (nodes, stored) in self._roe_start.stored_parameters().items():
            current = self._model.cell_parameter(pname)
            if current is None:
                raise ValueError('start_statef: the state file stores {}, which this model does '
                                 'not have'.format(pname))
            differ = np.abs(stored - current[nodes]) > ROE_PARAMETER_RTOL * np.maximum(
                np.abs(stored), np.abs(current[nodes]))
            if not np.any(differ):
                continue
            first = int(np.flatnonzero(differ)[0])
            self.__warn('start_statef = "{}": the state stores {} = {:g} where the parameters '
                        'give {:g} ({} node(s) differ, the first shown); the stored values are '
                        'used, as the reference does'.format(self._start_path, pname,
                                                             stored[first], current[nodes][first],
                                                             int(np.count_nonzero(differ))))
            values = np.array(current, dtype=np.float64)
            values[nodes] = stored
            self._model.add_nodal_material_property(pname, 'nodal', values)
            changed += 1
        if changed > 0:
            self._model.assign_nodal_properties()

    def __restore_state(self):
        """ loads the state read by __read_start_state, if any, into the solver
            and sets the step the run resumes from. It runs before
            finalize_for_run(), so the restored values are renumbered together
            with the rest of the solver.
            The time of a binary state file is a float32: it is snapped to the
            nearest step, which is where the reference's own clock resumes.
        """
        self._first_step = 0
        if len(self._start_kind) == 0:
            return
        fname = self._savestate['start_statef']
        model = self._model
        if self._start_kind == 'pkl':
            checkpoint = StateReader().read(self._start_path)
            saved      = checkpoint['time']
        else:
            checkpoint = self.__roe_checkpoint()
            saved      = checkpoint['time']
        dt    = model.dt()
        # lround(t / dt), as the reference turns a time into a step
        first = int(np.floor(saved / dt + 0.5))
        if self._start_kind == 'roe':
            checkpoint['time'] = first * dt
        # restored first, so that a state of another mesh or model is reported
        # as such rather than by its time
        model.restore_checkpoint(checkpoint)
        if first >= model.nt():
            raise ValueError('start_statef = "{}": the state was saved at t = {} ms, which leaves '
                             'no step before tend = {} ms'.format(fname, saved, model.Tend()))
        # a float32 time carries a relative error of 6e-8, so the test allows
        # for it on top of a thousandth of a step
        if abs(first * dt - saved) > max(1.0e-3 * dt, 4.0 * float(np.finfo(np.float32).eps) * saved):
            self._mapper.add_note('start_statef = "{}": the saved time {} ms is not a multiple of '
                                  'dt = {} ms; was it written with another dt?'.format(
                                      fname, saved, dt))
        self._first_step = first
        if self._verbose:
            print('restarting from {} at t = {} ms (step {})'.format(
                self._start_path, checkpoint['time'], first), flush=True)

    def __roe_checkpoint(self) -> dict:
        """ the solver checkpoint with the potential and the state variables of
            the binary state file. A plugin's variables are set on the nodes of
            the regions that list it; elsewhere they keep their initial value,
            as in the reference, where those nodes carry no plugin record.
        """
        checkpoint = self._model.checkpoint()
        states = {name: np.array(values, dtype=np.float64)
                  for name, values in checkpoint['state_variables'].items()}
        for name, (nodes, values) in self._roe_start.states().items():
            if name not in states:
                raise ValueError('start_statef: the state file holds {}, which this model does '
                                 'not have'.format(name))
            states[name][nodes] = values
        return({'ionic_model': self._model.checkpoint_model_name(),
                'time': self._roe_start.time(),
                'num_nodes': checkpoint['num_nodes'],
                'Vm': self._roe_start.Vm(),
                'state_variables': states})

    def __roe_state_file(self, fmt: int) -> RoeStateFile:
        """ a RoeStateFile for this run's regions, writing format fmt """
        nreg = max(1, self._mapper.count('imp_region'))
        return(RoeStateFile({'model': self._ionic,
                             'imp_name': self._mapper.ionic_model_name(),
                             'region_plugins': [self._mapper.region_plugins(index)
                                                for index in range(nreg)],
                             'node_regions': self.__node_regions(),
                             'format': fmt}))

    def __node_regions(self) -> np.ndarray:
        """ the imp_region index of every node, the region mask of the
            reference's binary state file: each node's region ID resolved
            through the same map the cell parameters are
        """
        ids = np.reshape(np.asarray(self._model.domain().point_region_ids()), (-1,))
        assignment = self._mapper.imp_region_assignment(self.__mesh_tags())
        unique, inverse = np.unique(ids, return_inverse=True)
        index = np.array([assignment.get(int(tag), 0) for tag in unique.tolist()], dtype=np.int8)
        return(index[np.reshape(inverse, (-1,))])

    def __warn(self, message: str):
        """ prints a warning now and keeps it for the end-of-run notes """
        print('WARNING: {}'.format(message), flush=True)
        self._mapper.add_note(message)

    def __find_state_file(self, fname: str) -> str:
        """ accepts the state name with or without its extension, as the
            reference does, relative to the working directory as meshname is
        """
        candidates = (fname, '{}{}'.format(fname, ROE_EXTENSION),
                      '{}{}'.format(fname, STATE_FILE_EXTENSION))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return(candidate)
        raise ValueError('start_statef = "{}": cannot find {}'.format(
            fname, ', '.join(candidates)))

    def __schedule_saves(self):
        """ prepares the steps of the tsav saves and of interval checkpointing,
            as the reference's timers do: a time t falls on step lround(t / dt),
            a checkpoint series runs from chkpt_start to chkpt_stop every
            chkpt_intv on those steps, and a save on or after the step a run
            starts from is made, one before it dropped with a note (a restarted
            run saves again at its own start time, as the reference does).
            The format of the files is chosen here; a pure diffusion run has no
            cell model for the binary file to describe, so it keeps gpuSolve's.
        """
        model = self._model
        dt    = model.dt()
        first = self._first_step
        fmt   = self._savestate['state_format']
        if fmt != 'pkl' and self._ionic is None:
            self._mapper.add_note('state_format = {}: a pure diffusion run has no cell model for '
                                  'the binary state file to hold, so its states are saved as '
                                  '{}'.format(fmt, STATE_FILE_EXTENSION))
            fmt = 'pkl'
        self._state_format = fmt
        self._state_writer = StateWriter()
        self._roe_writer   = None
        if fmt != 'pkl':
            self._roe_writer = self.__roe_state_file(1 if fmt == 'roe1' else 3)
        self._saved_names = set()
        self._pending_saves = []
        for tsav, _label in self._savestate['save_times']:
            step = int(np.floor(tsav / dt + 0.5))
            if step < first:
                self._mapper.add_note('tsav = {} ms is before the restart time {} ms: that state '
                                      'is not saved'.format(tsav, model.ctime()))
            else:
                self._pending_saves.append(step)
        self._pending_saves.sort()
        self._chkpt_steps = []
        intv = self._savestate['chkpt_intv']
        if intv > 0.0:
            d_start = int(np.floor(self._savestate['chkpt_start'] / dt + 0.5))
            d_intv  = max(1, int(np.floor(intv / dt + 0.5)))
            d_end   = int(np.floor(self._savestate['chkpt_stop'] / dt + 0.5))
            self._chkpt_steps = [step for step in range(d_start, d_end + 1, d_intv) if step >= first]

    def __save_due_states(self, step: int):
        """ writes every state due at this step: the tsav saves first, then the
            interval checkpoint, the order of the reference's checkpointing()
        """
        due = False
        while len(self._pending_saves) > 0 and self._pending_saves[0] <= step:
            self._pending_saves.pop(0)
            due = True
        if due:
            self.__write_state(self.__tsav_name(step))
        if len(self._chkpt_steps) > 0 and self._chkpt_steps[0] <= step:
            while len(self._chkpt_steps) > 0 and self._chkpt_steps[0] <= step:
                self._chkpt_steps.pop(0)
            self.__write_state(checkpoint_file_name(step * self._model.dt()))

    def __tsav_name(self, step: int) -> str:
        """ the file of a tsav save: write_statef, then the tsav_ext of the save
            time nearest to the step (the reference's get_tsav_ext)
        """
        ctime = step * self._model.dt()
        saves = self._savestate['save_times']
        label = min(saves, key=lambda save: abs(save[0] - ctime))[1]
        return(tsav_file_name(self._savestate['write_statef'], label))

    def __write_state(self, name: str):
        """ writes the current state to <simID>/<name>, name being the
            reference's file name; a gpuSolve checkpoint swaps its extension
        """
        if self._state_format == 'pkl':
            name = name[:-len(ROE_EXTENSION)] + STATE_FILE_EXTENSION
        path = os.path.join(self._outdir, name)
        if name in self._saved_names:
            self._mapper.add_note('{} is written more than once (two save times give it the same '
                                  'name); the last state is kept, as the reference does'.format(path))
        self._saved_names.add(name)
        if self._state_format == 'pkl':
            self._state_writer.write(self._model.checkpoint(), path)
        else:
            RoeWriter().write(self._roe_writer.content(self._model), path)
        if self._verbose:
            print('  state at t = {:9.3f} ms saved to {}'.format(self._model.ctime(), path),
                  flush=True)

    def __add_stimuli(self):
        """ turns each electrode description into the node mask Stimulus expects """
        points = self._model.domain().Pts()
        for props, geometry in self._mapper.stimuli():
            if 'vtx_file' in geometry:
                mask = self.__mask_from_vertex_file(geometry['vtx_file'],
                                                    points.shape[0], props['name'])
            else:
                mask = self.__mask_from_box(geometry['p0'], geometry['p1'],
                                            points, props['name'])
            self._model.add_stimulus(mask, props)

    def __mask_from_vertex_file(self, vtx_file: str, npt: int, name: str) -> np.ndarray:
        """ selects the nodes a `.vtx` file names outright """
        indices = VtxReader().read(vtx_file)
        if indices.size == 0:
            raise ValueError('stimulus "{}": vertex file {} names no node'.format(name, vtx_file))
        outside = indices[np.logical_or(indices < 0, indices >= npt)]
        if outside.size > 0:
            raise ValueError('stimulus "{}": vertex file {} names node {} but the mesh has '
                             '{} nodes (indices are 0-based)'.format(name, vtx_file,
                                                                     int(outside[0]), npt))
        mask = np.zeros(shape=(npt,), dtype=bool)
        mask[indices] = True
        return(mask)

    def __mask_from_box(self, p0: list, p1: list, points: np.ndarray, name: str) -> np.ndarray:
        """ selects the nodes inside the box spanned by the two corner points """
        lower = np.minimum(np.array(p0), np.array(p1))
        upper = np.maximum(np.array(p0), np.array(p1))
        if not np.any(upper > lower):
            raise ValueError('stimulus "{}" has no electrode: set elec.p0 and elec.p1 to '
                             'the corners of the box to stimulate (micrometres), or '
                             'elec.vtx_file to a vertex file'.format(name))
        mask = np.ones(shape=(points.shape[0],), dtype=bool)
        for axis in range(3):
            # an axis where the two corners coincide is left unconstrained,
            # so a box that is flat in z still selects a whole surface mesh
            if upper[axis] > lower[axis]:
                mask = np.logical_and(mask, points[:, axis] >= lower[axis])
                mask = np.logical_and(mask, points[:, axis] <= upper[axis])
        if not np.any(mask):
            raise ValueError('stimulus "{}" selects no node: the box {} to {} lies outside '
                             'the mesh'.format(name, list(lower), list(upper)))
        return(mask)

    def __add_lat_detectors(self):
        """ builds one LatDetector per actuated lats[] entry and seeds it with
            the initial potential, so the first step cannot read a stale
            previous signal and report an activation that never happened.

            The detectors are fed the potential in the mesh numbering
            (MonodomainSolver.U(), not the internal solver order), so the node
            indices they write are the ones the input mesh uses.
        """
        model      = self._model
        self._lats = []
        for config in self._mapper.lat_detectors():
            # dt comes from the solver rather than from the parameters: the
            # detector interpolates the activation instant inside a step, so
            # its dt must be the step actually taken, not the key it was
            # derived from
            config['dt'] = model.dt()
            # one compiled graph per step instead of about fifteen eager
            # operations: it detects the same activations to the bit and halves
            # what the detection costs per step on an idle GPU, measured from
            # 63k to 4M nodes (Tests/DEVTESTS/LAT/lat_detector_scaling.py). A
            # run driven by a parameter file is never short enough for the one
            # trace, built in init() below, to matter.
            config['use_graph_check'] = True
            # the deferred count would fit here too, since the loop ignores
            # what check() returns, and it is deliberately NOT set: it measured
            # slower at every mesh size, because the per-step read-back also
            # paces the eager queue
            detector = LatDetector(config)
            detector.init(model.U())
            self._lats.append(detector)

    def __write_lats(self):
        """ writes the activation file of every detector into the output
            directory, once the time loop has finished
        """
        for detector in self._lats:
            fname = detector.write(self._outdir)
            if self._verbose:
                print('activations written to {}'.format(fname), flush=True)

    def __open_output(self):
        """ creates the output directory and the IGB writer, and exports the
            mesh when gridout_i asks for it
        """
        settings     = self._mapper.output_settings()
        self._outdir = settings['simID']
        os.makedirs(self._outdir, exist_ok=True)
        model       = self._model
        dt_per_plot = model.dt_per_plot()
        # one frame before the loop, then one after every step that reaches a
        # multiple of dt_per_plot, i.e. one per k in (first_step, nt] with
        # k % dt_per_plot == 0 (see run())
        nframes = 1
        if model.nt() > self._first_step:
            nframes += model.nt() // dt_per_plot - self._first_step // dt_per_plot
        igbname = os.path.join(self._outdir, '{}.igb'.format(settings['vofile']))
        if self._first_step > 0 and os.path.isfile(igbname):
            self._mapper.add_note('restart: {} already exists and is overwritten by the '
                                  'segment from t = {} ms; choose another simID to keep '
                                  'it'.format(igbname, model.ctime()))
        self._writer = IGBWriter({'fname': igbname,
                                  'org_t': model.ctime(),
                                  'Tend': model.Tend(),
                                  'nt': nframes,
                                  'nx': model.domain().Pts().shape[0]})
        if settings['gridout_i'] != 0:
            basename = os.path.basename(self._mapper.value('meshname'))
            # a binary mesh is named with its extension (mesh.pkl), a text mesh
            # without one (mesh -> mesh.pts/.elem/.lon); the export always
            # writes the text files, so the extension is dropped or they
            # would come out as mesh.pkl.pts
            if basename.endswith(MESH_BINARY_EXTENSION):
                basename = basename[:-len(MESH_BINARY_EXTENSION)]
            model.domain().exportCarpFormat(os.path.join(self._outdir, basename))

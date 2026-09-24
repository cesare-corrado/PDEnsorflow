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
from gpuSolve.physics import no_mass_property
from gpuSolve.ionic.ionicmodelwithplugins import IonicModelWithPlugins
from gpuSolve.IO.readers import VtxReader
from gpuSolve.IO.readers import StateReader
from gpuSolve.IO.writers import IGBWriter
from gpuSolve.IO.writers import StateWriter
from gpuSolve.carp_compatibility.parametermapper import CHECKPOINT_BASENAME
from gpuSolve.carp_compatibility.parametermapper import time_label


# Extension of the checkpoint files. The format's own state files are binary
# and are not read here, so the extension says which kind a file is.
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
        self._chkpt_index : int      = 0
        self._next_chkpt : float     = None

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    # ---- accessors ----------------------------------------------------------
    def set_mapper(self, mapper):
        """ set_mapper(mapper) assigns the resolved ParameterMapper to build from """
        self._mapper = mapper

    def model(self):
        """ model() returns the solver that was built; None before build() """
        return(self._model)

    def outdir(self) -> str:
        """ outdir() returns the output directory of the run """
        return(self._outdir)

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
            self.__assign_materials()
            self.__assign_cell_parameters()
            self._model.assemble_matrices()
            self.__configure_linear_solver()
            self.__set_initial_condition()
            self.__add_stimuli()
            self.__restore_state()
            self.__open_output()
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
            self.__save_due_states(ctime)
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
                if (istep + 1) % model.dt_per_plot() == 0:
                    self._writer.imshow(model.U())
                    frames += 1
                self.__save_due_states(ctime)
                if self._verbose and ctime >= report:
                    print('  t = {:9.3f} ms  ({:5.1f}%)'.format(
                        ctime, 100.0 * ctime / model.Tend()), flush=True)
                    report += timedt
            self._elapsed = time.time() - then
            self._nframes = frames
            if self._verbose:
                model.solver().summary()
                print('solution, elapsed: {:f} sec'.format(self._elapsed), flush=True)
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
        if meshname.endswith(MESH_BINARY_EXTENSION):
            expected = [meshname]
        else:
            expected = ['{}{}'.format(meshname, suffix)
                        for suffix in ('.pts', '.elem', '.lon')]
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
            # the options (a cell type selected by im_param flags) are
            # constructor arguments: the type fixes the parameter defaults the
            # im_param modifiers are then resolved against
            self._ionic = modelclass(dt=config['dt'], **self._mapper.ionic_model_options())
            plugins     = self._mapper.ionic_plugin_classes()
            if len(plugins) > 0:
                # the plugins wrap the model; a run without plugins keeps the
                # bare model, so its numerics and checkpoints are unchanged
                wrapper = IonicModelWithPlugins(dt=config['dt'])
                wrapper.set_model(self._ionic)
                for pluginclass in plugins:
                    wrapper.add_plugin(pluginclass(dt=config['dt']))
                self._ionic = wrapper
            self._model = MonodomainSolver(self._ionic, config)

    def __mesh_tags(self) -> set:
        """ the set of element tags that actually occur in the mesh """
        tags : set = set()
        for _elemtype, elements in self._model.domain().Elems().items():
            if elements is not None and elements.shape[0] > 0:
                tags |= set(int(tag) for tag in np.unique(elements[:, -1]))
        return(tags)

    def __assign_materials(self):
        """ registers the conductivities and beta, and the two material
            functions the assembler calls
        """
        maps = self._mapper.element_property_maps(self.__mesh_tags())
        for pname in ('sigma_l', 'sigma_t', 'beta'):
            self._model.add_element_material_property(pname, 'region', maps[pname])
        self._model.add_material_function('mass', no_mass_property)
        self._model.add_material_function('stiffness', conductivity_tensor)

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

    def __restore_state(self):
        """ loads the checkpoint named by start_statef, if any, into the solver
            and sets the step the run resumes from. It runs before
            finalize_for_run(), so the restored values are renumbered together
            with the rest of the solver.
        """
        self._savestate  = self._mapper.savestate_settings()
        self._first_step = 0
        fname = self._savestate['start_statef']
        if len(fname) == 0:
            return
        path = self.__find_state_file(fname)
        checkpoint = StateReader().read(path)
        model = self._model
        model.restore_checkpoint(checkpoint)
        dt    = model.dt()
        first = int(round(checkpoint['time'] / dt))
        if first >= model.nt():
            raise ValueError('start_statef = "{}": the state was saved at t = {} ms, which leaves '
                             'no step before tend = {} ms'.format(fname, checkpoint['time'],
                                                                  model.Tend()))
        if abs(first * dt - checkpoint['time']) > 1.0e-3 * dt:
            self._mapper.add_note('start_statef = "{}": the saved time {} ms is not a multiple of '
                                  'dt = {} ms; was it written with another dt?'.format(
                                      fname, checkpoint['time'], dt))
        self._first_step = first
        if self._verbose:
            print('restarting from {} at t = {} ms (step {})'.format(
                path, checkpoint['time'], first), flush=True)

    def __find_state_file(self, fname: str) -> str:
        """ accepts the checkpoint name with or without its extension, relative
            to the working directory as meshname is
        """
        for candidate in (fname, '{}{}'.format(fname, STATE_FILE_EXTENSION)):
            if os.path.isfile(candidate):
                return(candidate)
        raise ValueError('start_statef = "{}": cannot find {} or {}{}'.format(
            fname, fname, fname, STATE_FILE_EXTENSION))

    def __schedule_saves(self):
        """ prepares the save times of tsav and of interval checkpointing.
            A save time is matched to the first step that reaches it within half
            a step, because the times in the parameters need not be multiples of
            dt. Save times already behind the start of a restarted run are
            dropped with a note.
        """
        model = self._model
        half  = 0.5 * model.dt()
        t0    = model.ctime()
        self._state_writer  = StateWriter()
        self._pending_saves = []
        for tsav, name in sorted(self._savestate['save_times']):
            if tsav < t0 - half:
                self._mapper.add_note('tsav = {} ms is before the restart time {} ms: that state '
                                      'is not saved'.format(tsav, t0))
            else:
                self._pending_saves.append((tsav, name))
        self._next_chkpt = None
        intv = self._savestate['chkpt_intv']
        if intv > 0.0:
            start = self._savestate['chkpt_start']
            # the first checkpoint time not behind the start of the run
            self._chkpt_index = max(0, int(np.ceil((t0 - half - start) / intv)))
            self.__advance_checkpoint_time(start + self._chkpt_index * intv)

    def __advance_checkpoint_time(self, candidate: float):
        """ arms the next checkpoint time, or disarms checkpointing past chkpt_stop """
        if candidate <= self._savestate['chkpt_stop'] + 0.5 * self._model.dt():
            self._next_chkpt = candidate
        else:
            self._next_chkpt = None

    def __save_due_states(self, ctime: float):
        """ writes every state whose save time the current step has reached """
        half = 0.5 * self._model.dt()
        while len(self._pending_saves) > 0 and ctime >= self._pending_saves[0][0] - half:
            _tsav, name = self._pending_saves.pop(0)
            self.__write_state(name)
        if self._next_chkpt is not None and ctime >= self._next_chkpt - half:
            self.__write_state('{}.{}'.format(CHECKPOINT_BASENAME, time_label(self._next_chkpt)))
            # a checkpoint interval shorter than dt would name several times
            # inside one step; the state is written once and the rest skipped
            start = self._savestate['chkpt_start']
            intv  = self._savestate['chkpt_intv']
            while self._next_chkpt is not None and ctime >= self._next_chkpt - half:
                self._chkpt_index += 1
                self.__advance_checkpoint_time(start + self._chkpt_index * intv)

    def __write_state(self, name: str):
        """ writes the current state to <simID>/<name>.pkl """
        path = os.path.join(self._outdir, '{}{}'.format(name, STATE_FILE_EXTENSION))
        self._state_writer.write(self._model.checkpoint(), path)
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

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
import tensorflow as tf

from gpuSolve.physics import HeatSolver
from gpuSolve.physics import MonodomainSolver
from gpuSolve.physics import conductivity_tensor
from gpuSolve.physics import no_mass_property
from gpuSolve.IO.writers import IGBWriter


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
            # ConjGrad carries the residual and the search direction in Python
            # attributes from one call of its kernels to the next, so those
            # kernels must not be traced into a graph that outlives the call.
            # Every example script enables eager execution before building a
            # solver for the same reason.
            tf.config.run_functions_eagerly(True)
            self.__build_solver()
            self.__assign_materials()
            self.__assign_cell_parameters()
            self._model.assemble_matrices()
            self.__configure_linear_solver()
            self.__set_initial_condition()
            self.__add_stimuli()
            self.__open_output()
            self._model.finalize_for_run()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def run(self):
        """ run() advances the solution to Tend, recording a frame every
            dt_per_plot steps, and closes the output file
        """
        try:
            settings = self._mapper.output_settings()
            timedt   = settings['timedt']
            model    = self._model
            self._writer.imshow(model.U())
            frames   = 1
            report   = timedt
            ctime    = 0.0
            then     = time.time()
            for istep in range(model.nt()):
                ctime += model.dt()
                model.step(ctime)
                if istep % model.dt_per_plot() == 0:
                    self._writer.imshow(model.U())
                    frames += 1
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
        if meshname.endswith('.pkl'):
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
            self._ionic = modelclass(dt=config['dt'])
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
        """ pushes the cell parameters named by im_param into the cell model.
            This must happen before the first differentiate() call, because the
            rescaling runs inside a tf.function that captures the parameters
            when it is first traced.
        """
        if self._ionic is None:
            return
        maps = self._mapper.ionic_parameter_maps(self._ionic, self.__mesh_tags())
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

    def __add_stimuli(self):
        """ turns each electrode box into the node mask Stimulus expects """
        points = self._model.domain().Pts()
        for props, (p0, p1) in self._mapper.stimuli():
            lower = np.minimum(np.array(p0), np.array(p1))
            upper = np.maximum(np.array(p0), np.array(p1))
            if not np.any(upper > lower):
                raise ValueError('stimulus "{}" has no electrode: set elec.p0 and elec.p1 to '
                                 'the corners of the box to stimulate '
                                 '(micrometres)'.format(props['name']))
            mask = np.ones(shape=(points.shape[0],), dtype=bool)
            for axis in range(3):
                # an axis where the two corners coincide is left unconstrained,
                # so a box that is flat in z still selects a whole surface mesh
                if upper[axis] > lower[axis]:
                    mask = np.logical_and(mask, points[:, axis] >= lower[axis])
                    mask = np.logical_and(mask, points[:, axis] <= upper[axis])
            if not np.any(mask):
                raise ValueError('stimulus "{}" selects no node: the box '
                                 '{} to {} lies outside the mesh'.format(props['name'],
                                                                         list(lower), list(upper)))
            self._model.add_stimulus(mask, props)

    def __open_output(self):
        """ creates the output directory and the IGB writer, and exports the
            mesh when gridout_i asks for it
        """
        settings     = self._mapper.output_settings()
        self._outdir = settings['simID']
        os.makedirs(self._outdir, exist_ok=True)
        model       = self._model
        dt_per_plot = model.dt_per_plot()
        # one frame before the loop, then one every dt_per_plot steps
        nframes = 1 + (model.nt() + dt_per_plot - 1) // dt_per_plot
        self._writer = IGBWriter({'fname': os.path.join(self._outdir,
                                                        '{}.igb'.format(settings['vofile'])),
                                  'Tend': model.Tend(),
                                  'nt': nframes,
                                  'nx': model.domain().Pts().shape[0]})
        if settings['gridout_i'] != 0:
            basename = os.path.basename(self._mapper.value('meshname'))
            model.domain().exportCarpFormat(os.path.join(self._outdir, basename))

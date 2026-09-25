#!/usr/bin/env python
"""
    Prepacer: puts the cell models of a tissue simulation into a preconditioned
    state before the run starts, by pacing single cells and distributing their
    state over the mesh with a per-node time offset taken from a file of local
    activation times.

    This is the equivalent of the reference's prepacing_* parameters
    (prepacing_lats, prepacing_beats, prepacing_bcl, prepacing_stimdur,
    prepacing_stimstr), implemented in its Electrics::prepace().

    Why it is not simply "run the tissue for a few beats"
    -----------------------------------------------------
    A cell started from the published initial conditions is not at a limit
    cycle: its intracellular concentrations drift for tens of beats. Pacing the
    tissue itself to steady state costs a full diffusion solve per step. The
    reference instead paces a *single* cell, which needs no linear solve at all,
    and hands the resulting state to every node.

    The time offset is what makes that more than a constant initial condition.
    Given the activation time LAT[i] that node i will have in the run to come,

        offset  = floor(min(LAT) / bcl) * bcl
        last_tm = bcl * beats
        save[i] = last_tm - (LAT[i] - offset)

    so a node that activates early takes the state of the paced cell late in the
    prepacing train, and a node that activates late takes it earlier. Every cell
    then enters the run at the phase of the cycle consistent with the moment the
    wavefront is about to reach it, instead of all cells sitting at the same
    point of the same beat.

    The protocol of the paced cell is the reference's, including its two
    peculiarities, which are kept deliberately:

      * the stimulus is added straight to the potential as
        Vm += stimstr * dt while fmod(t, bcl) < stimdur. With stimstr in uA/uF
        and dt in ms this is a millivolt increment, i.e. a transmembrane current
        density applied to a unit-capacitance membrane;
      * the last beat is suppressed by the `t < bcl*beats - 1` test, so the cell
        is handed over during the diastolic interval that follows beat `beats`,
        not in the middle of an upstroke.

    Which cells are paced: A and B
    ------------------------------
    The reference paces one cell per ionic region. gpuSolve has no per-region
    model object: it runs one vectorised model whose parameters are per-node
    columns, so "region" lives in the mesh, not in the model. Two strategies
    follow, and they are the same computation, not an approximation of it:

      A (group_by_region = True): pace one representative cell per mesh region.
        Nodes that share a region share a parameter row, so they follow a
        bitwise identical trajectory; computing it once per region rather than
        once per node removes redundancy, it does not lose anything. This is the
        default and matches the reference's cost, one cell for a homogeneous
        mesh.

      B (group_by_region = False): pace every node. This is the same set of ODEs
        with the conductivity switched off, and it is what is needed once a
        cell parameter varies *within* a region, where a single representative
        cell no longer stands for its neighbours. It costs roughly one ionic
        step per node per time step; measured on an RTX A2000 with ten
        Tusscher-Panfilov it is about 8x A at 10^6 nodes, not the orders of
        magnitude a per-node loop would suggest, because the cells are
        independent and the GPU runs them in parallel.

    group_by_region = None (the default) chooses between them: B as soon as any
    cell parameter was registered as a per-node ('nodal') property, A otherwise.
    Nothing in the parameter-file front end produces a 'nodal' property today,
    so a run driven by a parameter file always takes A; only a script that
    builds a per-node map selects B. The reference's adjustment files, which
    set cell parameters node by node, are not implemented yet: when they are,
    they are what will make B fire on its own.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""
import time

import numpy as np
import tensorflow as tf

from gpuSolve.IO.readers.latreader import LatReader


# How far past the end of the run an unactivated node (a negative activation
# time) is pushed, in ms. The reference uses tend + 10: the node then takes the
# smallest save time of all, i.e. the least prepaced state, which is the right
# answer for tissue the wavefront never reaches.
UNACTIVATED_MARGIN : float = 10.0

# Margin of the `t < bcl*beats - 1` test that suppresses the stimulus of the
# last beat, in ms. It is the reference's literal 1 ms and is kept as a named
# constant because it is a protocol choice, not a rounding guard.
LAST_BEAT_MARGIN : float = 1.0


class Prepacer:
    """
    class Prepacer: paces single cells and distributes their state over the mesh
    to precondition a simulation. Configured with a dict, as the solvers are:

        {'bcl': 500.0, 'beats': 20, 'stimdur': 1.0, 'stimstr': 60.0,
         'dt': 0.025, 'tend': 1000.0, 'lats': <array or None>,
         'group_by_region': None}

    set_model_factory() must be given a callable that returns a fresh, not yet
    initialised instance of the same cell model the run uses, and the activation
    times must be set, before prepace() is called.
    """

    def __init__(self, config: dict = None):
        self._bcl : float               = -1.0
        self._beats : int               = 0
        self._stimdur : float           = 1.0
        self._stimstr : float           = 60.0
        self._dt : float                = 0.0
        self._tend : float              = 0.0
        self._lats : np.ndarray         = None
        self._group_by_region : bool    = None
        self._model_factory             = None
        self._verbose : bool            = True
        self._save_times : np.ndarray   = None
        self._ncells : int              = 0
        self._nsteps : int              = 0
        self._per_node : bool           = False
        self._elapsed : float           = 0.0

        if config is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

    # ---- accessors ----------------------------------------------------------
    def bcl(self) -> float:
        """ bcl() returns the basic cycle length of the prepacing train (ms) """
        return(self._bcl)

    def set_bcl(self, bcl: float):
        """ set_bcl(bcl) sets the basic cycle length of the prepacing train (ms);
            prepacing is off when it is not positive
        """
        self._bcl = bcl

    def beats(self) -> int:
        """ beats() returns the number of prepacing beats """
        return(self._beats)

    def set_beats(self, beats: int):
        """ set_beats(beats) sets the number of prepacing beats """
        self._beats = beats

    def stimdur(self) -> float:
        """ stimdur() returns the duration of a prepacing stimulus (ms) """
        return(self._stimdur)

    def set_stimdur(self, stimdur: float):
        """ set_stimdur(stimdur) sets the duration of a prepacing stimulus (ms) """
        self._stimdur = stimdur

    def stimstr(self) -> float:
        """ stimstr() returns the strength of a prepacing stimulus (uA/uF) """
        return(self._stimstr)

    def set_stimstr(self, stimstr: float):
        """ set_stimstr(stimstr) sets the strength of a prepacing stimulus (uA/uF) """
        self._stimstr = stimstr

    def dt(self) -> float:
        """ dt() returns the time step of the prepacing integration (ms) """
        return(self._dt)

    def set_dt(self, dt: float):
        """ set_dt(dt) sets the time step of the prepacing integration (ms) """
        self._dt = dt

    def set_tend(self, tend: float):
        """ set_tend(tend) sets the end time of the run to come (ms). It is used
            only to place the nodes that never activate, as the reference does
        """
        self._tend = tend

    def lats(self) -> np.ndarray:
        """ lats() returns the per-node activation times prepacing is guided by """
        return(self._lats)

    def set_lats(self, lats: np.ndarray):
        """ set_lats(lats) sets the per-node activation times (ms), one per node
            in mesh order; a negative value marks a node that never activates
        """
        self._lats = np.asarray(lats, dtype=np.float64).reshape(-1)

    def read_lats(self, fname: str, npt: int = 0) -> np.ndarray:
        """ read_lats(fname,npt) reads the per-node activation times from the
            file fname, as written by LatDetector with all = 0
        """
        self.set_lats(LatReader().read(fname, npt))
        return(self._lats)

    def group_by_region(self) -> bool:
        """ group_by_region() returns True when one cell per mesh region is
            paced, False when every node is, None while the choice is left to
            prepace() to make from the parameters of the model
        """
        return(self._group_by_region)

    def set_group_by_region(self, flag: bool):
        """ set_group_by_region(flag) forces one cell per region (True) or one
            cell per node (False); None restores the automatic choice
        """
        self._group_by_region = flag

    def set_model_factory(self, factory):
        """ set_model_factory(factory) sets the callable that returns a fresh,
            not yet initialised instance of the run's cell model. A fresh
            instance is built rather than the run's own model reused because
            several models (Tomek, ten Tusscher-Panfilov) build lookup tables
            inside initialize_state_variables() from the parameters they hold at
            that moment, so the paced cells must be given their parameters
            before they are initialised.
        """
        self._model_factory = factory

    def save_times(self) -> np.ndarray:
        """ save_times() returns the per-node time, in the prepacing clock, at
            which each node takes the state of its paced cell; None before
            prepace()
        """
        return(self._save_times)

    def ncells(self) -> int:
        """ ncells() returns how many cells were paced: the number of regions
            for A, the number of nodes for B; 0 before prepace()
        """
        return(self._ncells)

    def nsteps(self) -> int:
        """ nsteps() returns how many time steps the prepacing integrated """
        return(self._nsteps)

    def per_node(self) -> bool:
        """ per_node() returns True when the last prepace() paced one cell per
            node (B), False when it paced one per region (A)
        """
        return(self._per_node)

    def elapsed(self) -> float:
        """ elapsed() returns the wall-clock duration of the last prepace() in
            seconds
        """
        return(self._elapsed)

    def enabled(self) -> bool:
        """ enabled() returns True when the parameters ask for prepacing. The
            reference's switch is the cycle length, so a bcl that is not
            positive disables it whatever the other parameters say
        """
        return(self._bcl > 0.0 and self._beats > 0)

    # ---- the prepacing itself ----------------------------------------------
    def prepace(self, solver):
        """ prepace(solver) paces the cells and writes the resulting state into
            solver. It must be called after the initial condition is set (the
            cell model needs its state variables) and before finalize_for_run()
            (it works in the user's node order, which the renumbering then
            permutes together with everything else).
        """
        try:
            self.__check(solver)
            npt   = int(solver.domain().Pts().shape[0])
            self._save_times = self.__compute_save_times(npt)
            # the state a node takes is the one the paced cell holds after this
            # many steps. The reference walks a cursor over the save times sorted
            # ascending and copies after the step, which leaves the cell that is
            # paced to the end one step short of the others; the nearest step to
            # the save time is used here for every node alike, a difference of at
            # most one dt.
            snap  = np.rint(np.maximum(self._save_times, 0.0) / self._dt).astype(np.int64)
            self._nsteps = int(snap.max())
            reg_ids, reps = self.__cell_map(solver, npt)
            # the potential the paced cells start from is read through the
            # checkpoint rather than through U(): a checkpoint is in the user's
            # node order whether or not the solver has been finalized, which is
            # the order the activation times and the region ids are in
            Vm0 = np.reshape(np.asarray(solver.checkpoint()['Vm']), (-1,))
            self._ncells  = reps.shape[0]
            self._per_node = (self._ncells == npt)
            if self._verbose:
                print('prepacing: {} beats at bcl = {} ms, {} cell(s) over {} steps '
                      '({})'.format(self._beats, self._bcl, self._ncells, self._nsteps,
                                    'one per node' if self._per_node else 'one per region'),
                      flush=True)
            then = time.time()
            if self._per_node:
                Vm, states = self.__pace_per_node(solver, snap, Vm0)
            else:
                Vm, states = self.__pace_by_region(solver, snap, reg_ids, reps, Vm0)
            self._elapsed = time.time() - then
            solver.restore_checkpoint({'ionic_model': solver.checkpoint_model_name(),
                                       'time': 0.0,
                                       'num_nodes': npt,
                                       'Vm': Vm,
                                       'state_variables': states})
            if self._verbose:
                print('prepacing: done in {:.2f} s, Vm in [{:.3f}, {:.3f}] mV'.format(
                    self._elapsed, float(Vm.min()), float(Vm.max())), flush=True)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    # ---- internals ----------------------------------------------------------
    def __check(self, solver):
        """ refuses the configurations that would otherwise fail deep inside the
            integration, or succeed and mean something else
        """
        if not self.enabled():
            raise ValueError('prepacing is not configured: bcl = {} and beats = {}; both must '
                             'be positive'.format(self._bcl, self._beats))
        if self._model_factory is None:
            raise ValueError('prepacing needs a cell model factory; call set_model_factory()')
        if self._dt <= 0.0:
            raise ValueError('prepacing dt = {}: must be positive'.format(self._dt))
        if self._lats is None:
            raise ValueError('prepacing needs the activation times; call set_lats() or '
                             'read_lats()')
        # getattr, not solver.ionic_model(): a pure diffusion solver has no such
        # accessor at all, and the point of the test is to say so in words
        if getattr(solver, 'ionic_model', None) is None or solver.ionic_model() is None:
            raise ValueError('prepacing needs a cell model; a pure diffusion problem has no '
                             'state to precondition')
        if solver.ready_for_run():
            raise ValueError('prepacing must run before finalize_for_run(): it works in the '
                             'user node order, which finalize_for_run() permutes')
        npt = int(solver.domain().Pts().shape[0])
        if self._lats.size != npt:
            raise ValueError('prepacing: {} activation times for a mesh of {} nodes'.format(
                self._lats.size, npt))

    def __compute_save_times(self, npt: int) -> np.ndarray:
        """ the per-node time at which the paced cell's state is taken, in the
            prepacing clock that runs from 0 to bcl*beats
        """
        lats = np.array(self._lats, dtype=np.float64)
        unactivated = lats < 0.0
        if np.all(unactivated):
            raise ValueError('prepacing: every activation time is negative, so no node ever '
                             'activates. The activation-time file does not describe this run')
        # a node the wavefront never reaches is placed past the end of the run,
        # so it takes the least prepaced state of all
        lats[unactivated] = self._tend + UNACTIVATED_MARGIN
        latmin  = float(lats.min())
        # the offset is a whole number of cycles, so the prepacing train stays
        # in phase with the activation sequence however far into the run the
        # earliest activation falls
        offset  = np.floor(latmin / self._bcl) * self._bcl
        last_tm = self._bcl * self._beats
        return(last_tm - (lats - offset))

    def __cell_map(self, solver, npt: int) -> tuple:
        """ returns (reg_ids, reps): the cell index of every node, and the node
            index that represents every cell. B is A with one cell per node, so
            the two strategies differ only in these two arrays.
        """
        per_node = self.__choose_per_node(solver)
        if per_node:
            reps    = np.arange(npt, dtype=np.int64)
            reg_ids = reps
            return((reg_ids, reps))
        region_of_node = np.asarray(solver.domain().point_region_ids()).reshape(-1)
        # np.unique gives both halves of the map in one pass: the first node of
        # each region (the cell actually paced) and, through the inverse index,
        # the cell every node reads its state back from
        _regions, first, reg_ids = np.unique(region_of_node, return_index=True,
                                             return_inverse=True)
        return((np.reshape(reg_ids, (-1,)).astype(np.int64), first.astype(np.int64)))

    def __choose_per_node(self, solver) -> bool:
        """ resolves the A / B choice. An explicit setting wins; otherwise B is
            taken as soon as a cell parameter varies node by node, because a
            single representative cell then no longer stands for its region.
        """
        if self._group_by_region is not None:
            return(not self._group_by_region)
        return(bool(solver.has_nodal_cell_parameters()))

    def __paced_cells(self, solver, reps: np.ndarray, Vm0: np.ndarray):
        """ builds the cell model that is paced: a fresh instance sized to the
            number of cells, holding the parameters of the representative nodes
            and started from their potential
        """
        model = self._model_factory()
        model.set_dt(self._dt)
        live = solver.ionic_model()
        npt  = Vm0.shape[0]
        for pname in solver.nodal_cell_parameters():
            values = live.get_parameter(pname)
            if values is None:
                continue
            values = np.asarray(values)
            # a parameter that did not resolve to one value per node is uniform,
            # and the fresh instance already carries it
            if values.size == npt:
                model.set_parameter(pname, np.reshape(np.reshape(values, (-1,))[reps], (-1, 1)))
        U = tf.Variable(np.reshape(Vm0[reps], (-1, 1)).astype(np.float32), name='prepace_U')
        model.initialize_state_variables(U)
        return((model, U))

    def __stimulus_flags(self) -> np.ndarray:
        """ whether the stimulus is on at each step of the prepacing train.
            It is precomputed on the host because it depends only on the step
            index, so the integration loop carries a lookup instead of a
            modulo and two comparisons per step.
        """
        t      = np.arange(self._nsteps, dtype=np.float64) * self._dt
        during = np.mod(t, self._bcl) < self._stimdur
        # the last beat is not stimulated, so the cells are handed over during
        # the diastolic interval that follows it rather than mid-upstroke
        before = t < self._bcl * self._beats - LAST_BEAT_MARGIN
        return(np.logical_and(during, before))

    def __pace_by_region(self, solver, snap: np.ndarray, reg_ids: np.ndarray,
                         reps: np.ndarray, Vm0: np.ndarray) -> tuple:
        """ strategy A: pace one cell per region, and read every node's state out
            of its region's cell at the node's own save step.

            The integration runs inside a tf.function between save steps, not
            step by step from Python: the loop is hundreds of thousands of steps
            long and the per-step dispatch cost dominates the arithmetic of a
            handful of cells. The states are read back only at the distinct save
            steps, of which there are at most as many as there are distinct
            activation times.
        """
        model, U = self.__paced_cells(solver, reps, Vm0)
        names    = tuple(model.state_variable_names())
        npt      = snap.shape[0]
        flags    = tf.constant(self.__stimulus_flags(), dtype=tf.bool)
        advance  = self.__advance_function(model, U, flags)

        Vm     = np.zeros(shape=(npt,), dtype=np.float64)
        states = {name: np.zeros(shape=(npt,), dtype=np.float64) for name in names}
        done   = 0
        for step in np.unique(snap):
            if step > done:
                advance(tf.constant(done, dtype=tf.int32), tf.constant(int(step) - done,
                                                                       dtype=tf.int32))
                done = int(step)
            due = np.nonzero(snap == step)[0]
            cells = reg_ids[due]
            Vm[due] = np.reshape(U.numpy(), (-1,))[cells]
            for name in names:
                values = np.reshape(model.state_variable(name).numpy(), (-1,))
                states[name][due] = values[cells]
        return((Vm, states))

    def __pace_per_node(self, solver, snap: np.ndarray, Vm0: np.ndarray) -> tuple:
        """ strategy B: pace one cell per node, freezing each cell once the step
            it is handed over at is reached.

            The freeze is a tf.where against the step index rather than a slice
            of the nodes still active: a select keeps every tensor the same
            shape from step to step, which is what lets the loop stay compiled.
            It also has to restore the state variables, not only the potential,
            because differentiate() advances them in place, so the values a
            frozen node must keep are already gone by the time the step returns.
        """
        reps     = np.arange(snap.shape[0], dtype=np.int64)
        model, U = self.__paced_cells(solver, reps, Vm0)
        names    = tuple(model.state_variable_names())
        svs      = [model.state_variable(name) for name in names]
        flags    = tf.constant(self.__stimulus_flags(), dtype=tf.bool)
        active   = tf.constant(np.reshape(snap, (-1, 1)), dtype=tf.int64)
        dt       = tf.constant(self._dt, dtype=U.dtype)
        strength = tf.constant(self._stimstr * self._dt, dtype=U.dtype)

        @tf.function
        def run(nstep):
            for step in tf.range(nstep):
                live = tf.cast(step, tf.int64) < active
                held = [sv.read_value() for sv in svs]
                if flags[step]:
                    U.assign_add(tf.where(live, strength, tf.zeros_like(U)))
                dU = model.differentiate(U)
                U.assign(tf.where(live, U + dt * dU, U))
                for sv, old in zip(svs, held):
                    sv.assign(tf.where(tf.cast(live, sv.dtype) > 0, sv, old))
            return(U)

        run(tf.constant(self._nsteps, dtype=tf.int32))
        Vm     = np.reshape(U.numpy(), (-1,)).astype(np.float64)
        states = {name: np.reshape(model.state_variable(name).numpy(), (-1,)).astype(np.float64)
                  for name in names}
        return((Vm, states))

    def __advance_function(self, model, U: tf.Variable, flags: tf.constant):
        """ the compiled integration of the paced cells over a run of steps.
            The step index is an argument so the stimulus lookup stays inside
            the graph: the function is traced once and called once per distinct
            save time.
        """
        dt       = tf.constant(self._dt, dtype=U.dtype)
        strength = tf.constant(self._stimstr * self._dt, dtype=U.dtype)

        @tf.function
        def advance(first, nstep):
            for step in tf.range(nstep):
                if flags[first + step]:
                    U.assign_add(tf.fill(tf.shape(U), strength))
                U.assign_add(dt * model.differentiate(U))
            return(U)

        return(advance)

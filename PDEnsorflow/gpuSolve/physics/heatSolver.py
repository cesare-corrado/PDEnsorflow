#!/usr/bin/env python
"""
    HeatSolver: one-step FEM heat-equation solver (TensorFlow backend), with the
    theta method in time and the source S taken explicitly, split from the
    diffusion as the reference does it:
        (M + theta dt K) U^{n+1} = (M - (1 - theta) dt K) (U^n + dt S)
    theta = 0.5 (default) Crank-Nicolson, theta = 1 implicit Euler. With
    split_source = False the source is taken unsplit instead:
        (M + theta dt K) U^{n+1} = (M - (1 - theta) dt K) U^n + dt M S

    Organised as load/setup -> assemble -> per-step kernel, so the expensive
    assembly runs once and the time loop only calls a small kernel, around the
    physics implemented in PDEnsorflow/Tests/FEM/HeatEquation/heat.py.

    The class advances the solution by a single time step (`step`). The main
    temporal loop must be driven by the caller.
"""
import numpy as np
import tensorflow as tf

from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.matrices.globalMatrices import compute_reverse_cuthill_mckee_indexing
from gpuSolve.matrices.globalMatrices import csr_axpby
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.linearsolvers.jacobi_precond import JacobiPrecond
from gpuSolve.force_terms import Stimulus


class HeatSolver:
    """ One implicit Euler step of the heat equation:
            (MASS + dt * STIFFNESS) U^{n+1} = MASS (U^n + dt I0)
        The caller owns the time loop and supplies t to step().
    """

    # Default initial value used by set_initial_condition when U0 is None.
    _U_DEFAULT = 0.0

    def __init__(self, cfgdict=None):
        self._mesh_file_name : str   = None
        self._dt : float             = 0.1
        self._dt_per_plot : int      = 2
        self._Tend : float           = 10
        # reverse Cuthill-McKee node renumbering, ON by default: it clusters the
        # non-zeros near the diagonal, so the sparse products of the CG read
        # memory almost in order. On a 3.3 M-node mesh with the mesher's own
        # node order it made the time step 3x faster. It changes the order of
        # the additions only, so results move at round-off (max 1e-3 mV there);
        # the default was changed from False with the maintainer's approval,
        # and use_renumbering = False restores the previous numerics.
        self._use_renumbering : bool = True
        # warm-start strategy for the CG solve: when True the initial
        # guess is the linear extrapolation 2 U^n - U^{n-1} of the two
        # previous solutions, which typically
        # saves CG iterations per step; when False the guess is U^n.
        self._linear_guess : bool    = True
        # weight of the new time level in the diffusion step (theta method):
        #   (M + theta dt K) U^{n+1} = (M - (1 - theta) dt K) U*
        # with U* the potential after the forcing (and, in the monodomain
        # solver, the ionic) update of the splitting. 0.5 is Crank-Nicolson,
        # second order in time and the reference simulator's default
        # (parab_solve = 1, theta = 0.5); 1.0 is implicit Euler, the scheme
        # used before this parameter existed. The default was changed from
        # implicit Euler to 0.5 with the maintainer's approval; theta = 1.0
        # restores the previous numerics bit for bit.
        self._theta : float          = 0.5
        # how the source S (forcing, and in the monodomain solver the ionic
        # current) enters the theta step, for theta < 1:
        #   True  (default, split): (M + theta dt K) U^{n+1} = (M - (1-theta) dt K) (U^n + dt S)
        #   False (unsplit):        (M + theta dt K) U^{n+1} = (M - (1-theta) dt K) U^n + dt M S
        # The split form is the reference simulator's operator splitting: the
        # explicit diffusion also acts on the dt S of the reaction update, a
        # -(1-theta) dt^2 K S splitting error the unsplit form does not have.
        # Which is better depends on the source. One independent of U keeps
        # Crank-Nicolson second order only unsplit (split: first order, 44x
        # the error at dt = 0.2 on a 1D test). A stiff cell model advanced with
        # forward Euler does better split: on the 1D mMS cable, CV -1.30%
        # split against -3.37% unsplit, and the unsplit front overshoots vmax
        # by 11 mV for 6 ms. Split is the default (the maintainer's choice
        # after these measurements, and the reference's scheme); for theta = 1
        # the two coincide.
        self._split_source : bool    = True

        if cfgdict is not None:
            for attribute in self.__dict__.keys():
                if attribute[1:] in cfgdict.keys():
                    setattr(self, attribute, cfgdict[attribute[1:]])

        self._Domain : Triangulation         = Triangulation()
        self._materials : MaterialProperties = MaterialProperties()
        self._Solver : ConjGrad              = ConjGrad()
        self._Precond : JacobiPrecond        = JacobiPrecond()
        # the mass matrix M, and the matrix M - (1 - theta) dt K that multiplies
        # U on the right-hand side (M itself for implicit Euler)
        self._MASS                           = None
        self._RHS_MATRIX                     = None
        self._U : tf.Variable                = None
        self._U_prev : tf.Variable           = None
        self._ready_for_run : bool           = False
        self._ctime : float                  = 0.0
        self._nbstim : int                   = 0
        self._renumbering : dict             = None
        self._StimulusDict : dict            = None
        self._I0_zero : tf.Tensor            = None
        # the number of steps that reach Tend. Floor division of the floats
        # lost a step whenever dt is not exact in binary (100 // 0.02 is 4999,
        # 1 // 0.025 is 39), so every such run stopped one step short of Tend.
        # The nearest whole number is taken instead, and lowered only if it
        # would step past Tend by more than round-off.
        nsteps = int(round(self._Tend / self._dt))
        if nsteps * self._dt > self._Tend * (1.0 + 1.0e-9):
            nsteps -= 1
        self._nt : int                       = nsteps
        # theta = 0 would be explicit Euler, which this solver does not
        # implement (it always solves a linear system), and theta > 1 weights
        # beyond the new level; both are refused rather than run
        if not (0.0 < self._theta <= 1.0):
            raise ValueError('theta = {}: the theta method needs 0 < theta <= 1 '
                             '(0.5 Crank-Nicolson, 1 implicit Euler)'.format(self._theta))

        if self._mesh_file_name is not None:
            self._Domain.readMesh('{}'.format(self._mesh_file_name))

    # ---- setup --------------------------------------------------------------
    def loadMesh(self, fname: str):
        self._mesh_file_name = fname
        self._Domain.readMesh('{}'.format(self._mesh_file_name))

    def add_element_material_property(self, pname: str, ptype: str, prop: dict):
        self._materials.add_element_property(pname, ptype, prop)

    def add_material_function(self, fname: str, fsign):
        self._materials.add_ud_function(fname, fsign)

    def assemble_matrices(self):
        connectivity = self._Domain.mesh_connectivity('True')
        pattern      = compute_coo_pattern(connectivity)
        if self._use_renumbering:
            self._renumbering = compute_reverse_cuthill_mckee_indexing(pattern)
        lmatr        = {'mass': localMass, 'stiffness': localStiffness}
        MATRICES     = assemble_matrices_dict(lmatr, pattern, self._Domain,
                                              self._materials, connectivity,
                                              renumbering=self._renumbering)
        MASS        = MATRICES['mass']
        STIFFNESS   = MATRICES['stiffness']
        A           = csr_axpby(MASS, 1.0, STIFFNESS, self._theta * self._dt)
        if self._theta == 1.0:
            # implicit Euler: the right-hand side is M itself, not M - 0 K,
            # so the previous scheme is reproduced to the bit
            self._RHS_MATRIX = MASS
        else:
            # built once; the split form needs only this matrix, the unsplit
            # one also M, for the source
            self._RHS_MATRIX = csr_axpby(MASS, 1.0, STIFFNESS, -(1.0 - self._theta) * self._dt)
        if self._theta != 1.0 and not self._split_source:
            self._MASS = MASS
        self._Domain.release_connectivity()
        self._materials.remove_all_element_properties()
        self._Solver.set_matrix(A)
        Ast = A.to_sparse_tensor()
        self._Precond.build_preconditioner(Ast.indices.numpy()[:, 0],
                                           Ast.indices.numpy()[:, 1],
                                           Ast.values.numpy(),
                                           int(Ast.dense_shape.numpy()[0]))
        self._Solver.set_precond(self._Precond)

    def set_initial_condition(self, U0: np.ndarray = None):
        npt = self._Domain.Pts().shape[0]
        if U0 is not None:
            if U0.ndim == 1:
                self._U = tf.Variable(U0[:, np.newaxis], name="U")
            else:
                self._U = tf.Variable(U0, name="U")
        else:
            self._U = tf.Variable(np.full(shape=(npt, 1),
                                          fill_value=self._U_DEFAULT),
                                  name="U", dtype=tf.float32)

    def add_stimulus(self, stimreg: np.ndarray, stimprops: dict):
        self._nbstim += 1
        if self._StimulusDict is None:
            self._StimulusDict = {}
        self._StimulusDict[self._nbstim] = Stimulus(stimprops)
        self._StimulusDict[self._nbstim].set_stimregion(stimreg)

    def finalize_for_run(self):
        if self._use_renumbering:
            self._U = tf.Variable(tf.gather(self._U, self._renumbering['perm']),
                                  name=self._U.name)
            if self._StimulusDict is not None:
                for _key, stim in self._StimulusDict.items():
                    stim.apply_indices_permutation(self._renumbering['perm'])
        self._ready_for_run = True

    # ---- per-step kernel ----------------------------------------------------
    def _warm_start_X0(self, U: tf.Variable) -> tf.constant:
        """ _warm_start_X0(U) returns the CG initial guess for the current step
            and records U as the new previous solution. With linear_guess
            active the guess is the linear extrapolation 2 U^n - U^{n-1} of the
            two previous solutions; otherwise it is U^n unchanged.
        """
        if not self._linear_guess:
            return(U)
        # snapshot U before the solver overwrites it: after the first step
        # the U handed to solve_step aliases ConjGrad's internal X variable,
        # which set_X0 assigns into.
        U_curr = tf.identity(U)
        if self._U_prev is None:
            X0 = U_curr
        else:
            X0 = 2.0 * U_curr - self._U_prev
        self._U_prev = U_curr
        return(X0)

    def solve_step(self, U: tf.Variable, I0: tf.constant) -> tf.Variable:
        """ theta-method solve for the heat equation (implicit Euler for
            theta = 1): the forcing is applied first, then the diffusion step.
            Not a tf.function: it drives the CG loop, whose convergence test
            reads the residual on the host, and keeps the warm-start history
            in a Python attribute. Its kernels (CG iterations, preconditioner,
            cell model) are traced functions of their own.
        """
        X0   = self._warm_start_X0(U)
        RHS  = self._diffusion_rhs(U, I0)
        self._Solver.set_X0(X0)
        self._Solver.set_RHS(RHS)
        self._Solver.solve()
        return self._Solver.X()

    def _diffusion_rhs(self, U: tf.Variable, S: tf.Tensor) -> tf.Tensor:
        """ _diffusion_rhs(U, S) returns the right-hand side of the theta step
            for the potential U and the source S (see _split_source):
            (M - (1-theta) dt K) U + dt M S unsplit, or
            (M - (1-theta) dt K) (U + dt S) split and for implicit Euler,
            where the two coincide and one sparse product suffices.
        """
        if self._MASS is None:
            RHS0 = tf.add(U, tf.math.scalar_mul(self._dt, S))
            return(tf.raw_ops.SparseMatrixMatMul(a=self._RHS_MATRIX._matrix, b=RHS0))
        RHS_U = tf.raw_ops.SparseMatrixMatMul(a=self._RHS_MATRIX._matrix, b=U)
        RHS_S = tf.raw_ops.SparseMatrixMatMul(a=self._MASS._matrix, b=tf.math.scalar_mul(self._dt, S))
        return(tf.add(RHS_U, RHS_S))

    def _compute_forcing(self, ctime: float) -> tf.constant:
        """ Aggregate active stimuli at simulation time ctime. """
        # The zero forcing is created once, on the device. Building it from a
        # host array at every step copied n_nodes values to the GPU per step
        # (4 MB on the 1M-node spiral mesh), and that copy waited for the queued
        # GPU work. An inactive stimulus adds exactly zero, so it is skipped.
        if self._I0_zero is None or self._I0_zero.shape != self._U.shape:
            self._I0_zero = tf.zeros(shape=self._U.shape, dtype=tf.float32, name="I")
        I0 = self._I0_zero
        if self._StimulusDict is not None:
            t_const = tf.constant(ctime, dtype=tf.float32)
            for _name, stim in self._StimulusDict.items():
                if stim.stimulate_tissue_timevalue(t_const):
                    I0 = tf.add(I0, stim())
        return I0

    def step(self, t: float) -> tf.Variable:
        """ Advance one time step. Updates internal state and returns the new U.
            t is the absolute simulation time at the end of the step.
        """
        if not self._ready_for_run:
            raise Exception("model not initialised for run!")
        I0 = self._compute_forcing(t)
        U1 = self.solve_step(self._U, I0)
        self._U = U1
        self._ctime = t
        return self._U

    # ---- checkpoints --------------------------------------------------------
    def checkpoint(self) -> dict:
        """ checkpoint() returns the state needed to resume the run from ctime():
            {'ionic_model': name of the cell model ('' for pure diffusion),
             'time': ctime() in ms, 'num_nodes': number of mesh nodes,
             'Vm': nodal potential, 'state_variables': {name: nodal values}}.
            Every nodal array is flat and in the user's node order, whatever the
            renumbering: a checkpoint then does not depend on the renumbering
            setting, and a run can resume with it switched on or off.
        """
        try:
            return({'ionic_model': self.checkpoint_model_name(),
                    'time': float(self._ctime),
                    'num_nodes': int(self._Domain.Pts().shape[0]),
                    'Vm': self._to_user_order(np.reshape(self._U.numpy(), (-1,))),
                    'state_variables': self._checkpoint_state_variables()})
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def restore_checkpoint(self, checkpoint: dict):
        """ restore_checkpoint(checkpoint) resumes from a dict made by checkpoint():
            it sets the potential, the state variables and ctime().
            It can be called before finalize_for_run() (the values are then
            renumbered with the rest of the solver) or after it (they are
            renumbered here). It raises ValueError when the checkpoint comes
            from a different mesh or cell model.
        """
        try:
            npt = int(self._Domain.Pts().shape[0])
            if int(checkpoint['num_nodes']) != npt:
                raise ValueError('the checkpoint holds {} nodes but the mesh has {}'.format(
                    int(checkpoint['num_nodes']), npt))
            if checkpoint['ionic_model'] != self.checkpoint_model_name():
                raise ValueError('the checkpoint was written with cell model "{}" but this run '
                                 'uses "{}"'.format(checkpoint['ionic_model'],
                                                    self.checkpoint_model_name()))
            Vm = np.asarray(checkpoint['Vm'])
            if Vm.size != npt:
                raise ValueError('the checkpoint potential has {} values but the mesh has '
                                 '{} nodes'.format(Vm.size, npt))
            U0 = np.reshape(Vm, (npt, 1)).astype(np.float32)
            if self._ready_for_run:
                self._U = tf.Variable(self._to_solver_order(U0), name="U", dtype=tf.float32)
            else:
                self.set_initial_condition(U0)
            self._restore_state_variables(checkpoint['state_variables'])
            self._ctime = float(checkpoint['time'])
            # the warm-start history U^{n-1} is not part of a checkpoint, so the
            # first step after a restart starts CG from U^n. This changes the
            # initial guess only: the solution agrees with an uninterrupted run
            # to within the CG tolerance, at the cost of a few extra iterations
            # on that one step.
            self._U_prev = None
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def checkpoint_model_name(self) -> str:
        """ checkpoint_model_name() returns the cell-model name recorded in a
            checkpoint: '' for the heat equation, which has no cell model
        """
        return('')

    def _checkpoint_state_variables(self) -> dict:
        """ the heat equation advances no state besides the potential """
        return({})

    def _restore_state_variables(self, states: dict):
        """ refuses state variables: the heat equation has none to receive them """
        if len(states) > 0:
            raise ValueError('the checkpoint carries state variables ({}) but this run has '
                             'no cell model'.format(', '.join(sorted(states.keys()))))

    def _to_user_order(self, values: np.ndarray) -> np.ndarray:
        """ maps a nodal array from the solver's node order to the user's.
            The permutation is applied by finalize_for_run(), so before it the
            two orders coincide.
        """
        if self._use_renumbering and self._ready_for_run:
            return(np.take(values, self._renumbering['iperm'], axis=0))
        return(values)

    def _to_solver_order(self, values: np.ndarray) -> np.ndarray:
        """ maps a nodal array from the user's node order to the solver's """
        if self._use_renumbering and self._ready_for_run:
            return(np.take(values, self._renumbering['perm'], axis=0))
        return(values)

    # ---- accessors ----------------------------------------------------------
    def domain(self) -> Triangulation:
        return self._Domain

    def solver(self) -> ConjGrad:
        return self._Solver

    def set_linear_guess(self, lg: bool):
        """ set_linear_guess(lg) enables (True) or disables (False) the
            linear-extrapolation warm start for the CG solve. """
        self._linear_guess = lg

    def linear_guess(self) -> bool:
        """ linear_guess() returns the warm-start strategy flag. """
        return(self._linear_guess)

    def precond(self) -> JacobiPrecond:
        return self._Precond

    def stimulus(self) -> dict:
        return self._StimulusDict

    def U(self) -> tf.Variable:
        if self._use_renumbering:
            return tf.gather(self._U, self._renumbering['iperm'])
        return self._U

    def renumbering(self) -> dict:
        return self._renumbering

    def ready_for_run(self) -> bool:
        """ ready_for_run() returns True once finalize_for_run() has been called,
            i.e. once the nodal quantities are in the solver's own node order.
            A setup step that works in the user's node order tests it to refuse
            to run too late rather than to renumber twice.
        """
        return(self._ready_for_run)

    def nt(self) -> int:
        return self._nt

    def split_source(self) -> bool:
        """ split_source() tells whether the source is split from the
            diffusion step (the reference's scheme) rather than taken unsplit """
        return(self._split_source)

    def theta(self) -> float:
        """ theta() returns the weight of the new time level in the diffusion
            step: 0.5 Crank-Nicolson, 1 implicit Euler """
        return(self._theta)

    def dt(self) -> float:
        return self._dt

    def dt_per_plot(self) -> int:
        return self._dt_per_plot

    def ctime(self) -> float:
        return self._ctime

    def Tend(self) -> float:
        return self._Tend

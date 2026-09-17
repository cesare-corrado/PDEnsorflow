import numpy as np
from time import time
import tensorflow as tf
from tensorflow.python.ops.linalg.sparse.sparse_csr_matrix_ops import CSRSparseMatrix
from gpuSolve.linearsolvers.abstract_precond import AbstractPrecond

class ConjGrad:
    """
    Class ConjGrad
    This class defines the conjugate gradient solver.
    To invert symmetric positive definite matrices
    """
    def __init__(self,config : dict = None):
        self._maxiter : int = 100
        self._toll : float   = 1.e-5
        # relative tolerance on the residual: a relative stopping test, scaled
        # by the norm of the preconditioned right-hand side, so the same
        # tolerance means the same accuracy whatever the size of the problem.
        # Inert by default: 0.0 can never fire the relative test, so every
        # caller that does not opt in via set_toll_rel/config keeps the
        # absolute-only stopping behaviour.
        self._toll_rel : float = 0.0
        self._verbose: bool = False
        self._A : tf.sparse.SparseTensor = None
        # CSR variant of A (opaque variant tensor with
        # handle_data set). Built once in set_matrix and consumed by every
        # SpMV via tf.raw_ops.SparseMatrixMatMul — 1.6x / 2.2x faster per
        # SpMV on coarse / fine meshes than tf.sparse.sparse_dense_matmul.
        self._A_csr = None
        self._RHS : tf.Variable = None
        self._X : tf.Variable = None
        self._r : tf.constant = None
        self._p : tf.constant = None
        self._Precond : AbstractPrecond = None
        # number of CG iterations between two GPU->CPU
        # synchronisations for the convergence check. The original code
        # synchronised at every iteration, forcing one device->host
        # transfer per CG step. Checking in blocks removes most of that
        # overhead. A small block (default 5) is used so that CG cases
        # that converge very fast (e.g. warm-start time-stepping where
        # convergence happens in <10 iterations) break out before the
        # algorithm runs into denormals (rzold -> 0, alpha = 0/0 -> NaN).
        self._check_every : int = 5
        # GPU-resident path (single @tf.function + tf.while_loop, see
        # _solve_graph), off by default; switch it on with set_use_graph_loop
        # for small meshes. Which path is faster depends on the mesh size
        # (TensorFlow 2.21, RTX A2000, ms per CG iteration, eager / graph loop):
        # 0.87 / 0.65 on 63001 nodes, 0.92 / 1.41 on 1002001, 1.70 / 2.33 on
        # 2002225. On small meshes the per-operation launch cost dominates and
        # one graph launch saves it (a monodomain step on 63001 nodes: 18.4 ms
        # against 51.7 ms); on production-size meshes the eager path is faster.
        # It also loses when CG converges within the first _check_every
        # iterations, where one graph launch per step costs more than it saves.
        # Both paths test convergence every _check_every iterations, so they
        # stop at the same iteration.
        self._use_graph_loop : bool = False

        if config is not None:
            for attribute in self.__dict__.keys():
                attribute_name = attribute[1:]
                if attribute_name in config.keys():
                    setattr(self, attribute, config[attribute_name])
            if 'precond' in config.keys():
                self._Precond = config['precond']

        self._niters : int = 0
        self._residual: float = 1.e32
        self._b_norm : float = 0.0


    def set_precond(self, prcnd: AbstractPrecond):
        """
        set_precond(prcnd) assigns the preconditioner prcnd
        """
        self._Precond = prcnd

    def set_maxiter(self, maxit:int):
        """
        set_maxiter(maxit) sets the maximum number of iterations of the solver to maxit
        """
        self._maxiter = maxit

    def set_toll(self, toll:float):
        """
        set_toll(toll) sets the tolerance on the residual used to determine the convergence to toll
        """
        self._toll = toll

    def set_toll_rel(self, tollrel:float):
        """
        set_toll_rel(tollrel) sets the relative tolerance on the residual to tollrel;
        the convergence test compares the residual norm against tollrel times the norm
        of the (preconditioned) right-hand side. 0.0 (default) disables the test.
        """
        self._toll_rel = tollrel

    def set_matrix(self, Amat):
        """
        set_matrix(Amat) assigns the sparse matrix that defines the linear
        system to the solver. Accepts a tf.sparse.SparseTensor (COO) or a
        CSRSparseMatrix wrapper. In both cases the CSR variant is cached on
        `self._A_csr` and used by every SpMV in the solve.
        """
        if isinstance(Amat, tf.sparse.SparseTensor):
            # SparseTensor is in COO; wrap once in CSR so
            # the iterative loop calls cuSPARSE CSR SpMV. tf.sparse.reorder
            # is required — an unordered COO produces an ill-formed CSR
            # whose SpMV output leaves the CUDA module cache in a broken
            # state, making the next op fail with CUDA_ERROR_INVALID_HANDLE.
            Amat        = tf.sparse.reorder(Amat)
            self._A     = Amat
            self._A_csr = CSRSparseMatrix(Amat)._matrix
        elif isinstance(Amat, CSRSparseMatrix):
            self._A     = None
            self._A_csr = Amat._matrix
        else:
            # Advanced: raw variant tensor already produced by
            # SparseTensorToCSRSparseMatrix (handle_data must already be set).
            self._A     = None
            self._A_csr = Amat

    def _spmv(self, v: tf.Tensor) -> tf.Tensor:
        """SpMV against the CSR matrix cached in set_matrix."""
        return tf.raw_ops.SparseMatrixMatMul(a=self._A_csr, b=v)

    def set_RHS(self, RHS: tf.constant):
        """
        set_RHS(RHS) assigns  the righ-hand side of the linear problem to the solver
        """
        if self._RHS is not None:
            self._RHS.assign(RHS)
        else:
            self._RHS = tf.Variable(RHS, trainable=False)

    def set_X0(self, X0: tf.Variable):
        """
        set_X0(X0) assigns the inital guess X0 to the solver
        """
        if self._X is not None:
            self._X.assign(X0)
        else:
            self._X = tf.Variable(X0)

    def Precond(self) -> AbstractPrecond:
        """Precond() returns the preconditioner object
        """
        return(self._Precond)

    def maxiter(self) -> int:
        """
        maxiter() returns the maximum number of iterations
        """
        return(self._maxiter)

    def toll(self) -> float:
        """
        toll() returns the tolerance on the residual used to determine the convergence
        """
        return(self._toll)

    def toll_rel(self) -> float:
        """
        toll_rel() returns the relative tolerance on the residual used to determine the convergence
        """
        return(self._toll_rel)

    def set_use_graph_loop(self, use_graph_loop: bool):
        """
        set_use_graph_loop(use_graph_loop) runs the whole CG loop as one graph
        on the GPU (True) instead of the eager per-iteration path (False,
        default). Faster on small meshes (about 63k nodes), slower on meshes of
        1M nodes and more.
        """
        self._use_graph_loop = use_graph_loop

    def use_graph_loop(self) -> bool:
        """ use_graph_loop() returns True if the CG loop runs as one graph """
        return(self._use_graph_loop)

    def matrix(self) ->  tf.sparse.SparseTensor :
        """
        matrix() returns the sparse matrix that defines the linear system
        """
        return(self._A)

    def RHS(self) ->  tf.constant :
        """
        RHS() returns the right-hand side of the linear problem
        """
        return(self._RHS)

    def X(self) ->  tf.Variable :
        """
        X() returns the solution/initial value
        """
        return(self._X)

    def verbose(self):
        """
        verbose() returns the verbosity flag
        """
        return(self._verbose)

    def summary(self):
        """
        summary() prints info on the solver convergence.
        """
        if(self._niters<self._maxiter):
            tf.print('CG converged in {} iterations (residual: {:4.3f})'.format(self._niters,self._residual))
        else:
            tf.print('WARNING: max nb of iteration reached (residual: {:4.3f})'.format(self._residual))


    # _iterate and _initialize serve the per-iteration path (use_graph_loop
    # False). They are plain eager functions, not tf.functions: the cost of
    # calling a traced function at every CG iteration exceeds what tracing
    # saves on a handful of vector operations (51.0 ms against 81.4 ms per
    # monodomain step on 63001 nodes). They return the vectors they update and
    # the caller stores them, so they behave the same whether or not
    # tf.config.run_functions_eagerly is set.
    def _iterate(self, rzold: tf.Tensor, r: tf.Tensor, p: tf.Tensor) -> tuple:
        """ one CG iteration from (rzold, r, p); returns (rznew, r, p, residual) """
        Ap             = self._spmv(p)
        # alpha and beta are safe divisions. An exactly zero residual (a zero
        # initial guess with a zero right-hand side, as in a pure-diffusion run
        # before any stimulus) makes r, p and rzold all 0, and the plain
        # division 0/0 would write NaN into X before the batched convergence
        # check (every _check_every iterations) could stop the loop. The graph
        # path does not need this: its loop condition tests the residual before
        # the first iteration. divide_no_nan returns exactly x/y for any
        # non-zero y, so every non-degenerate solve is bit-identical, and 0 for
        # y = 0, which leaves X at its exact value.
        alpha          = tf.math.divide_no_nan(rzold, tf.reduce_sum(tf.multiply(p, Ap)))
        self._X.assign_add(alpha * p)
        r              = r - alpha * Ap
        if self._Precond:
            # the tested quantity is the *preconditioned* residual
            # ||z||^2 = ||M^-1 r||^2, not the raw ||r||^2, so the stopping
            # test measures the error the preconditioned system actually
            # sees. z is needed anyway for the search direction, so this only
            # adds one reduction and no extra preconditioner solve.
            z        = self._Precond.solve_precond_system(r)
            rznew    = tf.reduce_sum(tf.multiply(r, z))
            residual = tf.reduce_sum(tf.multiply(z, z))
            beta     = tf.math.divide_no_nan(rznew, rzold)
            p        = z + beta * p
        else:
            # no preconditioner: the tested residual falls back to the raw
            # ||r||^2, which is the identity-preconditioner case of the above.
            residual = tf.reduce_sum(tf.multiply(r, r))
            rznew    = residual
            beta     = tf.math.divide_no_nan(rznew, rzold)
            p        = r + beta * p
        return(rznew, r, p, residual)


    def _initialize(self) -> tuple:
        """ starts CG from X; returns (rzold, r, p, residual, b_norm) """
        AX             = self._spmv(self._X)
        r              = tf.subtract(self._RHS, AX)
        if self._Precond:
            z         = self._Precond.solve_precond_system(r)
            p         = z
            rzold     = tf.reduce_sum(tf.multiply(r, z))
            residual  = tf.reduce_sum(tf.multiply(z, z))
            # reference norm for the relative test: ||M^-1 b||, computed
            # once per solve before the loop, since b does not change during
            # it (one extra preconditioner solve per solve() call).
            zb        = self._Precond.solve_precond_system(self._RHS)
            b_norm    = tf.sqrt(tf.reduce_sum(tf.multiply(zb, zb)))
        else:
            p         = r
            residual  = tf.reduce_sum(tf.multiply(r, r))
            rzold     = residual
            b_norm    = tf.sqrt(tf.reduce_sum(tf.multiply(self._RHS, self._RHS)))
        return(rzold, r, p, residual, b_norm)

    # fused GPU-resident CG.
    # The original solve() ran a Python for-loop that called _iterate() and
    # tested the residual on the host every iteration. Each host-side test
    # forces the CPU to block on a 4-byte device->host copy of the scalar
    # residual, which drains the GPU command queue between iterations.
    # The method below puts the *entire* CG loop inside a single
    # @tf.function with tf.while_loop. Control flow, convergence test, and
    # NaN/Inf guard are all tensor ops that execute on the GPU. The CPU
    # dispatches one graph launch per call to solve() and only reads back
    # the final X, the iteration count, and ||r||^2 -- so the cost of the
    # per-iteration synchronisation becomes O(1) instead of O(maxiter).
    @tf.function
    def _solve_graph(self, X0: tf.Tensor, RHS: tf.Tensor,
                     toll_sq: tf.Tensor, toll_rel_sq: tf.Tensor,
                     maxit: tf.Tensor):
        r0 = RHS - self._spmv(X0)
        # Same convergence bookkeeping as the eager path (_iterate /
        # _initialize): the tested quantity zsq is the *preconditioned*
        # residual ||z||^2 = ||M^-1 r||^2 (raw ||r||^2 without a
        # preconditioner), so both paths stop at the same point, and the
        # relative threshold is scaled by ||M^-1 b||^2 computed once before
        # the loop.
        if self._Precond is not None:
            z0       = self._Precond.solve_precond_system(r0)
            rzold0   = tf.reduce_sum(tf.multiply(r0, z0))
            zsq0     = tf.reduce_sum(tf.multiply(z0, z0))
            zb       = self._Precond.solve_precond_system(RHS)
            bnorm_sq = tf.reduce_sum(tf.multiply(zb, zb))
        else:
            z0       = r0
            rzold0   = tf.reduce_sum(tf.multiply(r0, r0))
            zsq0     = rzold0
            bnorm_sq = tf.reduce_sum(tf.multiply(RHS, RHS))
        p0     = z0
        rel_sq = toll_rel_sq * bnorm_sq

        def cond(i, X, r, p, rzold, zsq):
            # loop while unconverged: neither the absolute nor the
            # relative test is satisfied (their OR stops the loop).
            return tf.logical_and(
                i < maxit,
                tf.logical_and(
                    tf.logical_and(zsq > toll_sq, zsq > rel_sq),
                    tf.math.is_finite(zsq)))

        # The loop condition is a GPU value that the host must read before
        # every pass, and while it waits the GPU has nothing queued. One pass
        # therefore runs check_every iterations, unrolled when the function is
        # traced, so the host reads the residual as often as the per-iteration
        # path does. With a pass per iteration, a CG iteration on 1002001
        # nodes cost 1.49 ms against 1.10 ms on the per-iteration path.
        # Iterations after convergence inside a pass are harmless: the step
        # lengths are safe divisions, as in _iterate.
        check_every = self._check_every if self._check_every > 0 else 1

        def body(i, X, r, p, rzold, zsq):
            for _k in range(check_every):
                Ap      = self._spmv(p)
                alpha   = tf.math.divide_no_nan(rzold, tf.reduce_sum(tf.multiply(p, Ap)))
                X       = X + alpha * p
                r       = r - alpha * Ap
                if self._Precond is not None:
                    z   = self._Precond.solve_precond_system(r)
                    rznew = tf.reduce_sum(tf.multiply(r, z))
                    zsq   = tf.reduce_sum(tf.multiply(z, z))
                else:
                    z     = r
                    rznew = tf.reduce_sum(tf.multiply(r, r))
                    zsq   = rznew
                beta  = tf.math.divide_no_nan(rznew, rzold)
                p     = z + beta * p
                rzold = rznew
            return i + check_every, X, r, p, rzold, zsq

        i0 = tf.constant(0, dtype=tf.int32)
        i, Xf, _rf, _pf, _rzf, zsqf = tf.while_loop(
            cond, body,
            [i0, X0, r0, p0, rzold0, zsq0])
        return Xf, i, zsqf


    def solve(self):
        """
        solve()
        solves the linear system using CG
        """
        try:
            if self._verbose:
                t0 = time()
            # two implementations are kept:
            #   * use_graph_loop=True -> dispatch one @tf.function containing
            #     a tf.while_loop (see _solve_graph), GPU-resident. It cannot
            #     be compiled with XLA (the CSR SpMV has no XLA kernel). It
            #     pays off on small meshes; see the timings in __init__.
            #   * use_graph_loop=False (default) -> Python for-loop calling
            #     the eager kernels _initialize / _iterate. The convergence
            #     check reads ||z||^2 back to the host every _check_every
            #     iters, so 4/5 of the per-iteration stalls are avoided while
            #     the CPU keeps the GPU queue full.
            if self._use_graph_loop:
                toll_sq     = tf.constant(self._toll * self._toll, dtype=self._X.dtype)
                toll_rel_sq = tf.constant(self._toll_rel * self._toll_rel, dtype=self._X.dtype)
                maxit       = tf.constant(self._maxiter, dtype=tf.int32)
                Xf, niters, zsq = self._solve_graph(
                    self._X.read_value(), self._RHS.read_value(),
                    toll_sq, toll_rel_sq, maxit)
                self._X.assign(Xf)
                self._niters   = int(niters.numpy())
                self._residual = float(zsq.numpy())
            else:
                self._niters   = 0
                rzold, self._r, self._p, self._residual, self._b_norm = self._initialize()
                if self._verbose:
                    tf.print('initial residual: {:4.3f}'.format(self._residual))
                check_every = self._check_every if self._check_every > 0 else 1
                toll_sq     = self._toll * self._toll
                # relative threshold, fixed once per solve: (toll_rel *
                # ||M^-1 b||)^2 -- or (toll_rel * ||b||)^2 without a
                # preconditioner -- from the right-hand-side norm computed in
                # _initialize before the loop. With the default _toll_rel = 0.0 the
                # threshold is 0, the relative test never fires, and the
                # host-side read of _b_norm is skipped so the default path
                # pays no extra GPU->CPU synchronisation.
                if self._toll_rel > 0.0:
                    rel_sq  = (self._toll_rel * float(self._b_norm)) ** 2
                else:
                    rel_sq  = 0.0
                for self._niters in range(1,1+self._maxiter):
                    rznew, self._r, self._p, self._residual = self._iterate(rzold, self._r, self._p)
                    if (self._niters % check_every) == 0:
                        # Batched convergence check + NaN/Inf guard
                        # (see _check_every docstring in __init__). The
                        # tested _residual is the preconditioned ||z||^2
                        # (see _iterate); the solver stops when either the
                        # absolute OR the relative test is satisfied.
                        r_val = float(self._residual)
                        if (not np.isfinite(r_val)) or (r_val < toll_sq) or (r_val < rel_sq):
                            break
                    rzold = rznew

            if self._verbose:
                elapsed = time() - t0
                print('done in {:3.2f} s'.format(elapsed),flush=True)
                self.summary()
            if(self._niters>=self._maxiter):
                tf.print('WARNING: max nb of iteration reached (residual: {:4.3f})'.format(self._residual))

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

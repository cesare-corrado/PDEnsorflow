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
        # relative tolerance on the residual, scaled by the norm of the
        # (preconditioned) right-hand side, mirroring torchcor's r_tol.
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
        # opt-in GPU-resident path (single @tf.function +
        # tf.while_loop, see _solve_graph). Correct, but empirically slower
        # on this system (no XLA: ptxas 10.1 < 11.1). Default stays on the
        # eager+batched-sync path which is measurably faster here. Flip to
        # True on hardware where XLA is available and can fuse the body.
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


    @tf.function
    def _iterate(self,rzold:tf.constant) -> tf.constant:
        Ap             = self._spmv(self._p)
        alpha          = rzold /tf.reduce_sum(tf.multiply(self._p, Ap))
        self._X.assign_add(alpha *self._p)
        self._r      -= alpha * Ap
        if self._Precond:
            # Convergence bookkeeping is uniform with torchcor's
            # ConjugateGradient: the tested quantity is the *preconditioned*
            # residual ||z||^2 = ||M^-1 r||^2, not the raw ||r||^2. z is
            # needed anyway for the search direction, so this only adds one
            # reduction and no extra preconditioner solve.
            z        = self._Precond.solve_precond_system(self._r)
            rznew    = tf.reduce_sum(tf.multiply(self._r, z))
            self._residual = tf.reduce_sum(tf.multiply(z, z))
            beta     = (rznew / rzold)
            self._p = z + beta * self._p
        else:
            # no preconditioner: the tested residual falls back to the raw
            # ||r||^2 (torchcor's identity-preconditioner case).
            self._residual = tf.reduce_sum(tf.multiply(self._r, self._r))
            rznew    = self._residual
            beta     = (rznew / rzold)
            self._p = self._r + beta * self._p
        return(rznew)


    @tf.function
    def _initialize(self) -> tf.constant:
        AX             = self._spmv(self._X)
        self._r       = tf.subtract(self._RHS, AX)
        if self._Precond:
            z         = self._Precond.solve_precond_system(self._r)
            self._p  = z
            rzold     = tf.reduce_sum(tf.multiply(self._r, z))
            self._residual = tf.reduce_sum(tf.multiply(z, z))
            # reference norm for the relative test: ||M^-1 b||, computed
            # once per solve exactly as torchcor computes b_norm before
            # its loop (one extra preconditioner solve per solve() call).
            zb            = self._Precond.solve_precond_system(self._RHS)
            self._b_norm  = tf.sqrt(tf.reduce_sum(tf.multiply(zb, zb)))
        else:
            self._p = self._r
            self._residual = tf.reduce_sum(tf.multiply(self._r, self._r))
            rzold    = self._residual
            self._b_norm  = tf.sqrt(tf.reduce_sum(tf.multiply(self._RHS, self._RHS)))
        return(rzold)

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
        # preconditioner), uniform with torchcor, and the relative
        # threshold is scaled by ||M^-1 b||^2 computed once before the loop.
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

        def body(i, X, r, p, rzold, zsq):
            Ap      = self._spmv(p)
            alpha   = rzold / tf.reduce_sum(tf.multiply(p, Ap))
            X       = X + alpha * p
            r       = r - alpha * Ap
            if self._Precond is not None:
                z       = self._Precond.solve_precond_system(r)
                rznew   = tf.reduce_sum(tf.multiply(r, z))
                zsq_new = tf.reduce_sum(tf.multiply(z, z))
            else:
                z       = r
                rznew   = tf.reduce_sum(tf.multiply(r, r))
                zsq_new = rznew
            beta = rznew / rzold
            p    = z + beta * p
            return i + 1, X, r, p, rznew, zsq_new

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
            #     a tf.while_loop (see _solve_graph). Fully GPU-resident; no
            #     device->host sync during the loop. This is the right path
            #     on hardware where XLA can fuse the body, but on systems
            #     without XLA (e.g. ptxas < 11.1) the tf.while_loop overhead
            #     exceeds the saved per-iteration sync cost.
            #   * use_graph_loop=False (default) -> Python for-loop calling
            #     a cached @tf.function (_iterate). The convergence check
            #     reads ||r||^2 back to the host every _check_every iters,
            #     so 4/5 of the per-iteration stalls are avoided while the
            #     CPU still keeps the GPU queue full between graph launches.
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
                rzold = self._initialize()
                if self._verbose:
                    tf.print('initial residual: {:4.3f}'.format(self._residual))
                check_every = self._check_every if self._check_every > 0 else 1
                toll_sq     = self._toll * self._toll
                # relative threshold, fixed once per solve: (toll_rel *
                # ||M^-1 b||)^2 -- or (toll_rel * ||b||)^2 without a
                # preconditioner -- mirroring torchcor's b_norm computed
                # before its loop. With the default _toll_rel = 0.0 the
                # threshold is 0, the relative test never fires, and the
                # host-side read of _b_norm is skipped so the default path
                # pays no extra GPU->CPU synchronisation.
                if self._toll_rel > 0.0:
                    rel_sq  = (self._toll_rel * float(self._b_norm)) ** 2
                else:
                    rel_sq  = 0.0
                for self._niters in range(1,1+self._maxiter):
                    rznew = self._iterate(rzold)
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

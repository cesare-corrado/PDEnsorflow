import numpy as np
from time import time
import tensorflow as tf
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
        self._verbose: bool = False
        self._A : tf.sparse.SparseTensor = None
        self._RHS : tf.Variable = None
        self._X : tf.Variable = None
        self._r : tf.constant = None
        self._p : tf.constant = None
        self._Precond : AbstractPrecond = None
        
        if config is not None:
            for attribute in self.__dict__.keys():
                attribute_name = attribute[1:]
                if attribute_name in config.keys():
                    setattr(self, attribute, config[attribute_name])
            if 'precond' in config.keys():
                self._Precond = config['precond']
            
        self._niters : int = 0
        self._residual: float = 1.e32


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

    def set_matrix(self, Amat: tf.sparse.SparseTensor):
        """
        set_matrix(Amat) assigns the sparse matrix that defines the linear system to the solver
        """
        self._A = Amat

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
        Ap             = tf.sparse.sparse_dense_matmul(self._A,self._p)
        alpha          = rzold /tf.reduce_sum(tf.multiply(self._p, Ap))
        self._X.assign_add(alpha *self._p)
        self._r      -= alpha * Ap
        self._residual = tf.reduce_sum(tf.multiply(self._r, self._r))
        if self._Precond:
            z        = self._Precond.solve_precond_system(self._r)
            rznew    = tf.reduce_sum(tf.multiply(self._r, z))
            beta     = (rznew / rzold)
            self._p = z + beta * self._p
        else:
            rznew    = self._residual
            beta     = (rznew / rzold)
            self._p = self._r + beta * self._p
        return(rznew)

    
    @tf.function    
    def _initialize(self) -> tf.constant:
        AX             = tf.sparse.sparse_dense_matmul(self._A,self._X)
        self._r       = tf.subtract(self._RHS, AX)
        self._residual = tf.reduce_sum(tf.multiply(self._r, self._r))
        if self._Precond:
            z         = self._Precond.solve_precond_system(self._r)
            self._p  = z
            rzold     = tf.reduce_sum(tf.multiply(self._r, z))
        else:
            self._p = self._r
            rzold    = self._residual
        return(rzold)


    def solve(self):
        """
        solve()    
        solves the linear system using CG
        """
        try:
            self._niters   = 0
            if self._verbose:
                t0             = time()
            rzold = self._initialize()
            
            if self._verbose:
                tf.print('initial residual: {:4.3f}'.format(self._residual))    
            for self._niters in range(1,1+self._maxiter):
                rznew = self._iterate(rzold)
                if (tf.sqrt(self._residual) < self._toll):
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

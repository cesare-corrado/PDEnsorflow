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
        self.__A : tf.sparse.SparseTensor = None
        self.__RHS : tf.Variable = None
        self.__X : tf.Variable = None
        self.__r : tf.constant = None
        self.__p : tf.constant = None
        self.__Precond : AbstractPrecond = None
        
        if config is not None:
            for attribute in self.__dict__.keys():
                attribute_name = attribute[1:]
                if attribute_name in config.keys():
                    setattr(self, attribute, config[attribute_name])
            if 'precond' in config.keys():
                self.__Precond = config['precond']
            
        self._niters : int = 0
        self._residual: float = 1.e32


    def set_precond(self, prcnd: AbstractPrecond):
        """
        set_precond(prcnd) assigns the preconditioner prcnd
        """
        self.__Precond = prcnd

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
        self.__A = Amat 

    def set_RHS(self, RHS: tf.constant):
        """
        set_RHS(RHS) assigns  the righ-hand side of the linear problem to the solver
        """
        if self.__RHS is not None:
            self.__RHS.assign(RHS)
        else:
            self.__RHS = tf.Variable(RHS, trainable=False)

    def set_X0(self, X0: tf.Variable):
        """
        set_X0(X0) assigns the inital guess X0 to the solver 
        """
        if self.__X is not None:
            self.__X.assign(X0)
        else:
            self.__X = tf.Variable(X0)

    def Precond(self) -> AbstractPrecond:
        """Precond() returns the preconditioner object
        """
        return(self.__Precond)

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
        return(self.__A) 

    def RHS(self) ->  tf.constant :
        """
        RHS() returns the right-hand side of the linear problem
        """
        return(self.__RHS) 

    def X(self) ->  tf.Variable :
        """
        X() returns the solution/initial value
        """
        return(self.__X) 

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
    def __iterate(self,rzold:tf.constant) -> tf.constant:
        Ap             = tf.sparse.sparse_dense_matmul(self.__A,self.__p)
        alpha          = rzold /tf.reduce_sum(tf.multiply(self.__p, Ap))
        self.__X.assign_add(alpha *self.__p) 
        self.__r      -= alpha * Ap
        self._residual = tf.reduce_sum(tf.multiply(self.__r, self.__r))
        if self.__Precond:
            z        = self.__Precond.solve_precond_system(self.__r)
            rznew    = tf.reduce_sum(tf.multiply(self.__r, z)) 
            beta     = (rznew / rzold)
            self.__p = z + beta * self.__p
        else:
            rznew    = self._residual
            beta     = (rznew / rzold)
            self.__p = self.__r + beta * self.__p
        return(rznew)

    
    @tf.function    
    def __initialize(self) -> tf.constant:    
        AX             = tf.sparse.sparse_dense_matmul(self.__A,self.__X)
        self.__r       = tf.subtract(self.__RHS, AX)
        self._residual = tf.reduce_sum(tf.multiply(self.__r, self.__r))
        if self.__Precond:
            z         = self.__Precond.solve_precond_system(self.__r)
            self.__p  = z
            rzold     = tf.reduce_sum(tf.multiply(self.__r, z)) 
        else:
            self.__p = self.__r
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
            rzold = self.__initialize()
            
            if self._verbose:
                tf.print('initial residual: {:4.3f}'.format(self._residual))    
            for self._niters in range(1,1+self._maxiter):
                rznew = self.__iterate(rzold)
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

import numpy as np
import sys
from time import time
import tensorflow as tf


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
        self._RHS : tf.constant = None
        self._X : tf.constant = None
        
        if config is not None:
            for attribute in self.__dict__.keys():
                attribute_name = attribute[1:]
                if attribute_name in config.keys():
                    setattr(self, attribute, config[attribute_name])
        
        self._niters : int = 0
        self._residual: float = 1.e32


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
        self._RHS = RHS 

    def set_X0(self, X0: tf.constant):
        """
        set_X0(X0) assigns the inital guess X0 to the solver 
        """
        self._X = X0 

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

    def X(self) ->  tf.constant :
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
        
    def solve(self):
        """
        solve()    
        solves the linear system using CG
        """
        try:
            self._niters   = 0
            t0             = time()
            r              = self._RHS - tf.sparse.sparse_dense_matmul(self._A,self._X)
            p              = r
            self._residual = tf.reduce_sum(tf.multiply(r, r))
            if self._verbose:
                tf.print('initial residual: {:4.3f}'.format(self._residual))    
            for self._niters in range(1,1+self._maxiter):
                Ap             = tf.sparse.sparse_dense_matmul(self._A,p)
                alpha          = self._residual /tf.reduce_sum(tf.multiply(p, Ap))
                self._X       += alpha * p 
                r             -= alpha * Ap
                rsnew          = tf.reduce_sum(tf.multiply(r, r))
                beta           = (rsnew / self._residual)
                p              = r + beta * p
                self._residual = rsnew 
                if (tf.sqrt(self._residual) < self._toll):
                    break
            elapsed = time() - t0
            print('done in {:3.2f} s'.format(elapsed),flush=True)            
            if self._verbose:
                self.summary()
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

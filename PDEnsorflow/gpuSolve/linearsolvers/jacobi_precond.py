from gpuSolve.linearsolvers.abstract_precond import AbstractPrecond
import numpy as np
import tensorflow as tf

class JacobiPrecond(AbstractPrecond):
    """
    class Jacobi_precond
    This class implements the Jacobi preconditioner
    
    """

    def __init__(self):
        self._I : tf.constant    = None
        self._J : tf.constant    = None 
        self._V : tf.constant    = None
        self._dim : int          = None


    def build_preconditioner(self,I0: np.ndarray, J0: np.ndarray, V0: np.ndarray,msize: int):
        """
        build_preconditioner(I0,J0,V0,msize): builds the Jacobi preconditioner in sparse form
        Since the preconditioner is diagonal, we do not convert I0 and J0
        to tf.constant and we evaluate the inverse directly
        """
        self._dim = msize
        # vectorised diagonal extraction replaces the Python
        # loop over all non-zeros (was O(nnz) with per-entry python overhead).
        V          = np.zeros(shape=(self._dim,1),dtype = float)
        diag_mask  = (I0 == J0)
        diag_idx   = I0[diag_mask]
        V[diag_idx, 0] = 1.0 / V0[diag_mask]
        
    @tf.function
    def solve_precond_system(self, residual : tf.constant) -> tf.constant:
        """solve_precond_system(residual) computes the preconditioned residual 
        solving z = M^-1 r.
        For a jacobi preconditioner, N^1 is easily evaluated.
        """
        Z = tf.multiply(self._V,residual)
        return (Z)
        
        
        
        
        
        
        
        
        

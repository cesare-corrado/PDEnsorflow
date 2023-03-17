from gpuSolve.linearsolvers.abstract_precond import AbstractPrecond
import numpy as np
import tensorflow as tf

class JacobiPrecond(AbstractPrecond):
    """
    class Jacobi_precond
    This class implements the Jacobi preconditioner
    
    """

    def __init__(self):
        self.__I : tf.constant    = None
        self.__J : tf.constant    = None 
        self.__V : tf.constant   = None
        self.__dim : int         = None


    def build_preconditioner(self,I0: np.ndarray, J0: np.ndarray, V0: np.ndarray,msize: int):
        """
        build_preconditioner(I0,J0,V0,msize): builds the Jacobi preconditioner in sparse form
        Since the preconditioner is diagonal, we do not convert I0 and J0 
        to tf.constant and we evaluate the inverse directly
        """
        self.__dim = msize
        nnzero     = I0.shape
        I          = np.zeros(shape=(self.__dim,1),dtype=int)
        J          = np.zeros(shape=(self.__dim,1),dtype=int)
        V          = np.zeros(shape=(self.__dim,1),dtype=float)
        nnzero     = I.shape[0]
        for ir,jc,mv in zip(I0,J0,V0):
            if ir==jc:
                I[ir] = ir
                J[ir] = jc
                V[ir] = 1./mv
        #self.__I = tf.constant(self.__I,name="Iprecond",dtype=np.int32)
        #self.__J = tf.constant(self.__J,name="Jprecond",dtype=np.int32)
        self.__V = tf.constant(V,name="Vprecond",dtype=np.float32)
        
    @tf.function
    def solve_precond_system(self, residual : tf.constant) -> tf.constant:
        """solve_precond_system(residual) computes the preconditioned residual 
        solving z = M^-1 r.
        For a jacobi preconditioner, N^1 is easily evaluated.
        """
        Z = tf.multiply(self.__V,residual)
        return (Z)
        
        
        
        
        
        
        
        
        

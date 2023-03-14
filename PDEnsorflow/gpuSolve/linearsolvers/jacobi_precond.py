from gpuSolve.linearsolvers.abstract_precond import AbstractPrecond
import numpy as np
import tensorflow as tf

class JacobiPrecond(AbstractPrecond):
    """
    class Jacobi_precond
    This class implements the Jacobi preconditioner
    
    """

    def __init__(self):
        self.__I : np.ndarray    = None
        self.__J : np.ndarray    = None 
        self.__V : tf.constant   = None
        self.__dim : int         = None


    def build_preconditioner(self,I: np.ndarray, J: np.ndarray, V: np.ndarray,msize: int):
        """
        build_preconditioner(I,V,V,msize): builds the Jacobi preconditioner in sparse form
        Since the preconditioner is diagonal, we do not convert I and J 
        to tf.constant and we evaluate the inverse directly
        """
        self.__dim = msize
        nnzero     = I.shape
        self.__I = np.zeros(shape=(self.__dim),dtype=int)
        self.__J = np.zeros(shape=(self.__dim),dtype=int)
        self.__V = np.zeros(shape=(self.__dim),dtype=float)
        nnzero = I.shape[0]
        for ir,jc,mv in zip(I,J,V):
            if ir==jc:
                self.__I[ir] = ir
                self.__J[ir] = jc
                self.__V[ir] = 1./mv
        #self.__I = tf.constant(self.__I,name="Iprecond",dtype=np.int32)
        #self.__J = tf.constant(self.__J,name="Jprecond",dtype=np.int32)
        self.__V = tf.constant(self.__V,name="Vprecond",dtype=np.float32)
        
    @tf.function
    def solve_precond_system(self, residual : tf.constant) -> tf.constant:
        """solve_precond_system(residual) computes the preconditioned residual 
        solving z = M^-1 r.
        For a jacobi preconditioner, N^1 is easily evaluated.
        """
        return (tf.expand_dims(self.__V*tf.squeeze(residual),axis=1) )
        
        
        
        
        
        
        
        
        

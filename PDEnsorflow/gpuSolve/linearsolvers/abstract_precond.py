from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf


class AbstractPrecond(metaclass=ABCMeta):
    """
    class AbstractPrecond
    This class implements the abstract class for a generic preconditioner.
    Each derived preconditioners must implement:
    * build_preconditioner(I,J,V,msize): to build the preconditioner from the system matrix (in coo format)
    * solve_precond_system(residual) to apply the preconditioner to the residual and return the preconditioned residual
    
    """
    @abstractmethod
    def build_preconditioner(self, I: np.ndarray, J: np.ndarray, V: np.ndarray, msize: int):
        """
        build_preconditioner(I,V,V,msize): builds the preconditioner from the matrix in sparse coo form
        """
        pass

    @abstractmethod
    @tf.function
    def solve_precond_system(self, residual : tf.constant) -> tf.constant:
        """solve_precond_system(residual) computes the preconditioned residual 
        solving z = M^-1 r
        """
        pass


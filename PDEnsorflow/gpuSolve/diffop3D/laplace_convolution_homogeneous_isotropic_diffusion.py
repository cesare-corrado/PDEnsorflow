import tensorflow as tf
import numpy as np

@tf.function
def laplace_convolution_homogeneous_isotropic_diffusion(X0,DX,DY,DZ):
    """
    This function computes the 3D Laplace operator on X for homogeneous isotropic diffusion.
    Using the conv3d convolution operator. Conv3d si the extension of conv2d to 3d images,
    where the 3rd dimension IS NOT a channel (the channela are in the 4th)
    This formula does not take into account of the domain, except its sizes along x,y, and z.
    We use the laplace formula found in : https://en.wikipedia.org/wiki/Discrete_Laplace_operator, adapted
    to take into account DX,DY and DZ. (experimental)

    Input:
        X0:       the (tensor) variable one wants to compute the laplace operator; shape: (D, H, W )
        D{X,Y,Z}: the element sizes along the 3 directions
    Output: 
        The laplace oprator
    """
    
    #Pad here with symmetric padding to properly set the B.C.
    padmode = 'symmetric'
    paddings = tf.constant([[1,1], [1,1], [1,1]])
    X = tf.pad(X0, paddings=paddings, mode=padmode) 
  
    # Build the kernel  
    dxsq = DX*DX
    dysq = DY*DY
    dzsq = DZ*DZ
    dssq=1.0/dxsq+1.0/dysq+1.0/dzsq
    
    kernel = np.array([[[0.0, 0.0, 0.0],
             [0.0, 1.0/dzsq, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 1.0/dysq, 0.0],
             [1.0/dxsq, -2.0*dssq, 1.0/dxsq],
             [0.0, 1.0/dysq, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 1.0/dzsq, 0.0],
             [0.0, 0.0, 0.0]]] )
             
    laplace_k = tf.constant(kernel.reshape(list(kernel.shape) + [1,1]) , dtype=1   )
    
    X = tf.expand_dims(tf.expand_dims(X, 0), -1)
    #y = tf.nn.conv3d(X, laplace_k, [1,1, 1, 1, 1], padding='SAME')
    y = tf.nn.conv3d(X, laplace_k, [1,1, 1, 1, 1], padding='VALID')
    return y[0, :, :,:, 0]
    

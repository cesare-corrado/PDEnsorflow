import tensorflow as tf

@tf.function
def laplace_heterogeneous_isotropic_diffusion(X0,DIFF0,DX,DY,DZ):
    """
    This function computes the 3D Laplace operator on X for an heterogeneous isotropic diffusion
    The diffusion encodes the phase field that describes the domain 
    (e.g., elements outside the domain must have diffusion equal to 0)
    The boundaries of the 3D cube are extended with a symmetric padding to generate a non-flux b.c.
    Input:
        X0:       the (tensor) variable one wants to compute the laplace operator
        DIFF0:    the (tensor of the) conductivity values
        D{X,Y,Z}: the element sizes along the 3 directions
    Output: 
        The tensor with the laplace operator values
    
    """
    padmode = 'symmetric'
    paddings = tf.constant([[1,1], [1,1], [1,1]])
    X    = tf.pad(X0, paddings=paddings, mode=padmode) 
    DIFF = tf.pad(DIFF0, paddings=paddings, mode=padmode) 
    # Gx
    eGrad =  0.5*( DIFF[2:,1:-1,1:-1]   + DIFF[1:-1, 1:-1,1:-1] )* (X[2:,1:-1,1:-1]   - X[1:-1, 1:-1,1:-1] )
    wGrad = -0.5*( DIFF[0:-2,1:-1,1:-1] + DIFF[1:-1, 1:-1,1:-1] )* (X[0:-2,1:-1,1:-1] - X[1:-1, 1:-1,1:-1] )
    # Gy
    nGrad =  0.5*( DIFF[1:-1,2:,1:-1] + DIFF[1:-1, 1:-1,1:-1]   )* (X[1:-1,2:,1:-1]   - X[1:-1, 1:-1,1:-1] )
    sGrad = -0.5*( DIFF[1:-1,0:-2,1:-1] + DIFF[1:-1, 1:-1,1:-1] )* (X[1:-1,0:-2,1:-1] - X[1:-1, 1:-1,1:-1] )
    # Gz    
    uGrad =  0.5*( DIFF[1:-1,1:-1,2:] + DIFF[1:-1, 1:-1,1:-1]   )* (X[1:-1,1:-1,2:]   - X[1:-1, 1:-1,1:-1] )
    dGrad = -0.5*( DIFF[1:-1,1:-1,0:-2] + DIFF[1:-1, 1:-1,1:-1] )* (X[1:-1,1:-1,0:-2] - X[1:-1, 1:-1,1:-1] )
    dxsq = DX*DX
    dysq = DY*DY
    dzsq = DZ*DZ
    lapla = (eGrad - wGrad)/dxsq + (nGrad - sGrad)/dysq  + (uGrad - dGrad)/dzsq
    return lapla


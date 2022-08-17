import tensorflow as tf

@tf.function
def laplace_homogeneous_isotropic_diffusion(X0,DX,DY):
    """
    This function computes the 2D Laplace operator on X for homogeneous isotropic diffusion.
    This formula does not take into account of the domain, except its sizes along x,y, and z.
    We use the laplace formula found in : https://en.wikipedia.org/wiki/Discrete_Laplace_operator, adapted
    to take into account DX,DY. (experimental)

    Input:
        X0:       the (tensor) variable one wants to compute the laplace operator; shape: (W, H )
        D{X,Y}: the element sizes along the 2 directions
    Output: 
        The laplace oprator
    """
    paddings = tf.constant([[1,1], [1,1]])
    

    # padding with value at n-1 (the border is the reflection plane)
    #padmode = 'reflect'   
    # padding with n value (the value on the border)
    padmode = 'symmetric'
    X = tf.pad(X0, paddings=paddings, mode=padmode) 
    
    dxsq = DX*DX
    dysq = DY*DY
    lapla   = ((X[0:-2,1:-1]  -2.0*X[1:-1,1:-1] + X[2:,1:-1])/dxsq 
            +  (X[1:-1,0:-2]  -2.0*X[1:-1,1:-1] + X[1:-1,2:])/dysq   )

    '''        
    laplacross 
    '''    
    

    return lapla



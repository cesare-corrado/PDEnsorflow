import tensorflow as tf

@tf.function
def laplace_heterogeneous_anisotropic_diffusion(X0: tf.Variable, DIFF0: tf.constant, AVEC0: tf.constant, DX: tf.constant, DY: tf.constant, DZ: tf.constant) -> tf.constant:
    """
    This function computes the 3D Laplace operator on X when diffusion is anisotropic and inhomogeneous.
    When the domain is different from a cube, the domain shape must be taken into account in the 
    definition of the tensors. E.G.: diffusion must be 0 outside the domain.
    The boundaries of the 3D cube are extended with a symmetric padding to generate a non-flux b.c.
    Note: to speed-up the simulations, some quantities that do not change in time (DIFF0,AVEC0) 
          are pre-computed and hence must be passed in the correct format (see below)

    Algorithm: We proceed as for isotropic laplacian, evaluating the derivatives of left and right vectors.
               The left and right vectors are composed by:
               * The left/right derivatives for the direction of the vector (e.g., x)
               * For the other two directions, a centered finite difference at left/right directions 
                 (e.g. centred derivative of y at xi+1 for right and e.g. centred derivative of y at xi-1 for left)
    Input:
        X0:       the (tensor) variable one wants to compute the laplace operator; shape: (D, H, W )
       DIFF0:  diffusion tensor of shape: (D, H, W,2 ). the first channel is transverse diffusivity sigma_t;
                   the second channel is the difference: sigma_l - sigma_t
       AVEC0:  direction tensor (D, H, W,6 ). Each channel represents the following components of 
               the diadic product: A1A1 A1A2 A1A3  A2A2 A2A3 A3A3
        D{X,Y,Z}: the element sizes along the 3 directions
    Output: 
        The tensor with the values of Div(S*grad(X)), S being the conductivity tensor

    """

    padmode = 'symmetric'
    paddings = tf.constant([[1,1], [1,1], [1,1] ])
    padD     = tf.constant([[1,1], [1,1], [1,1],[0,0] ])

    X    = tf.pad(X0, paddings=paddings, mode=padmode) 
    # sigma_t, sigma_l - sigma_t
    DIFF = tf.pad(DIFF0, paddings=padD, mode=padmode) 
    # A1A1 A1A2 A1A3  A2A2 A2A3 A3A3
    AVEC = tf.pad(AVEC0, paddings=padD, mode=padmode) 
    
    CXX = DIFF[:,:,:,0] + AVEC[:,:,:,0]*DIFF[:,:,:,1]
    CXY =                 AVEC[:,:,:,1]*DIFF[:,:,:,1]
    CXZ =                 AVEC[:,:,:,2]*DIFF[:,:,:,1]
    CYY = DIFF[:,:,:,0] + AVEC[:,:,:,3]*DIFF[:,:,:,1]
    CYZ =                 AVEC[:,:,:,4]*DIFF[:,:,:,1]
    CZZ = DIFF[:,:,:,0] + AVEC[:,:,:,5]*DIFF[:,:,:,1]

    ##################################################################################################        
    ########                                         Gx                                       ########
    ##################################################################################################
    eGrad =  0.5*(
                  (CXX[2:,1:-1,1:-1]+CXX[1:-1, 1:-1,1:-1])* (X[2:,1:-1,1:-1] - X[1:-1, 1:-1,1:-1] )/DX+   
                  0.5*(CXY[2:,2:,1:-1]+CXY[2:, 0:-2,1:-1])* (X[2:,2:,1:-1] - X[2:, 0:-2,1:-1] )/DY+
                  0.5*(CXZ[2:,1:-1,2:]+CXZ[2:,1:-1,0:-2])* (X[2:,1:-1,2:] - X[2:,1:-1,0:-2] )/DZ
                )

    wGrad = -0.5*(
                  (CXX[0:-2,1:-1,1:-1]+CXX[1:-1, 1:-1,1:-1])* (X[0:-2,1:-1,1:-1] - X[1:-1, 1:-1,1:-1] )/DX+    
                  0.5*(CXY[0:-2,2:,1:-1]+CXY[0:-2, 0:-2,1:-1])* (X[0:-2,2:,1:-1] - X[0:-2, 0:-2,1:-1] )/DY+
                  0.5*(CXZ[0:-2,1:-1,2:]+CXZ[0:-2,1:-1,0:-2])* (X[0:-2,1:-1,2:] - X[0:-2:,1:-1,0:-2] )/DZ
                 )
    
    ##################################################################################################        
    ########                                         Gy                                       ########
    ##################################################################################################
    nGrad =  0.5*( 
                   0.5*(CXY[2:,2:,1:-1]+CXY[0:-2, 2:, 1:-1])* (X[2:,2:,1:-1] - X[0:-2,2:, 1:-1] )/DX +    
                   (CYY[1:-1,2:,1:-1] + CYY[1:-1, 1:-1,1:-1]   )* (X[1:-1,2:,1:-1]   - X[1:-1, 1:-1,1:-1] )/DY+
                   0.5*(CYZ[1:-1,2:,2:]+CYZ[1:-1, 2:, 0:-2])* (X[1:-1,2:,2:] - X[ 1:-1,2:,0:-2] )/DZ
                 )  

    sGrad = -0.5*( 
                   0.5*(CXY[2:,0:-2,1:-1]+CXY[0:-2, 0:-2, 1:-1])* (X[2:,0:-2,1:-1] - X[0:-2,0:-2, 1:-1] )/DX +    
                   (CYY[1:-1,0:-2,1:-1] + CYY[1:-1, 1:-1,1:-1] )* (X[1:-1,0:-2,1:-1] - X[1:-1, 1:-1,1:-1] )/DY+
                   0.5*(CYZ[1:-1,0:-2,2:]+CYZ[1:-1, 0:-2, 0:-2])* (X[1:-1,0:-2,2:] - X[ 1:-1,0:-2,0:-2] )/DZ
                 )

    ##################################################################################################        
    ########                                         Gz                                       ########
    ##################################################################################################
    uGrad =  0.5*( 
                  0.5*(CXZ[2:,1:-1,2:]+CXZ[0:-2,1:-1,2:])* (X[2:,1:-1,2:] - X[0:-2,1:-1,2:]  )/DX+
                  0.5*(CYZ[1:-1,2:,2:]+CYZ[1:-1,0:-2,2:])* (X[1:-1,2:,2:] - X[1:-1,0:-2,2:]  )/DY+    
                  (CZZ[1:-1,1:-1,2:] + CZZ[1:-1, 1:-1,1:-1]   )* (X[1:-1,1:-1,2:]   - X[1:-1, 1:-1,1:-1] )/DZ
                 )
    
    dGrad = -0.5*( 
                  0.5*(CXZ[2:,1:-1,0:-2]+CXZ[0:-2,1:-1,0:-2])* (X[2:,1:-1,0:-2] - X[0:-2,1:-1,0:-2] )/DX+
                  0.5*(CYZ[1:-1,2:,0:-2]+CYZ[1:-1,0:-2,0:-2])* (X[1:-1,2:,0:-2] - X[1:-1,0:-2,0:-2] )/DY+
                  (CZZ[1:-1,1:-1,0:-2] + CZZ[1:-1, 1:-1,1:-1] )* (X[1:-1,1:-1,0:-2] - X[1:-1, 1:-1,1:-1] )/DZ
                 )


    lapla = (eGrad - wGrad)/DX + (nGrad - sGrad)/DY  + (uGrad - dGrad)/DZ

    return lapla

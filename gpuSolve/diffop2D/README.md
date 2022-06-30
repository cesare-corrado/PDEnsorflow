# diffop2D

This sub-module of `gpuSolve` defines the 2D space differential operators:

* `laplace_homogeneous_isotropic_diffusion`   with alias `laplace_homog`
* `laplace_heterogeneous_isotropic_diffusion` with alias `laplace_heterog`

## laplace_homog

This function is used on homogeneous isotropic domains and has signature:

```
laplace_homogeneous_isotropic_diffusion(X0,DX,DY)
```

Where `X0` is a *TensorFlow* Variable and `DX`,`DY` are the pixel dimensions along `x` and `y`

This function evaluates the Laplace operator with classical finite differences:

```
lapla   = ((X[0:-2,1:-1]  -2.0*X[1:-1,1:-1] + X[2:,1:-1])/dxsq 
        +  (X[1:-1,0:-2]  -2.0*X[1:-1,1:-1] + X[1:-1,2:])/dysq   )

```

## laplace_heterog
This function is used on hetrogeneous isotropic domains and has signature:

```
laplace_heterogeneous_isotropic_diffusion(X0,DIFF0,DX,DY)
```

Where `X0` is a *TensorFlow* Variable and `DX`,`DY` are the pixel dimensions along `x` and `y`


On each point, this function first evaluate the left and right products of the gradients time the conductivity;
 then, it evaluates the the divergence:
 
```
    # Gx
    eGrad =  0.5*( DIFF[2:,1:-1]   + DIFF[1:-1, 1:-1] )* (X[2:,1:-1]   - X[1:-1, 1:-1] )
    wGrad = -0.5*( DIFF[0:-2,1:-1] + DIFF[1:-1, 1:-1] )* (X[0:-2,1:-1] - X[1:-1, 1:-1] )
    # Gy
    nGrad =  0.5*( DIFF[1:-1,2:] + DIFF[1:-1, 1:-1]   )* (X[1:-1,2:]   - X[1:-1, 1:-1] )
    sGrad = -0.5*( DIFF[1:-1,0:-2] + DIFF[1:-1, 1:-1] )* (X[1:-1,0:-2] - X[1:-1, 1:-1] )

    lapla = (eGrad - wGrad)/dxsq + (nGrad - sGrad)/dysq  

```
 
 
 


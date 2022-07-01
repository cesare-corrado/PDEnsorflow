# diffop3D

This sub-module of `gpuSolve` defines the 3D space differential operators:

* `laplace_homogeneous_isotropic_diffusion`   with alias `laplace_homog`
* `laplace_convolution_homogeneous_isotropic_diffusion` with alias `laplace_conv_homog`
* `laplace_heterogeneous_isotropic_diffusion` with alias `laplace_heterog`
* `laplace_heterogeneous_anisotropic_diffusion` with alias `laplace_heterog_aniso`



## laplace_homog

This function is used on homogeneous isotropic domains and has signature:

```
laplace_homogeneous_isotropic_diffusion(X0,DX,DY,DZ)
```

Where `X0` is a *TensorFlow* Variable and `DX`,`DY` and `DZ` are the voxel dimensions along `x` and `y` and `z`.

This function evaluates the Laplace operator with classical finite differences.


## laplace_conv_homog

This function is used on homogeneous isotropic domains and has signature:

```
laplace_convolution_homogeneous_isotropic_diffusion(X0,DX,DY,DZ)
```

Where `X0` is a *TensorFlow* Variable and `DX`,`DY` and `DZ` are the voxel dimensions along `x` and `y` and `z`.
This function evaluates the Laplace operator using a convolutional layer (`tf.nn.conv3d`), with kernel:

```
    kernel = np.array(
             [
             [[0.0, 0.0, 0.0],
             [0.0, 1.0/dzsq, 0.0],
             [0.0, 0.0, 0.0]],
             
            [[0.0, 1.0/dysq, 0.0],
             [1.0/dxsq, -2.0*dssq, 1.0/dxsq],
             [0.0, 1.0/dysq, 0.0]],
             
            [[0.0, 0.0, 0.0],
             [0.0, 1.0/dzsq, 0.0],
             [0.0, 0.0, 0.0]]
             ] 
             )
```



## laplace_heterog
This function is used on hetrogeneous isotropic domains and has signature:

```
laplace_heterogeneous_isotropic_diffusion(X0,DIFF0,DX,DY,DZ)
```

Where `X0` is a *TensorFlow* Variable, `DIFF0` is the tensor that defines the diffusion coefficients at each voxel and `DX`,`DY` and `DZ` are the voxel dimensions along `x` and `y` and `z`.

On each point, this function first evaluate the left and right products of the gradients time the conductivity;
 then, it evaluates the the divergence.


## laplace_heterog_aniso 
This function is used on hetrogeneous anisotropic conductivities; the conductivity is axyseiimetric, with dominant direction defined by a vector field `A` with unitary modulus and has signature:

```
laplace_heterogeneous_anisotropic_diffusion(X0,DIFF0,AVEC0,DX,DY,DZ)
```

Where: 
* `X0` is the *TensorFlow* Variable we want to detemine the Laplacian
* `DIFF0` is the tensor with dimension *(D, H, W,2 )* that defines the transveral (*channel=0*) diffusion coefficient and the difference between the longitudinal and the transveral diffusion coefficents (*channel=1*)  at each voxel
* `AVEC0` is the tensor with dimension *(D, H, W,6 )* that at each voxel defines the components of the diadic product: *A1A1*, *A1A2*, *A1A3*, *A2A2*, *A2A3*, *A3A3*
* `DX`,`DY` and `DZ` are the voxel dimensions along `x` and `y` and `z`


 The function first evaluates the gradent of `X` multiplied by the diffusivity tensor; then it evaluates the divergence (`Div(S*grad(X))`, `S` being the conductivity tensor). 


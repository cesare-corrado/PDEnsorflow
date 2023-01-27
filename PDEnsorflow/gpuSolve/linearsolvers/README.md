# Linear Solvers

This sub-module of `gpuSolve` defines the linear solvers to solve linear problems (inversion of the matrices).
It contains:

* `conjGrad`: conjugate gradient



## ConjGrad
A Class to solve linear systems with the conjugate gradient (CG) method. 
This method works **OLNY** if the matrix is symmetric positive definite.
Member functions:

* `set_maxiter(maxit)`: sets the maximum number of iterations to *maxit*
* `set_toll(toll)`: sets the tolerance on the residual used to determine the convergence to *toll*
* `set_matrix(Amat)`: assigns the sparse matrix that defines the linear system to the solver
* `set_RHS(RHS)`: assigns  the righ-hand side of the linear problem to the solver
* `set_X0(X0)`: assigns the inital guess X0 to the solver
* `maxiter()`: returns the maximum number of iterations
* `toll()`: returns the tolerance on the residual used to determine the convergence
* `matrix()`: returns the sparse matrix that defines the linear system
* `RHS()`: returns the right-hand side of the linear problem
* `X()`: returns the solution/initial value
* `verbose()`: returns the verbosity flag
* `summary()`: prints info on the solver convergence
* `solve()` solves the linear system using the conjugate gradients

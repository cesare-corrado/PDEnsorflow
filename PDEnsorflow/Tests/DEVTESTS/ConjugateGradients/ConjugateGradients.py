#!/usr/bin/env python
"""
    Conjugate Gradient Temporal Profiling Test

    This script measures the time required to solve a linear system
    A x = b with the ConjGrad solver (Jacobi-preconditioned).
    The matrix is assembled as A = alpha * M + beta * K using the
    finite element mass (M) and stiffness (K) matrices.

    A gold-truth solution x_star is prescribed, then b = A x_star is
    used as RHS. Initial guess is zero. The solver is run several times
    and the mean and standard deviation of the elapsed time are
    reported, together with the number of CG iterations and the
    relative residual ||A x - b|| / ||b||.

    Copyright 2022-2023 Cesare Corrado (cesare.corrado@kcl.ac.uk)
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# XLA is allowed. Requires ptxas >= 11.1 on PATH; this
# env ships ptxas 11.8 via conda (`cuda-nvcc`). 
# Set TF_XLA_FLAGS=--tf_xla_auto_jit=0 from the shell if you want to disable XLA for a comparison run.
# Ensure the in-tree gpuSolve package is imported before any system-wide install,
# unless the user overrides via GPUSOLVE_USE_SYSPATH=1 (useful for comparing
# against a pre-installed version of the package).
if os.environ.get('GPUSOLVE_USE_SYSPATH', '0') != '1':
    _PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
import argparse
import numpy as np
import time
import tensorflow as tf
tf.config.run_functions_eagerly(os.environ.get('GPUSOLVE_EAGER', '0') == '1')
if(tf.config.list_physical_devices('GPU')):
    print('GPU device')
else:
    print('CPU device')
print('Tensorflow version is: {0}'.format(tf.__version__))


from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict, csr_axpby
from gpuSolve.linearsolvers.conjgrad import ConjGrad
from gpuSolve.linearsolvers.jacobi_precond import JacobiPrecond


def dfmass(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties):
    """ empty function for mass properties"""
    return(None)


def sigmaTens(elemtype: str, iElem: int, domain: Triangulation, matprop: MaterialProperties) -> np.ndarray:
    """ function to evaluate the diffusion tensor (isotropic unit tensor)"""
    fib   = domain.Fibres()[iElem, :]
    rID   = domain.Elems()[elemtype][iElem, -1]
    sigma_l = matprop.ElementProperty('sigma_l', elemtype, iElem, rID)
    sigma_t = matprop.ElementProperty('sigma_t', elemtype, iElem, rID)
    Sigma = sigma_t * np.eye(3)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii, jj] = Sigma[ii, jj] + (sigma_l - sigma_t) * fib[ii] * fib[jj]
    return(Sigma)


def build_linear_system(mesh_file: str, alpha: float, beta: float):
    """
    Assembles A = alpha * M + beta * K on the given mesh.
    Returns (A, MASS, STIFFNESS, npt).
    """
    diffusl = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    diffust = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Domain = Triangulation()
    Domain.readMesh(mesh_file)
    npt  = Domain.Pts().shape[0]
    nels = 0
    for elemtype, Elements in Domain.Elems().items():
        nels += Elements.shape[0]
    print('Mesh: {} nodes, {} elements'.format(npt, nels))

    materials = MaterialProperties()
    materials.add_element_property('sigma_l', 'region', diffusl)
    materials.add_element_property('sigma_t', 'region', diffust)
    materials.add_ud_function('mass', dfmass)
    materials.add_ud_function('stiffness', sigmaTens)

    connectivity = Domain.mesh_connectivity(False)
    pattern      = compute_coo_pattern(connectivity)

    lmatr = {'mass': localMass, 'stiffness': localStiffness}
    MATRICES = assemble_matrices_dict(lmatr, pattern, Domain, materials, connectivity)
    MASS      = MATRICES['mass']
    STIFFNESS = MATRICES['stiffness']

    # A = alpha * M + beta * K (CSR-native; assemble_matrices_dict now
    # returns CSRSparseMatrix instances).
    A = csr_axpby(MASS, alpha, STIFFNESS, beta)
    materials.remove_all_element_properties()
    Domain.release_connectivity()
    return A, MASS, STIFFNESS, npt


def relative_residual(A, X: tf.Tensor, B: tf.Tensor) -> float:
    """ Computes ||A X - B|| / ||B||. A is a CSRSparseMatrix."""
    AX = tf.raw_ops.SparseMatrixMatMul(a=A._matrix, b=X)
    num = tf.sqrt(tf.reduce_sum(tf.square(AX - B)))
    den = tf.sqrt(tf.reduce_sum(tf.square(B)))
    return float(num.numpy() / den.numpy())


def profile_cg(mesh_file: str, alpha: float, beta: float, nrep: int,
               maxiter: int, toll: float, use_graph_loop: bool = False) -> dict:
    """
    Build A = alpha*M + beta*K on mesh_file, prescribe a gold solution,
    build the RHS, then repeat CG nrep times and record statistics.
    """
    A, _MASS, _STIFFNESS, npt = build_linear_system(mesh_file, alpha, beta)
    # A is a CSRSparseMatrix; extract COO once for stats + precond.
    A_st = A.to_sparse_tensor()
    nnz = int(A_st.values.shape[0])
    print('Matrix: shape={}, nnz={}'.format(A_st.dense_shape.numpy(), nnz))

    # Gold truth: smooth function of the mesh index (deterministic and non-trivial)
    rng   = np.random.default_rng(0)
    Xstar = rng.standard_normal(size=(npt, 1)).astype(np.float32)
    Xstar_tf = tf.constant(Xstar, dtype=tf.float32)

    # Build RHS = A Xstar
    B = tf.raw_ops.SparseMatrixMatMul(a=A._matrix, b=Xstar_tf)

    # Build solver and preconditioner
    solver  = ConjGrad({'maxiter': maxiter, 'toll': toll, 'verbose': False,
                        'use_graph_loop': use_graph_loop})
    print('Solver path: {}'.format(
        'GPU-resident (_use_graph_loop=True)' if use_graph_loop
        else 'eager + batched sync (default)'))
    solver.set_matrix(A)
    precond = JacobiPrecond()
    precond.build_preconditioner(A_st.indices.numpy()[:, 0],
                                 A_st.indices.numpy()[:, 1],
                                 A_st.values.numpy(), int(A_st.dense_shape.numpy()[0]))
    solver.set_precond(precond)

    times    = []
    niters   = []
    relres   = []
    l2_err   = []

    X0_np = np.zeros(shape=(npt, 1), dtype=np.float32)
    for irep in range(nrep):
        print('--- Repetition {}/{} ---'.format(irep + 1, nrep))
        solver.set_X0(tf.constant(X0_np))
        solver.set_RHS(B)
        t0 = time.time()
        solver.solve()
        # Force completion before timing (GPU is async)
        _ = solver.X().numpy()
        elapsed = time.time() - t0
        times.append(elapsed)
        X_sol = solver.X()
        r_rel = relative_residual(A, X_sol, B)
        err   = float(tf.sqrt(tf.reduce_sum(tf.square(X_sol - Xstar_tf))).numpy() /
                      tf.sqrt(tf.reduce_sum(tf.square(Xstar_tf))).numpy())
        niters.append(int(solver._niters))
        relres.append(r_rel)
        l2_err.append(err)
        print('  elapsed: {:6.3f} s | iters: {} | ||Ax-b||/||b||: {:.3e} | ||x-x*||/||x*||: {:.3e}'
              .format(elapsed, niters[-1], r_rel, err))

    return {
        'npt':    npt,
        'nnz':    nnz,
        'times':  np.array(times),
        'niters': np.array(niters),
        'relres': np.array(relres),
        'l2_err': np.array(l2_err),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',   type=float, default=0.01)
    parser.add_argument('--beta',    type=float, default=1.0)
    parser.add_argument('--nrep',    type=int,   default=5)
    parser.add_argument('--maxiter', type=int,   default=5000)
    parser.add_argument('--toll',    type=float, default=1.0e-7)
    parser.add_argument('--mesh',    type=str,   default='both',
                        choices=['coarse', 'fine', 'both'])
    parser.add_argument('--graph-loop', action='store_true',
                        help='Use the GPU-resident _solve_graph path (tf.while_loop).')
    args = parser.parse_args()

    data_dir = os.path.join('..', '..', 'data')
    meshes = []
    if args.mesh in ('coarse', 'both'):
        meshes.append(('triangulated_square.pkl', 'Coarse mesh'))
    if args.mesh in ('fine', 'both'):
        meshes.append(('triangulated_square_fine_mm.pkl', 'Fine mesh'))

    print('=' * 60)
    print('Conjugate Gradient Temporal Profiling')
    print('  alpha = {:.4g}  beta = {:.4g}  maxiter = {}  toll = {:g}'
          .format(args.alpha, args.beta, args.maxiter, args.toll))
    print('=' * 60)

    for mesh_name, description in meshes:
        mesh_path = os.path.join(data_dir, mesh_name)
        print('\n' + '=' * 60)
        print('{}: {}'.format(description, mesh_name))
        print('=' * 60)
        res = profile_cg(mesh_path, args.alpha, args.beta,
                         args.nrep, args.maxiter, args.toll,
                         use_graph_loop=args.graph_loop)
        t = res['times']
        print('\nResults for {}:'.format(description))
        print('  Nodes: {}   nnz: {}'.format(res['npt'], res['nnz']))
        print('  Repetitions: {}'.format(t.shape[0]))
        print('  Solve times (s): {}'.format(
            ', '.join(['{:.3f}'.format(v) for v in t])))
        print('  Mean: {:6.3f} s   Std: {:6.3f} s'.format(t.mean(), t.std()))
        print('  Mean iters: {:.1f}   Mean relres: {:.3e}   Mean rel.err: {:.3e}'
              .format(res['niters'].mean(), res['relres'].mean(), res['l2_err'].mean()))
        # Exclude first (warmup) repetition for steady-state statistics
        if t.shape[0] > 1:
            ts = t[1:]
            print('  Steady-state (excluding first): mean {:6.3f} s, std {:6.3f} s'
                  .format(ts.mean(), ts.std()))

    print('\n' + '=' * 60)
    print('Profiling complete.')
    print('=' * 60)

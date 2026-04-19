#!/usr/bin/env python
"""
    Matrix Assembly Temporal Profiling Test

    This script measures the time required to assemble the global
    stiffness and mass matrices using different input meshes.
    It repeats the assembly several times and provides mean and
    standard deviation of the elapsed time.

    Copyright 2022-2023 Cesare Corrado (cesare.corrado@kcl.ac.uk)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
import tensorflow as tf
tf.config.run_functions_eagerly(True)
if(tf.config.list_physical_devices('GPU')):
      print('GPU device' )
else:
      print('CPU device' )
print('Tensorflow version is: {0}'.format(tf.__version__))


from gpuSolve.entities.triangulation import Triangulation
from gpuSolve.entities.materialproperties import MaterialProperties
from gpuSolve.matrices.globalMatrices import compute_coo_pattern
from gpuSolve.matrices.localMass import localMass
from gpuSolve.matrices.localStiffness import localStiffness
from gpuSolve.matrices.globalMatrices import assemble_matrices_dict
from gpuSolve.matrices.globalMatrices import compute_reverse_cuthill_mckee_indexing


def dfmass(elemtype:str, iElem:int,domain:Triangulation,matprop:MaterialProperties):
    """ empty function for mass properties"""
    return(None)

def sigmaTens(elemtype:str, iElem:int,domain:Triangulation,matprop:MaterialProperties) -> np.ndarray :
    """ function to evaluate the diffusion tensor """
    fib   = domain.Fibres()[iElem,:]
    rID   = domain.Elems()[elemtype][iElem,-1]
    sigma_l = matprop.ElementProperty('sigma_l',elemtype,iElem,rID)
    sigma_t = matprop.ElementProperty('sigma_t',elemtype,iElem,rID)
    Sigma = sigma_t *np.eye(3)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii,jj] = Sigma[ii,jj]+ (sigma_l-sigma_t)*fib[ii]*fib[jj]
    return(Sigma)


def assemble_matrices(mesh_file, nrep=5):
    """
    Assembles mass and stiffness matrices from the given mesh file.
    Repeats the assembly nrep times and returns timing statistics.

    Input:
        mesh_file: path to the .pkl mesh file
        nrep:      number of repetitions for timing
    Output:
        times:     list of elapsed times for each repetition
        MATRICES:  the assembled matrices from the last repetition
    """
    diffusl = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}
    diffust = {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001}

    # Load the mesh once
    Domain = Triangulation()
    Domain.readMesh(mesh_file)
    npt  = Domain.Pts().shape[0]
    nels = 0
    for elemtype, Elements in Domain.Elems().items():
        nels += Elements.shape[0]
    print('Mesh: {} nodes, {} elements'.format(npt, nels))

    times = []
    MATRICES = None
    for irep in range(nrep):
        print('--- Repetition {}/{} ---'.format(irep+1, nrep))
        # Set up materials
        materials = MaterialProperties()
        materials.add_element_property('sigma_l','region',diffusl)
        materials.add_element_property('sigma_t','region',diffust)
        materials.add_ud_function('mass',dfmass)
        materials.add_ud_function('stiffness',sigmaTens)

        # Compute connectivity and pattern
        connectivity = Domain.mesh_connectivity(False)
        pattern      = compute_coo_pattern(connectivity)

        # Assembly
        lmatr = {'mass':localMass,'stiffness':localStiffness}
        t0 = time.time()
        MATRICES = assemble_matrices_dict(lmatr, pattern, Domain, materials, connectivity)
        elapsed = time.time() - t0
        times.append(elapsed)
        print('Assembly time: {:6.3f} s'.format(elapsed))

        # Clean up
        materials.remove_all_element_properties()
        Domain.release_connectivity()

    return times, MATRICES


if __name__=='__main__':
    nrep = 5
    data_dir = os.path.join('..','..','data')
    mesh_files = [
        ('triangulated_square.pkl', 'Coarse mesh'),
        ('triangulated_square_fine_mm.pkl', 'Fine mesh'),
    ]

    print('='*60)
    print('Matrix Assembly Temporal Profiling')
    print('='*60)

    for mesh_name, description in mesh_files:
        mesh_path = os.path.join(data_dir, mesh_name)
        print('\n' + '='*60)
        print('{}: {}'.format(description, mesh_name))
        print('='*60)

        times, MATRICES = assemble_matrices(mesh_path, nrep=nrep)
        times_arr = np.array(times)
        print('\nResults for {}:'.format(description))
        print('  Number of repetitions: {}'.format(nrep))
        print('  Assembly times (s): {}'.format(
            ', '.join(['{:.3f}'.format(t) for t in times])))
        print('  Mean:  {:6.3f} s'.format(times_arr.mean()))
        print('  Std:   {:6.3f} s'.format(times_arr.std()))

        # Print matrix info (CSRSparseMatrix -> convert to COO for stats only)
        for matr_name, M in MATRICES.items():
            st = M.to_sparse_tensor()
            print('  Matrix {}: shape={}, nnz={}'.format(
                matr_name, st.dense_shape.numpy(), st.values.shape[0]))

    print('\n' + '='*60)
    print('Profiling complete.')
    print('='*60)

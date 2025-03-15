import os
from simFrame.environment import Environment
import simFrame.permittivities as permittivities
from simFrame.remoteSolver.fdfd.utils import calculate_adjoint_sources, calculate_gradient_field
from simFrame.buildStructure import flipScaledPixelInCentralPlanar
import simFrame
import meanas
import numpy as np
import cupy as cp
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import yaml
import warnings
import random
import tempfile
import subprocess
from jax import config
import scipy.sparse as sp
import scipy
from meanas.fdmath.vectorization import vec, unvec
config.update("jax_enable_x64", True)
import zarr
import h5py
import cupyx.scipy.sparse as cpx_sparse  # Import CuPy sparse module
import multiprocessing as mp
import cupy.cuda
import multiprocessing.shared_memory as shm


def sp_sparse_to_cupy_sparse(scipy_sparse,device):
    # Convert to CSR format (for efficient GPU operations)
    scipy_csr = scipy_sparse.tocsr()
    with cp.cuda.Device(device):
        # Get CSR components (row pointers, column indices, values)
        crow_indices = cp.array(scipy_csr.indptr, dtype=cp.int32)  # Compressed row indices
        col_indices = cp.array(scipy_csr.indices, dtype=cp.int32)  # Column indices
        values = cp.array(scipy_csr.data, dtype=cp.complex64)      # Values

        # Create the CuPy sparse CSR matrix
        cupy_sparse = cpx_sparse.csr_matrix(
            (values, col_indices, crow_indices),
            shape=scipy_csr.shape,
            dtype=cp.complex64
        )
        return cupy_sparse

def numpy_to_cupy_gpu(np_matrix,device):
    # Ensure it's a CuPy array and convert to complex64
    with cp.cuda.Device(device):
        return cp.asarray(np_matrix, dtype=cp.complex64)

def cupy_sparse_size(matrix) -> int:
    total_bytes = (matrix.data.nbytes +
                    matrix.indices.nbytes +
                    matrix.indptr.nbytes)

    return total_bytes

def get_scipy_sparse_size(sparse_matrix):
    if not sp.issparse(sparse_matrix):
        raise ValueError("Input matrix must be a SciPy sparse matrix.")
    # Calculate total memory usage
    print("The matrix contains values of the type", sparse_matrix.data.dtype)
    total_bytes = total_bytes = sparse_matrix.data.nbytes + sparse_matrix.offsets.nbytes
    return total_bytes

def objective_function(P_0, P_1):
    return (P_0 - 0.7)**2 + (P_1 - 0.3)**2

def get_dA_dp_k(p_k,k,env,eps,fields):
    p_k_indices = divmod(p_k, env.designArea[0])
    eps_altered = flipScaledPixelInCentralPlanar(xy = p_k_indices,
                                    epsilon = env.epsilon.copy(),
                                    designArea = (env.designArea[0], env.designArea[1]),
                                    structureScalingFactor = env.structureScalingFactor,
                                    thickness = env.thickness,
                                    minPermittivity = env.surroundingPermittivity,
                                    maxPermittivity = env.structurePermittivity)

    eps_altered_interp = simFrame.remoteSolver.fdfd.utils.interpolate_eps(np.array(eps_altered), np.array(env.simManager.dxes, dtype=object))
    #eps_mask = np.where(vec(eps_altered_interp) == vec(eps), 0, 1)
    bool_mask = (eps_altered_interp != eps)
    bool_diagonal = (vec(eps_altered_interp == eps)).astype(np.uint8)  # Convert to small int (or keep as bool)
    sparse_diag = sp.diags(bool_diagonal, format="dia",dtype = np.uint8)
    return sparse_diag


def create_dA_dp_dict(k, env, eps,fields,verbose=False):
    """
    Creates a dictionary mapping each p_k to its corresponding sparse matrix dA_dp_k.
    """
    
    dA_dp_dict = {}  # Initialize the dictionary to store matrices

    for p_k in k:
        if (p_k % 100) == 0:
            print("Computing dA_dp_", p_k)

        dA_dp_dict[p_k] = get_dA_dp_k(p_k,k,env,eps,fields)

    return dA_dp_dict


def get_power(env,fields):
    P_vals = []
    for mode in env.simManager.target_modes:
        overlap = np.abs(np.sum(vec(fields) @ vec(mode['overlap_operator']).conj()))**2
        P_vals.append(overlap)
    return P_vals

def extract_nonzero_values(A, B):
    return A[B != 0]  # Mask A using the condition "B != 0"


def get_second_derivative_non_mixed(env, ratio, fields, second_derivative0, second_derivative1):
    power_mode = 0
    ratios = [ratio, 1 - ratio]
    length = len(meanas.fdmath.vectorization.vec(fields))  # Ensure this is correctly defined
    second_derivatives = [second_derivative0, second_derivative1]  # Store references to modify them
    for mode in env.simManager.target_modes:
        E_out = extract_nonzero_values(vec(mode['overlap_operator']), vec(mode['overlap_operator']))
        print(f"The shape of E_out is {E_out.shape}")

        inner_product = np.sum(extract_nonzero_values(vec(fields).conj(), vec(mode['overlap_operator'])) @ E_out)
        # Modify the preallocated array in place
        np.copyto(second_derivatives[power_mode], inner_product**2 * np.outer(E_out.conj(), E_out.conj()).astype(np.complex64))

        print("The dimensions of second_derivative are", second_derivatives[power_mode].shape)
        power_mode += 1

def get_second_derivative_mixed(env,ratio,fields,powers,second_derivative0, second_derivative1):
    power_mode = 0
    ratios = [ratio, 1 - ratio]
    length = len(meanas.fdmath.vectorization.vec(fields))  # Ensure this is correctly defined
    second_derivatives = [second_derivative0, second_derivative1]  # Store references to modify them
    for mode in env.simManager.target_modes:
        E_out = extract_nonzero_values(vec(mode['overlap_operator']), vec(mode['overlap_operator']))
        print(f"The shape of E_out is {E_out.shape}")
        # Modify the preallocated array in place
        np.copyto(second_derivatives[power_mode], (2*powers[power_mode] - ratios[power_mode] ) * np.outer(E_out,E_out.conj()).astype(np.complex64))

        print("The dimensions of second_derivative are", second_derivatives[power_mode].shape)
        power_mode += 1

def matrix_symmetrizer(matrix):
    return matrix + matrix.T - np.diag(np.diag(matrix))

def get_modes(env):
    power_mode = 0
    modes = {}
    for mode in env.simManager.target_modes:
        modes[power_mode] = (vec(mode['overlap_operator']) != 0)
        power_mode += 1
    return modes


def load_de_dp_into_shared_memory(hf_path, k):
    """Loads de_dp data from an HDF5 file into shared memory."""
    shared_memory_dict = {}  # Dictionary to store shared memory metadata

    with h5py.File(hf_path, "r") as hf:
        for p_k in k:
            data = hf[f"de_dp_{p_k}"][:]  # Read from HDF5
            shape = data.shape
            dtype = data.dtype
            
            # Allocate shared memory
            size = np.prod(shape) * data.itemsize
            shm_block = shm.SharedMemory(create=True, size=size)

            # Store the data in shared memory
            shared_array = np.ndarray(shape, dtype=dtype, buffer=shm_block.buf)
            np.copyto(shared_array, data)  # Copy once into shared memory

            # Store metadata (name, shape, dtype) for worker reconstruction
            shared_memory_dict[p_k] = {
                "shm_name": shm_block.name,
                "shape": shape,
                "dtype": dtype.name  # Store dtype as a string for reconstruction
            }

            print(f"Loaded de_dp_{p_k} into shared memory: {shm_block.name}")

    return shared_memory_dict  # Return metadata instead of actual arrays

def allocate_shared_matrix(shape, dtype=np.complex64):
    """ Creates shared memory and returns a numpy array backed by it. """
    size = np.prod(shape) * np.dtype(dtype).itemsize
    shared_mem = shm.SharedMemory(create=True, size=size)
    matrix = np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)
    return shared_mem, matrix


def compute_full_second_derivative(Block1, Block2, k, dA_dp, hf_path,shm_adjointField,shm_Eout0,shm_Eout1,shm_eField,
                                   powers, ratios, shm_secondDerivative, device, de_dp_dict, shm_adjoint, shm_hessian,shape_field):
    cp.cuda.Device(device).use()  # Assign this process to a GPU
    stream = cp.cuda.Stream(non_blocking=True)  # Create a CUDA stream
    time_round = time.time()
    with stream:
        second_derivative_mixed = {}

        #Recreate matrices to write to
        shm_secondDerivative = shm.SharedMemory(name=shm_secondDerivative)
        secondDerivative = np.ndarray((len(k), len(k)), dtype=np.float32, buffer=shm_secondDerivative.buf)
        shm_adjoint = shm.SharedMemory(name=shm_adjoint)
        secondDerivativeAdjoint = np.ndarray((len(k), len(k)), dtype=np.float32, buffer=shm_adjoint.buf)
        shm_hessian = shm.SharedMemory(name=shm_hessian)
        secondDerivativeHessian = np.ndarray((len(k), len(k)), dtype=np.float32, buffer=shm_hessian.buf)
        
        #Recreate fields
        shm_adjointField= shm.SharedMemory(name=shm_adjointField)
        adjoint_field = numpy_to_cupy_gpu(np.ndarray(shape_field, dtype=np.complex64, buffer=shm_adjointField.buf), device)
        shm_eField= shm.SharedMemory(name=shm_eField)
        eField = numpy_to_cupy_gpu(np.ndarray(shape_field, dtype=np.complex64, buffer=shm_eField.buf), device)
        shm_Eout0= shm.SharedMemory(name=shm_Eout0)
        E_out0 = numpy_to_cupy_gpu(np.ndarray(shape_field, dtype=np.complex64, buffer=shm_Eout0.buf), device)
        shm_Eout1= shm.SharedMemory(name=shm_Eout1)
        E_out1 = numpy_to_cupy_gpu(np.ndarray(shape_field, dtype=np.complex64, buffer=shm_Eout1.buf), device)

        '''# Attach to shared memory inside worker
        shm_mixed0 = shm.SharedMemory(name=shm_mixed0_name)
        shm_mixed1 = shm.SharedMemory(name=shm_mixed1_name)
        shm_non_mixed0 = shm.SharedMemory(name=shm_non_mixed0_name)
        shm_non_mixed1 = shm.SharedMemory(name=shm_non_mixed1_name)


        # Reconstruct NumPy arrays from shared memory
        second_derivative_matrix_mixed0 = numpy_to_cupy_gpu(np.ndarray((16800, 16800), dtype=np.complex64, buffer=shm_mixed0.buf), device)
        second_derivative_matrix_mixed1 = numpy_to_cupy_gpu(np.ndarray((16500, 16500), dtype=np.complex64, buffer=shm_mixed1.buf), device)
        second_derivative_matrix_non_mixed0 = numpy_to_cupy_gpu(np.ndarray((16800, 16800), dtype=np.complex64, buffer=shm_non_mixed0.buf), device)
        second_derivative_matrix_non_mixed1 = numpy_to_cupy_gpu(np.ndarray((16500, 16500), dtype=np.complex64, buffer=shm_non_mixed1.buf), device)'''

        de_dp_GPU = {}
        dA_dp_GPU = {}

        for p_k in Block1:
            #Same for de_dp_k
            if p_k not in de_dp_GPU:
                metadata = de_dp_dict[p_k]
                shm_block = shm.SharedMemory(name=metadata["shm_name"])
                de_dp_GPU[p_k] = numpy_to_cupy_gpu(np.ndarray(metadata["shape"], dtype=np.dtype(metadata["dtype"]), buffer=shm_block.buf), device)
                shm_block.close()

            de_dp_k = de_dp_GPU[p_k]

            if p_k not in dA_dp_GPU:
                dA_dp_GPU[p_k] = sp_sparse_to_cupy_sparse(dA_dp[p_k],device)
            dA_dp_k = dA_dp_GPU[p_k]



            for p_l in Block2:
                if p_l <= p_k:
                    #Same for de_dp_l
                    if p_l not in de_dp_GPU:
                        metadata = de_dp_dict[p_l]
                        shm_block = shm.SharedMemory(name=metadata["shm_name"])
                        de_dp_GPU[p_l] = numpy_to_cupy_gpu(np.ndarray(metadata["shape"], dtype=np.dtype(metadata["dtype"]), buffer=shm_block.buf), device)
                        shm_block.close()
                    
                    de_dp_l = de_dp_GPU[p_l]

                    '''de_dp_l_red0 = extract_nonzero_values(de_dp_l,mode0)
                    de_dp_l_red1 = extract_nonzero_values(de_dp_l,mode1)'''

                    if p_l not in dA_dp_GPU:
                        dA_dp_GPU[p_l] = sp_sparse_to_cupy_sparse(dA_dp[p_l],device)
                    dA_dp_l = dA_dp_GPU[p_l]

                    #print("The adjoint field has shape", adjoint_field.shape)
                    #print("DA_dp @ de_dp_l has shape", (dA_dp_k@ de_dp_l).shape)
                    #print("de_dp_l has shape", de_dp_l.shape)
                    #print("E_out0 has shape", E_out0.shape)
                    #time_adjoint = time.time()
                    adjoint_term = 2*cp.sum(cp.dot(adjoint_field.T, dA_dp_k@ de_dp_l + dA_dp_l@ de_dp_k))
                    
                    #print("Calculating the adjoint term took", time.time() - time_adjoint)
                    #print("The shape of the adjoint term is ", adjoint_term.shape)
                    
                    #time_hessian = time.time()
                    '''second_order_mixed = 4*(de_dp_l_red0.T.conj() @ (second_derivative_matrix_mixed0 @ de_dp_k_red0) + 
                                            de_dp_l_red1.T.conj() @ (second_derivative_matrix_mixed1 @ de_dp_k_red1))
                    second_order_non_mixed = 4*(de_dp_l_red0.T @ (second_derivative_matrix_non_mixed0 @ de_dp_k_red0) + 
                                            de_dp_l_red1.T @ (second_derivative_matrix_non_mixed1 @ de_dp_k_red1))'''
                    
                    #mixed term:
                    second_order_mixed = 2*((2*powers[0] - ratios[0] ) * (cp.dot(de_dp_l.conj().T,E_out0) * cp.dot(E_out0.conj().T, de_dp_k))
                            + (2*powers[1] - ratios[1] ) *(cp.dot(de_dp_l.conj().T,E_out1) * cp.dot(E_out1.conj().T, de_dp_k))).astype(np.complex64)
                    
                    #non-mixed term:
                    second_order_non_mixed = 2*(cp.dot(eField.T.conj(), E_out0)**2 * cp.dot(de_dp_l.T,E_out0.conj()) * cp.dot(de_dp_k.T,E_out0.conj())
                                            + cp.dot(eField.T.conj(), E_out1)**2 * cp.dot(de_dp_l.T,E_out1.conj()) * cp.dot(de_dp_k.T,E_out1.conj())).astype(np.complex64)

                    #print("Computing hessian terms took", time.time()-time_hessian)
                    secondDerivative[p_k,p_l] = adjoint_term.item().real + second_order_mixed.item().real + second_order_non_mixed.item().real
                    #print("k is", p_k, ", l is", p_l,", second derivative is",secondDerivative[p_k,p_l])

                    secondDerivativeAdjoint[p_k,p_l] = adjoint_term.item().real
                    secondDerivativeHessian[p_k,p_l] = second_order_mixed.item().real + second_order_non_mixed.item().real

                # Delete de_dp_l from GPU memory and free up space
                #del de_dp_l
                #del de_dp_l_red0
                #del de_dp_l_red1
                #del dA_dp_l
                #cp.get_default_memory_pool().free_all_blocks()

        stream.synchronize()  # Ensure all operations are completed'''

        shm_secondDerivative.close()
        shm_adjoint.close()
        shm_hessian.close()
        shm_adjointField.close()
        shm_Eout0.close()
        shm_Eout1.close()
        shm_eField.close()
        #shm_mixed0.close()
        #shm_mixed1.close()
        #shm_non_mixed0.close()
        #shm_non_mixed1.close()
        
        #Free memory
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        

    print(f"Process for last p_k={p_k} and last p_l={p_l} completed on GPU {device}. It took {time.time() - time_round}s")
    cp.get_default_memory_pool().free_all_blocks()  # Free unused GPU memory
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return p_k


def main():
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None
        def construct_python_tuple(self, node):
            return tuple(self.construct_sequence(node))

    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)
    SafeLoaderIgnoreUnknown.add_constructor(
        u'tag:yaml.org,2002:python/tuple',
        SafeLoaderIgnoreUnknown.construct_python_tuple)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    #custom FOM, this one just prints figures
    def figureOfMerit(overlaps, powers):
        print("overlaps:", overlaps)
        print("powers:", powers)
        return 0

    dimension = "3D"

    if dimension == "3D":
        config_path = "PS7030_3D.yml"
        cwd = "/scratch/tmp/jklemmen"
        zarr_filename = os.path.join(cwd, "de_dp_3D.h5")
        structure_path = "result_structure_3d.csv"
    elif dimension == "2D":
        config_path = "PS7030_2D.yml"
        cwd = os.getcwd()
        zarr_filename = os.path.join(cwd, "de_dp.h5")
        structure_path = "result_structure_2d.csv"
    else:
        print("No dimension was given")


    #config_path = "PS7030_3D.yml"
    ratio = 0.7
    ratios = [ratio, 1 - ratio]
    verbose = True

    #cwd = os.getcwd()
    #cwd = "/scratch/tmp/jklemmen"
    #zarr_filename = os.path.join(cwd, "de_dp_3D.h5")

    with open(config_path) as file:
        full_dict = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)
        envdict = full_dict.get("environment", full_dict.get("env_config", {}).get("environment", {}))
        env = Environment(**envdict)
        #set FOM
        env.setFOM(figureOfMerit)
        structure = np.genfromtxt(structure_path, delimiter=',')
        env.setStructure(structure.reshape(envdict["designArea"][0], envdict["designArea"][1]))
        #get interpolated epsilon from env.espilonm which is not interpolated
        #use this interpolated epsilon for actual simulations
        if env.epsilon_weighting == "volumetric":
            eps = simFrame.remoteSolver.fdfd.utils.interpolate_eps(np.array(env.epsilon), np.array(env.simManager.dxes, dtype=object))
        else:
            eps = np.array(env.epsilon)
        
        now = time.time()
        #calculate E:
        fom, fields, overlaps, powers, iterations_needed = env.evaluate(method="fdfd", E_initial=None, solver=envdict["solver_configuration"]["default_solver"])
        if verbose:
            print("simulating E field took", time.time()-now, "s")

        #calculate adjoint sources
        adjoint_sources = calculate_adjoint_sources(fields[0], env, 0, objective_function)
            
        adjoint_field = np.zeros(dtype="complex128", shape=[3,env.dimensions[0],env.dimensions[1],env.dimensions[2]])
        #calculate gradient (this will conduct a simulation)
        gradient_field, adjoint_field = calculate_gradient_field(env, 0, 0, adjoint_sources, fields[0],adjoint_field)
        adjoint_field = vec(adjoint_field)
        #Replace later once you find the error of the adjoint field
        #adjoint_field = numpy_to_cupy_gpu(np.matrix(meanas.fdmath.vectorization.vec(fields)).T)
        print("The shape of the adjoint field is ", adjoint_field.shape)

    #k = range(0, 500)
    k = [i for i in range(env.designArea[0]*env.designArea[1]-1)]
    k_final = [i for i in range(env.designArea[0]*env.designArea[1]-1)]
    length_field = len(adjoint_field)

    manager = mp.Manager()

    de_dp_dict = load_de_dp_into_shared_memory(zarr_filename, k)

    #dA_dp = {p_k: sp_sparse_to_cupy_sparse(sp.diags(np.random.rand(10))) for p_k in k}
    dA_dp = create_dA_dp_dict(k, env, eps,fields,verbose=False)
    print("The size of dA_dp at 0 is", get_scipy_sparse_size(dA_dp[0]))
    print("dA_dp finished")

    powers = get_power(env,fields)
    #powers = 0
    print("Powers are", powers)

    #Allocate memory for fields
    shape_field = adjoint_field.reshape(-1,1).shape
    shm_adjointField, adjointField = allocate_shared_matrix(shape_field, dtype=np.complex64)
    shm_eField, eField = allocate_shared_matrix(shape_field, dtype=np.complex64)
    shm_Eout0, Eout0 = allocate_shared_matrix(shape_field, dtype=np.complex64)
    shm_Eout1, Eout1 = allocate_shared_matrix(shape_field, dtype=np.complex64)

    #Assign fields
    np.copyto(adjointField, adjoint_field.reshape(-1,1).astype(np.complex64))
    np.copyto(eField, vec(fields).reshape(-1,1).astype(np.complex64))
    power_mode = 0
    E_out = {}
    for mode in env.simManager.target_modes:
        E_out[power_mode] = vec(mode['overlap_operator']).reshape(-1,1)
        power_mode += 1
    np.copyto(Eout0, E_out[0].astype(np.complex64))
    np.copyto(Eout1, E_out[1].astype(np.complex64))
    print("Succesfully transferred the fields to shared memory")

    '''shape0 = (16800,16800)
    shape1 = (16500,16500)

    # Allocate shared memory
    shm_mixed0, second_derivative_mixed0 = allocate_shared_matrix(shape0, dtype=np.complex64)
    shm_mixed1, second_derivative_mixed1 = allocate_shared_matrix(shape1, dtype=np.complex64)
    shm_non_mixed0, second_derivative_non_mixed0 = allocate_shared_matrix(shape0, dtype=np.complex64)
    shm_non_mixed1, second_derivative_non_mixed1 = allocate_shared_matrix(shape1, dtype=np.complex64)

    # Compute derivatives and store in shared memory
    get_second_derivative_mixed(env, ratio, fields, powers, second_derivative_mixed0,second_derivative_mixed1)
    get_second_derivative_non_mixed(env, ratio, fields, second_derivative_non_mixed0,second_derivative_non_mixed1)

    modes = get_modes(env)'''

    #second_derivative_non_mixed0 = np.zeros((len(k), len(k)), dtype=np.float32)
    #second_derivative_non_mixed1 = np.zeros((len(k), len(k)), dtype=np.float32)
    #second_derivative_mixed0 = np.zeros((len(k), len(k)), dtype=np.float32)
    #second_derivative_mixed1 = np.zeros((len(k), len(k)), dtype=np.float32)
    
    shm_secondDerivative, secondDerivative = allocate_shared_matrix((len(k), len(k)), dtype=np.float32)
    shm_adjoint, adjointMatrix = allocate_shared_matrix((len(k), len(k)), dtype=np.float32)
    shm_hessian, hessianMatrix = allocate_shared_matrix((len(k), len(k)), dtype=np.float32)

    num_gpus = cp.cuda.runtime.getDeviceCount()
    devices = print("Number of available GPU devices is", num_gpus)
    devices = list(range(num_gpus))
    processes = []

    total_time = time.time()
    
    # Multiprocessing setup
    mp.set_start_method("spawn", force=True)
    processes = []

    block_size = 250      # Maximum block size

    # Loop over k for Block1
    for start1 in range(0, len(k), block_size):  
        Block1 = k[start1 : min(start1 + block_size, len(k))]  # First block

        # Loop over k independently for Block2, ensuring Block2 ≤ Block1
        for start2 in range(0, start1 + 1, block_size):  # Ensures Block2 values are ≤ Block1
            Block2 = k[start2 : min(start2 + block_size, len(k))]  # Second block

            device = devices[(start1 // block_size + start2 // block_size) % num_gpus]  # Distribute across GPUs

            p = mp.Process(target=compute_full_second_derivative, args=(
                Block1, Block2, k, dA_dp, zarr_filename,shm_adjointField.name,shm_Eout0.name,shm_Eout1.name,shm_eField.name,  # Pass shared memory names
                        powers, ratios, shm_secondDerivative.name, device, de_dp_dict, shm_adjoint.name, shm_hessian.name,shape_field
            ))

            p.start()
            processes.append(p)
    print("Go!")
    for p in processes:
        p.join()

    print("Parallel computation completed!")
    print("Finished!. The total time was", time.time() - total_time)
    print(secondDerivative)

    # Get available RAM in bytes and convert to GB
    os.system("free -h")


    if dimension == "3D":
        np.savez('SecondDerivative3D70_30.npz', matrix=matrix_symmetrizer(secondDerivative))
        np.savez('SecondDerivative3D70_30_hessian.npz', matrix=matrix_symmetrizer(adjointMatrix))
        np.savez('SecondDerivative3D70_30_adjoint.npz', matrix=matrix_symmetrizer(hessianMatrix))
    elif dimension == "2D":
        np.savez('SecondDerivative2D70_30.npz', matrix=matrix_symmetrizer(secondDerivative))
        np.savez('SecondDerivative2D70_30_hessian.npz', matrix=matrix_symmetrizer(adjointMatrix))
        np.savez('SecondDerivative2D70_30_adjoint.npz', matrix=matrix_symmetrizer(hessianMatrix))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensure correct multiprocessin
    main()
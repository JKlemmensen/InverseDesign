import os
from simFrame.environment import Environment
import simFrame.permittivities as permittivities
from simFrame.buildStructure import flipScaledPixelInCentralPlanar
import simFrame
import meanas
import numpy as np
import os
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

import fast_matrix_market

import h5py

'''
def objective_function_1(P_0, P_1):
             return (P_0 - 0.7)**2 + (P_1 - 0.3)**2
'''

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

config_path = "PS7030_2D.yml"
verbose = False #for verbose output of the solver and timing

plots = False

cwd = os.getcwd()

hdf5_filename = os.path.join(cwd, "de_dp.h5")

with open(config_path) as file:
    print(config_path)
    full_dict = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)
    try:
        envdict = full_dict["environment"]
    except KeyError:
        envdict = full_dict["env_config"]["environment"]

    print("Loaded", config_path, ":")
    print(envdict)

    env = Environment(**envdict)

    #set FOM
    env.setFOM(figureOfMerit)

    #set custom structure;
    structure = np.genfromtxt("result_structure_2d.csv",delimiter=',')
    env.setStructure(structure.reshape(envdict["designArea"][0],envdict["designArea"][1]))

    #set uniform permittivity:
    #env.setStructure(np.ones(np.prod(envdict["designArea"])).reshape(envdict["designArea"][0],envdict["designArea"][1])*1.00)

    now = time.time()

    #calculate E:
    fom, fields, overlaps, powers, iterations_needed = env.evaluate(method="fdfd", E_initial=None, solver=envdict["solver_configuration"]["default_solver"])
    if verbose:
        print("simulating E field took", time.time()-now, "s")

    with h5py.File("E_field.h5", "w") as hf:
        hf.create_dataset("E_field", data=fields[0])
        #hf.create_dataset("E_field_imag", data=np.imag(fields[0]))
        hf.attrs["description"] = "E-field data (real and imaginary parts) from FDFD simulation"
        print("E-field successfully saved to E_field.h5")


    '''
    #This is how you load the electric field again:
    with h5py.File("E_field.h5", "r") as hf:'
    E_field = hf["E_field"][:]
    description = hf.attrs["description"]

    #E_real = hf["E_field_real"][:]
    #E_imag = hf["E_field_imag"][:]
    #description = hf.attrs["description"]
    #print("Loaded E-field:", description)

    #Reconstruct the complex field
    #E_field = E_real + 1j * E_imag
    '''
    

    #get interpolated epsilon from env.espilonm which is not interpolated
    #use this interpolated epsilon for actual simulations
    if env.epsilon_weighting == "volumetric":
        eps = simFrame.remoteSolver.fdfd.utils.interpolate_eps(np.array(env.epsilon), np.array(env.simManager.dxes, dtype=object))
    else:
        eps = np.array(env.epsilon)


    if plots:
        plt.figure()
        plt.imshow(np.real(fields[0][2 if env.simManager.is2D else 1,:,:,env.dimensions[2]//2]))
        plt.savefig("forward_field"+".png")
        plt.close()

    #generate the system matrix A
    A = env.simManager._generate_A(0, vec(f=[eps[0][0],eps[0][1],eps[0][2]]))

    A_path = os.path.join(cwd, "A.mtx")
    with h5py.File(hdf5_filename, "w") as hf:   #Create new hdf5 file to save de_dp
        print(f"Created new {hdf5_filename}")

    #whole loop time:
    whole_loop_time = time.time()
    #we now loop through all p_k
    #####for demonstration, I just pick 10 random p_k (I'll do it for all beause I was curious about the time reqirements :P)
    k = [i for i in range(env.designArea[0]*env.designArea[1]-1)] #[random.randint(0, env.designArea[0]*env.designArea[1]-1) for _ in range(10)]
    for p_k in k:
        if verbose:
            print("setting k = ", p_k)
        #construct epsilon where epsilon != 0 only at points influenced by pixel flip at p_k
        p_k_indices = divmod(p_k, env.designArea[0])
        eps_altered = flipScaledPixelInCentralPlanar(xy = p_k_indices,
                                        epsilon = env.epsilon.copy(),
                                        designArea = (env.designArea[0], env.designArea[1]),
                                        structureScalingFactor = env.structureScalingFactor,
                                        thickness = env.thickness,
                                        minPermittivity = env.surroundingPermittivity,
                                        maxPermittivity = env.structurePermittivity)

        eps_altered_interp = simFrame.remoteSolver.fdfd.utils.interpolate_eps(np.array(eps_altered), np.array(env.simManager.dxes, dtype=object))
        eps_mask = np.where(vec(eps_altered_interp) == vec(eps), 0, 1)

        dA_dp = sp.diags(eps_mask)

        #solve for de_dp

        #A * de_dp = dA_dp * e
        now = time.time()
        dA_dp_e = dA_dp * vec(fields[0])
        if verbose:
            print("calculation of dA_dp_e = dA_dp * e took", time.time()-now, "s")
            #note: if this is too long, it might be worth parallelizing this step

        # Default initial guess (zeros)
        de_dp_initial = np.zeros(eps_mask.shape)

        # Check if k-1 exists in the HDF5 file and load it. Change later to make it more efficient
        loadPrevVector = True
        if loadPrevVector:
            if p_k > 0:  # Ensure k-1 is a valid index
                with h5py.File(hdf5_filename, "r") as hf:
                    prev_key = f"de_dp_{p_k-1}"
                    if prev_key in hf:
                        de_dp_initial = hf[prev_key][:].ravel()
                        #print(f"Loaded initial guess for k={p_k} from {prev_key}")
            else:
                print(f"No previous de_dp_{p_k-1} found, using zeros as initial guess")


        #determine simframe dir to access standalone magma solver
        simFrame_dir = os.path.dirname(simFrame.__file__)
        temp_dir = tempfile.gettempdir()

        """
        write Matrices and vectors to temporary directory
        they will be read by the standalone solver
        if you want to parallelize the caclulation of all de_dpk, you have to keep in mind not to override these matrices before they are read from independent
        solver instances. I suggest to use unique names then.
        process-based parallelization makes sense here,
        Consider that writing the matrices to disk takes a lot of time. It would therefore be smarter to write A (which by far is the biggest amoung them)
        outside of this loop and do not write the same A for all k over and over again.
        However, it's done here for demonstration purposes (and because I thought about that after writing these docs)
        """

        de_dp_initial_path = temp_dir + "/de_dp_initial.mtx"
        dA_dp_e_path = temp_dir + "/dA_dp_e.mtx"
        result_path = temp_dir + "/de_dp.mtx"

        #configuration of the standalone solver
        max_iterations = 50000
        relative_target_residual = 0.0001
        gmres_krylov_dim = 30

        now = time.time()
        fast_matrix_market.mmwrite(A_path, A)
        fast_matrix_market.mmwrite(de_dp_initial_path, np.expand_dims(de_dp_initial, axis=1))
        fast_matrix_market.mmwrite(dA_dp_e_path, np.expand_dims(dA_dp_e, axis=1))
        if verbose:
            print("writing matrices took", time.time()-now, "s")

        now = time.time()
        # Construct the relative path to the executable
        command_path = os.path.join(simFrame_dir, "remoteSolver/fdfd/standalone_ginkgo/fdfdGinkgoStandalone")
        command = [
                    command_path,
                    A_path,
                    dA_dp_e_path,
                    de_dp_initial_path,
                    result_path,
                    str(max_iterations),
                    str(relative_target_residual),
                    "true" if verbose else "false",
                    str(gmres_krylov_dim)
                ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if verbose:
            print("simulation of de_dpk took", time.time()-now, "s")

        if verbose:
            print("Command output:", result.stdout)

        de_dp = np.array(fast_matrix_market.mmread(result_path))

        with h5py.File(hdf5_filename, "a") as hf:
            # Create a new dataset indexed by k in the loop
            dataset_name = f"de_dp_{p_k}"
            hf.create_dataset(dataset_name, data=de_dp, compression="gzip")
            hf.attrs[f"description_{p_k}"] = f"de_dp for k={p_k}"
            #print(f"Saved de_dp for k={p_k} to {hdf5_filename}")

        '''
        #How to loead a specific index later in the de_dp hdf5
        with h5py.File("de_dp.h5", "r") as hf:
            de_dp_k = hf[f"de_dp_{some_k}"][:]  # Replace `some_k` with desired index
            print(f"Loaded de_dp for k={some_k}: shape {de_dp_k.shape}")
        '''

        if p_k%10 == 0:
            print(p_k)
        if plots:
            #visulization of result:
            plt.figure()
            plt.imshow(np.real(unvec(v = de_dp, shape=(env.dimensions[0], env.dimensions[1], env.dimensions[2]))[2 if env.simManager.is2D else 1,:,:,env.dimensions[2]//2]))
            plt.savefig("de_dp_flipIndex_"+str(p_k)+".png")
            plt.close()

    end_of_loop_time = time.time()
    print("total time for de_dp in this configurations:", time.time()-whole_loop_time)

"""Driver of the sampler fro x-ray, UV and LW.

Paramerers are gonna be:
wavelength -> possible options are 'X', 'UV' and 'LW' (uppercase)
z_init
z_end
z_number of steps
dlog10m
N_iter -> this is per process
n_process

sample_SFR
sample_emiss
sample_Poiss

R_bias -> default
log10_Mmin -> default
mass_binning -> default
"""

from sampler_x import Sampler_x
from sampler_uv import Sampler_UV
from sampler_lw import Sampler_LW
from sampler_all import Sampler_ALL
from bpass_read import bpass_loader
import os
import time
import sys
import numpy as np
from multiprocessing import Pool, cpu_count, Process, Manager
from save import HdF5Saver


if __name__=='__main__':
    time_start_run = time.time()
    wavelength = str(sys.argv[1])
    z_init = float(sys.argv[2])
    z_end = float(sys.argv[3])
    z_steps = int(sys.argv[4])
    dlog10m = float(sys.argv[5])
    N_iter = int(sys.argv[6])
    n_processes = int(sys.argv[7])

    sample_SFR = sys.argv[8]
    sample_emiss = sys.argv[9]
    sample_Poiss = sys.argv[10]

    R_bias = 5

    if sample_SFR in ["False", "FALSE", "false", "0", "No"]:
        sample_SFR = False
    else: 
        sample_SFR = True

    if sample_emiss in ["False", "FALSE", "false", "0", "No"]:
        sample_emiss = False
    else:
        sample_emiss = True

    if sample_Poiss in ["False", "FALSE", "false", "0", "No"]:
        sample_Poiss = False
    else:
        sample_Poiss = True
    
    assert type(sample_SFR) == bool, "Something went wrong, type is not boolean"

    time_ini = time.time()
    if wavelength == 'UV' or wavelength =='X' or wavelength == 'all':
        start = time.time()
        bpass_read = bpass_loader(parallel = n_processes)
    time_out = time.time()
    #print("bpass reading now", time_out-time_ini)
    #assert bpass_read==0, f"running now!"    
    if wavelength!= 'all':
        emissivities_redshift = []
    else:
        emissivities_x_z = []
        emissivities_lw_z = []
        emissivities_uv_z = []

    current_pid = os.getpid()

    for index,z in enumerate(np.linspace(z_init,z_end,z_steps)):

        #initialize the h5 file

        container = HdF5Saver(
            z,
            current_pid,
            '/home/inikolic/projects/stochasticity/samples/dir_080323/full/'
        )
        container.create_file()
        container.create_redshift()
        container.add_Rbias(R_bias)

        with Manager() as manager:
            if wavelength!='all':
                emissivities = manager.list()
            else:
                emissivities_x = manager.list()
                emissivities_lw = manager.list()
                emissivities_uv = manager.list()
            processes=[]
            for i in range(n_processes):
                if wavelength == 'X':
                    p = Process(target=Sampler_x, 
                                kwargs={'emissivities_list':emissivities,
                                        'z':z, 
                                        'dlog10m':dlog10m,
                                        'N_iter':N_iter,
                                        'R_bias':5,  #default choices 
                                        'log10_Mmin':7.2, 
                                        'mass_binning':150,
                                        'sample_hmf':sample_Poiss,
                                        'sample_SFR':sample_SFR,
                                        'sample_Lx':sample_emiss})
                elif wavelength == 'UV':
                    p = Process(target=Sampler_UV,
                                kwargs={'emissivities_list':emissivities,
                                        'z':z,
                                        'dlog10m':dlog10m,
                                        'N_iter':N_iter,
                                        'R_bias':5,  #default choices 
                                        'log10_Mmin':7.2,
                                        'mass_binning':150,
                                        'sample_hmf':sample_Poiss,
                                        'sample_SFR':sample_SFR,
                                        'sample_UV':sample_emiss,
                                        'bpass_read': bpass_read})
                elif wavelength == 'LW':
                    p = Process(target=Sampler_LW,
                                kwargs={'emissivities_list':emissivities,
                                        'z':z,
                                        'dlog10m':dlog10m,
                                        'N_iter':N_iter,
                                        'R_bias':5,  #default choices 
                                        'log10_Mmin':7.2,
                                        'mass_binning':150,
                                        'sample_hmf':sample_Poiss,
                                        'sample_SFR':sample_SFR,
                                        'sample_LW':sample_emiss,
                                        'bpass_read': bpass_read})
                elif wavelength == 'all':
                    time_start_sampling = time.time()
                    print("Currently in the function run.py, starting to sample soon:", time_start_sampling - time_start_run)
                    p = Process(target = Sampler_ALL,
                               kwargs={'emissivities_x_list': emissivities_x,
                                      'emissivities_lw_list': emissivities_lw,
                                      'emissivities_uv_list': emissivities_uv,
                                      'z': z,
                                      'dlog10m': dlog10m,
                                      'N_iter': N_iter,
                                      'R_bias': R_bias,
                                      'log10_Mmin': 5.0,
                                      'mass_binning': 1,
                                      'sample_hmf': sample_Poiss,
                                      'sample_SFR': sample_SFR,
                                      'sample_emiss': sample_emiss,
                                      'bpass_read': bpass_read,
                                      'main_pid': current_pid,
                                      })
                    time_end_sampling = time.time()
                    print("Finished sampling for this redshift, for", N_iter,"iterations, it took", time_end_sampling-time_start_sampling)
                else: 
                    raise ValueError('Wrong wavelenght string!')
                processes.append(p)
                p.start()
             #   print(p.pid)
            for p in processes:
                p.join()
            if wavelength != 'all':
                emissivities_redshift.append(np.array(emissivities).flatten())
            else:
                emissivities_x_z.append(np.array(emissivities_x).flatten())
                emissivities_lw_z.append(np.array(emissivities_lw).flatten())
                emissivities_uv_z.append(np.array(emissivities_uv).flatten())

        container = None
    directory = '/home/inikolic/projects/stochasticity/samples/'
    
    if wavelength!='all':
        filename = directory + wavelength + '_emissivities' + str(z_init) + '_'
        filename = filename + str(z_end) + '_' + str(N_iter * n_processes)
        if sample_SFR == 0 or sample_SFR ==2:
            filename = filename + '_sfrTRUE'
        else:
            filename = filename + '_sfrFALSE'
        if sample_emiss ==0 or sample_emiss == 2 :
            filename = filename + '_emissTRUE_'
        else:
            filename = filename + '_emissFALSE_'
        filename = filename + str(sample_Poiss).upper() + '.npy'
        np.save(filename, np.array(emissivities_redshift, dtype=object))
    
    else:
        filename_x = directory + 'Xall' + '_emissivities' + str(z_init) + '_' + str(z_end) + '_R' + str(R_bias)
        filename_lw = directory + 'LWall' + '_emissivities' + str(z_init) + '_' + str(z_end) + '_R' + str(R_bias)
        filename_uv = directory + 'UVall' + '_emissivities' + str(z_init) + '_' + str(z_end) + '_R' + str(R_bias)
        
        if sample_SFR:
            filename_x = filename_x + '_sfrTRUE'
            filename_lw = filename_lw + '_sfrTRUE'
            filename_uv = filename_uv + '_sfrTRUE'
        
        else:
            filename_x = filename_x + '_sfrFALSE'
            filename_lw = filename_lw + '_sfrFALSE'
            filename_uv = filename_uv + '_sfrFALSE'
        
        if sample_emiss:
            filename_x = filename_x + '_emissTRUE_'
            filename_lw = filename_lw + '_emissTRUE_'
            filename_uv = filename_uv + '_emissTRUE_'
            
        else:
            filename_x = filename_x + '_emissFALSE_'
            filename_lw = filename_lw + '_emissFALSE_'
            filename_uv = filename_uv + '_emissFALSE_'
        
        filename_x = filename_x + str(sample_Poiss).upper() + '.npy'
        filename_lw = filename_lw + str(sample_Poiss).upper() + '.npy'
        filename_uv = filename_uv + str(sample_Poiss).upper() + '.npy'
        
        np.save(filename_x, np.array(emissivities_x_z, dtype = object))
        np.save(filename_lw, np.array(emissivities_lw_z, dtype = object))
        np.save(filename_uv, np.array(emissivities_uv_z, dtype = object))
        print("From the end of sampling to end of everything it took", time.time()-time_end_sampling)

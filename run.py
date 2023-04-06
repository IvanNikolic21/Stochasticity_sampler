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

import argparse
from sampler_x import Sampler_x
from sampler_uv import Sampler_UV
from sampler_lw import Sampler_LW
from sampler_all import Sampler_ALL
from bpass_read import bpass_loader
import os
import time
import sys
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count, Process, Manager
from save import saving_function, error_function
import datetime

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelength", type=str, default='all')
    parser.add_argument("--z_init", type=float, default=5.0)
    parser.add_argument("--z_end", type=float, default = 20.0)
    parser.add_argument("--z_steps", type=int, default = 10)
    parser.add_argument("--dlog10m", type=float, default = 0.01)
    parser.add_argument("--N_iter", type=int, default = 100)
    parser.add_argument("--n_processes", type=int, default = 1)

    parser.add_argument("--sample_SFR", type=bool, default=True)
    parser.add_argument("--sample_emiss", type=bool, default=True)
    parser.add_argument("--sample_Poiss", type=bool, default=True)
    parser.add_argument("--sample_met", type=bool, default = True)

    parser.add_argument("--R_bias", type=float, default=5.0)
    parser.add_argument("--f_esc_option", type=str, default='binary')

    parser.add_argument("--control_run", type=bool, default=False)
    parser.add_argument("--get_previous", type=bool, default=False)

    inputs = parser.parse_args()
    assert inputs.control_run!= True or inputs.get_previous!=True, "Not compatible combination"

    if inputs.sample_SFR in ["False", "FALSE", "false", "0", "No"] or inputs.sample_SFR==False:
        sample_SFR = False
    else: 
        sample_SFR = True

    if inputs.sample_emiss in ["False", "FALSE", "false", "0", "No"] or inputs.sample_emiss==False:
        sample_emiss = False
    else:
        sample_emiss = True

    if inputs.sample_Poiss in ["False", "FALSE", "false", "0", "No"] or inputs.sample_Poiss==False:
        sample_Poiss = False
    else:
        sample_Poiss = True
    
    assert type(sample_SFR) == bool, "Something went wrong, type is not boolean"

    time_ini = time.time()
    if inputs.wavelength == 'UV' or inputs.wavelength =='X' or inputs.wavelength == 'all':
        start = time.time()
        bpass_read = bpass_loader(parallel = inputs.n_processes)
    time_out = time.time()
    #print("bpass reading now", time_out-time_ini)
    #assert bpass_read==0, f"running now!"    
    if inputs.wavelength!= 'all':
        emissivities_redshift = []
    else:
        emissivities_x_z = []
        emissivities_lw_z = []
        emissivities_uv_z = []

    current_pid = os.getpid()
    directory = '/home/inikolic/projects/stochasticity/samples/dir_080323/full/'


    filename = directory + 'full' + str(
            inputs.z_init) + '_' + str(inputs.z_end) + '_R' + str(inputs.R_bias)

    if sample_SFR:
        filename = filename + '_sfrTRUE'
    else:
        filename = filename + '_sfrFALSE'

    if sample_emiss:
        filename = filename + '_emissTRUE_'
    else:
        filename = filename + '_emissFALSE_'

    filename = filename + str(sample_Poiss).upper()
    filename = filename +  '_' + str(datetime.datetime.now().date()) + '.h5'

    f = h5py.File(filename,'a')
    f.attrs["Rbias"] = inputs.R_bias
    f.attrs["sample_SFR"] = sample_SFR
    f.attrs["sample_emiss"] = sample_emiss
    f.attrs["sample_Poiss"] = sample_Poiss
    f.close()
    print('Sampling blah')
    for index,z in enumerate(np.linspace(inputs.z_init,inputs.z_end,inputs.z_steps)):
        f = h5py.File(filename, 'a')
        f.create_group(str(z))
        f.close()
        #initialize the h5 file

        # container = HdF5Saver(
        #     z,
        #     current_pid,
        #     '/home/inikolic/projects/stochasticity/samples/dir_080323/full/'
        # )
        #container.create_file()
        #container.create_redshift()
        #container.add_Rbias(R_bias)
#        Sampler_ALL(emissivities_x_list= [],
 #                   emissivities_lw_list= [],
  #                  emissivities_uv_list=[],
   #                 z = z,
    #                dlog10m = inputs.dlog10m,
     #               N_iter =  inputs.N_iter,
      #              R_bias = inputs.R_bias,
       #             log10_Mmin = 5.0,
        #            mass_binning = 1,
         #           sample_hmf = sample_Poiss,
          #          sample_SFR = sample_SFR,
           #         sample_emiss = sample_emiss,
            #        sample_met = inputs.sample_met,
             #       bpass_read = bpass_read,
              #      filename = filename,
               #     control_run= inputs.control_run,
                #    f_esc_option = inputs.f_esc_option,
                 #   proc_number = 1.0,
                  #  get_previous = inputs.get_previous,
 
                   # )
        with Manager() as manager:
            if inputs.wavelength!='all':
                emissivities = manager.list()
            else:
                emissivities_x = manager.list()
                emissivities_lw = manager.list()
                emissivities_uv = manager.list()
            processes=[]
            pool = Pool(inputs.n_processes)
            for i in range(inputs.n_processes):
                if inputs.wavelength == 'X':
                    p = Process(target=Sampler_x, 
                                kwargs={'emissivities_list':emissivities,
                                        'z':z, 
                                        'dlog10m':inputs.dlog10m,
                                        'N_iter':inputs.N_iter,
                                        'R_bias':inputs.R_bias,  #default choices
                                        'log10_Mmin':7.2, 
                                        'mass_binning':150,
                                        'sample_hmf':sample_Poiss,
                                        'sample_SFR':sample_SFR,
                                        'sample_Lx':sample_emiss})
                elif inputs.wavelength == 'UV':
                    p = Process(target=Sampler_UV,
                                kwargs={'emissivities_list':emissivities,
                                        'z':z,
                                        'dlog10m':inputs.dlog10m,
                                        'N_iter':inputs.N_iter,
                                        'R_bias':inputs.R_bias,  #default choices
                                        'log10_Mmin':7.2,
                                        'mass_binning':150,
                                        'sample_hmf':sample_Poiss,
                                        'sample_SFR':sample_SFR,
                                        'sample_UV':sample_emiss,
                                        'bpass_read': bpass_read})
                elif inputs.wavelength == 'LW':
                    p = Process(target=Sampler_LW,
                                kwargs={'emissivities_list':emissivities,
                                        'z':z,
                                        'dlog10m':inputs.dlog10m,
                                        'N_iter':inputs.N_iter,
                                        'R_bias':inputs.R_bias,  #default choices
                                        'log10_Mmin':7.2,
                                        'mass_binning':150,
                                        'sample_hmf':sample_Poiss,
                                        'sample_SFR':sample_SFR,
                                        'sample_LW':sample_emiss,
                                        'bpass_read': bpass_read})
                elif inputs.wavelength == 'all':
                    time_start_sampling = time.time()
                    #print("Currently in the function run.py, starting to sample soon:", time_start_sampling - time_start_run)
                    # p = Process(target = Sampler_ALL,
                    #            kwargs={'emissivities_x_list': emissivities_x,
                    #                   'emissivities_lw_list': emissivities_lw,
                    #                   'emissivities_uv_list': emissivities_uv,
                    #                   'z': z,
                    #                   'dlog10m': dlog10m,
                    #                   'N_iter': N_iter,
                    #                   'R_bias': R_bias,
                    #                   'log10_Mmin': 5.0,
                    #                   'mass_binning': 1,
                    #                   'sample_hmf': sample_Poiss,
                    #                   'sample_SFR': sample_SFR,
                    #                   'sample_emiss': sample_emiss,
                    #                   'bpass_read': bpass_read,
                    #                   'main_pid': current_pid,
                    #                   })
                    pool.apply_async(Sampler_ALL,
                                     kwds = {
                                         'emissivities_x_list': emissivities_x,
                                         'emissivities_lw_list': emissivities_lw,
                                         'emissivities_uv_list': emissivities_uv,
                                         'z': z,
                                         'dlog10m': inputs.dlog10m,
                                         'N_iter': inputs.N_iter,
                                         'R_bias': inputs.R_bias,
                                         'log10_Mmin': 5.0,
                                         'mass_binning': 1,
                                         'sample_hmf': sample_Poiss,
                                         'sample_SFR': sample_SFR,
                                         'sample_emiss': sample_emiss,
                                         'sample_met' : inputs.sample_met,
                                         'bpass_read': bpass_read,
                                         'filename': filename,
                                         'control_run': inputs.control_run,
                                         'f_esc_option' : inputs.f_esc_option,
                                         'proc_number' : i,
                                         'get_previous' : inputs.get_previous,
                                     },
                                     callback=saving_function,
                                     error_callback = error_function)

                    time_end_sampling = time.time()
                else:
                    raise ValueError('Wrong wavelength string!')
                #processes.append(p)
                #p.start()
             #   print(p.pid)

           # for p in processes:
           #     p.join()
            pool.close()
            pool.join()
            if inputs.wavelength != 'all':
                emissivities_redshift.append(np.array(emissivities).flatten())
            else:
                emissivities_x_z.append(np.array(emissivities_x).flatten())
                emissivities_lw_z.append(np.array(emissivities_lw).flatten())
                emissivities_uv_z.append(np.array(emissivities_uv).flatten())


        container = None
    directory = '/home/inikolic/projects/stochasticity/samples/'
    
    if inputs.wavelength!='all':
        filename = directory + inputs.wavelength + '_emissivities' + str(inputs.z_init) + '_'
        filename = filename + str(inputs.z_end) + '_' + str(inputs.N_iter * inputs.n_processes)
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
        filename_x = directory + 'Xall' + '_emissivities' + str(inputs.z_init) + '_' + str(inputs.z_end) + '_R' + str(inputs.R_bias)
        filename_lw = directory + 'LWall' + '_emissivities' + str(inputs.z_init) + '_' + str(inputs.z_end) + '_R' + str(inputs.R_bias)
        filename_uv = directory + 'UVall' + '_emissivities' + str(inputs.z_init) + '_' + str(inputs.z_end) + '_R' + str(inputs.R_bias)
        
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

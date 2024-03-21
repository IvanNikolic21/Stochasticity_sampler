"""Driver of the sampler for x-ray, UV and LW.

Parameters are going to be:
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
from sampler_all import sampler_all_func
from bpass_read import bpass_loader
from common_stuff import SFH_sampler, _sample_densities
from chmf import chmf
import os
import time
import numpy as np
import h5py
from multiprocessing import Pool, Manager
from save import saving_function
import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelength", type=str, default='all')
    parser.add_argument("--z_init", type=float, default=5.0)
    parser.add_argument("--z_end", type=float, default=20.0)
    parser.add_argument("--z_steps", type=int, default=10)
    parser.add_argument("--dlog10m", type=float, default=0.01)
    parser.add_argument("--N_iter", type=int, default=100)
    parser.add_argument("--n_processes", type=int, default=1)

    parser.add_argument(
        "--no_SFR_sampling",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--no_emiss_sampling",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--no_Poiss_sampling",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--no_met_sampling",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--no_Mstar_sampling",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument("--R_bias", type=float, default=5.0)
    parser.add_argument("--f_esc_option", type=str, default='binary')

    parser.add_argument("--control_run", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_previous_run", type=str, default='False')

    parser.add_argument("--shift_scaling", action=argparse.BooleanOptionalAction)

    parser.add_argument("--literature_run", type=str, default='None')
    parser.add_argument(
        "--flattening",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        "--density_distribution",
        type=bool,
        default=True
    )
    inputs = parser.parse_args()
    assert inputs.control_run!= True or inputs.use_previous_run=='False', "Not compatible combination"

    assert inputs.use_previous_run in [
        'False',
        'bigz',
        'bigger',
        'lowz',
        'bigz_noPo',
        'bigz_noPo_new',
        'lowz_noPo',
        'bigger_noPo',
        'biggestz',
        'midz',
        'bigz2',
        'bigger_hmfnew',
        'bigz_hmfnew',
    ], "No cache file for this combination"

    if inputs.no_SFR_sampling:
        sample_SFR = False
    else: 
        sample_SFR = True

    if inputs.no_Mstar_sampling:
        sample_Mstar = False
    else:
        sample_Mstar = True

    if inputs.no_emiss_sampling:
        sample_emiss = False
    else:
        sample_emiss = True

    if inputs.no_Poiss_sampling:
        sample_Poiss = False
    else:
        sample_Poiss = True

    if inputs.no_met_sampling:
        sample_met = False
    else:
        sample_met = True

    if inputs.shift_scaling:
        shift_scaling = True
    else:
        shift_scaling = False
    
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

    if sample_Mstar:
        filename = filename + '_MstarTRUE'
    else:
        filename = filename + '_MstarFALSE'

    if sample_emiss:
        filename = filename + '_emissTRUE_'
    else:
        filename = filename + '_emissFALSE_'

    if sample_met:
        filename = filename + '_metTrue_'
    else:
        filename = filename + '_metFalse_'

    if inputs.f_esc_option != 'binary':
        filename = filename + '_' + str(inputs.f_esc_option) + '_'

    filename = filename + str(sample_Poiss).upper()
    filename = filename +  '_' + str(datetime.datetime.now().date()) + '.h5'

    f = h5py.File(filename,'a')
    f.attrs["Rbias"] = inputs.R_bias
    f.attrs["sample_SFR"] = sample_SFR
    f.attrs["sample_Mstar"] = sample_Mstar
    f.attrs["sample_emiss"] = sample_emiss
    f.attrs["sample_Poiss"] = sample_Poiss
    f.close()

    iter_per_par = 100 #Number of iterations to be performed in the sampler

    for index,z in enumerate(np.linspace(inputs.z_init,inputs.z_end,inputs.z_steps)):
        f = h5py.File(filename, 'a')
        f.create_group(str(z))
        f.close()

        SFH_samp = SFH_sampler(z)
        if not inputs.control_run and inputs.use_previous_run=='False' and inputs.density_distribution:
            delta_list = _sample_densities(z,
                                           inputs.N_iter * inputs.n_processes,
                                           5.0,
                                           15.0,
                                           0.01,
                                           inputs.R_bias)
            delta_list = delta_list.reshape((inputs.N_iter, inputs.n_processes))
            # np.savetxt('/home/inikolic/projects/stochasticity/samples/density{}.txt'.format(z), np.array(delta_list))
            hmf_this = chmf(z=z, delta_bias=0.0, R_bias=inputs.R_bias)
            hmf_this.prep_for_hmf_st(5.0, 15.0, 0.01)
            #hmf_this.prep_collapsed_fractions(check_cache=False)
        elif not inputs.density_distribution:
            delta_list = np.zeros((inputs.N_iter, inputs.n_processes))
            hmf_this = chmf(z=z, delta_bias=0.0, R_bias=inputs.R_bias)
            hmf_this.prep_for_hmf_st(5.0, 15.0, 0.01)
            #hmf_this.prep_collapsed_fractions(check_cache=False)
        else:
            delta_list = np.zeros((inputs.N_iter, inputs.n_processes)) 
            hmf_this = None
        #initialize the h5 file


        #assert 1==0, "Done with this small thingy"
        with Manager() as manager:
            if inputs.wavelength!='all':
                emissivities = manager.list()
            else:
                emissivities_x = manager.list()
                emissivities_lw = manager.list()
                emissivities_uv = manager.list()
            processes=[]
            #pool = Pool(inputs.n_processes )
            for iter_num in range(int(inputs.N_iter / iter_per_par)):
                with Pool(processes=inputs.n_processes) as pool:
                

                    if inputs.wavelength == 'all':
                        time_start_sampling = time.time()
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

                        #for proc in psutil.process_iter():
                        #    print(proc.open_files())
                        #assert 1==0, "at least got here"
                        results = [pool.apply_async(sampler_all_func,
                                                   kwds = {
                                                       'emissivities_x_list': emissivities_x,
                                                       'emissivities_lw_list': emissivities_lw,
                                                       'emissivities_uv_list': emissivities_uv,
                                                       'z': z,
                                                       'dlog10m': inputs.dlog10m,
                                                       'N_iter': iter_per_par,
                                                       'r_bias': inputs.R_bias,
                                                       'log10_m_min': 5.0,
                                                       'mass_binning': 1,
                                                       'sample_hmf': sample_Poiss,
                                                       'sample_SFR': sample_SFR,
                                                       'sample_emiss': sample_emiss,
                                                       'sample_met' : sample_met,
                                                       'sample_Mstar': sample_Mstar,
                                                       'bpass_read': bpass_read,
                                                       'filename': filename,
                                                       'control_run': inputs.control_run,
                                                       'f_esc_option': inputs.f_esc_option,
                                                       'proc_number': i,
                                                       'get_previous': inputs.use_previous_run,
                                                       'density_inst': delta_list[(iter_num * iter_per_par) : ((iter_num+1) * iter_per_par), i],
                                                       'hmf_this': hmf_this,
                                                       'SFH_samp': SFH_samp,
                                                       'iter_num': iter_num,
                                                       'shift_scaling': shift_scaling,
                                                       'literature_run': inputs.literature_run,
                                                       'flattening': inputs.flattening
                                         })
                                 #        callback=saving_function,
                        #                 error_callback = error_function,
                             for i in range(inputs.n_processes-1)
                        ]
                        saving_function([j for i in results for j in i.get()])
                        time_end_sampling = time.time()
                    else:
                        raise ValueError('Wrong wavelength string!')
           #processes.append(p)
                #p.start()

           # for p in processes:
           #     p.join()
                #pool.close()
                #pool.join()
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

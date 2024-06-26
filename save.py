import numpy as np
import h5py
import os.path

class HdF5Saver:
    def __init__(self, z, pid, output_dir):
        self.redshift = z
        self.pid = pid #necessary not to have multiprocessing issues
        self.output_dir = output_dir
        self.create_file()

    def create_file(self):
        self.filename = self.output_dir + str(self.pid) + str(self.redshift) + '.hdf5'
        f  = h5py.File(self.filename, 'a')
        f.attrs['redshift'] = self.redshift
        f.attrs['pid'] = self.pid
        f.close()
        self.created = True

    def create_redshift(self):
        f = h5py.File(self.filename, 'a')
        f.create_group(str(self.redshift))
        f.close()
    def add_Rbias(self, Rbias):
        f = h5py.File(self.filename, 'a')
        f.attrs["Rbias"] = Rbias
        f.close()

    def add_X(self, X):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)].create_dataset(
            "X",
            dtype = "float",
            data = X,
            compression = 'gzip',
            compression_opts = 9,
        )
        f.close()

    def add_UV(self, UV):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)].create_dataset(
            "UV",
            dtype = "float",
            data = UV,
            compression = 'gzip',
            compression_opts = 9,
        )
        f.close()

    def add_LW(self, LW):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)].create_dataset(
            "LW",
            dtype = "float",
            data = LW,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_delta_group(self, delta):
        f = h5py.File(self.filename, 'a')
        self.delta = delta
        f[str(self.redshift)].create_group(str(delta))
        f[str(self.redshift)][str(delta)].attrs['delta'] = delta
        f.close()

    def add_halo_masses(self, Mh):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "Mh",
            dtype = "float",
            data = Mh,
            compression = 'gzip',
            compression_opts = 9,
        )
    def add_stellar_masses(self, Ms):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "Ms",
            dtype = "float",
            data = Ms,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_SFR(self, SFR):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "SFR",
            dtype = "float",
            data = SFR,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_metal(self, Z):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "metalicity",
            dtype = "float",
            data = Z,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_Lx(self, Lx):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "Lx",
            dtype = "float",
            data = Lx,
            compression = 'gzip',
            compression_opts = 9,
        )
    def add_L_LW(self, L_lw):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "L_lw",
            dtype = "float",
            data = L_lw,
            compression = 'gzip',
            compression_opts = 9,
        )
    def add_L_UV(self, L_uv):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "L_uv",
            dtype = "float",
            data = L_uv,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_L_LyC(self, L_lyc):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "L_LyC",
            dtype = "float",
            data = L_lyc,
            compression = 'gzip',
            compression_opts = 9,
        )


    def add_beta(self, beta):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "beta",
            dtype = "float",
            data = beta,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_nion(self, nion):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "nion",
            dtype = "float",
            data = nion,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_uvlf(self, uvlf):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "uvlf",
            dtype = "float",
            data = uvlf,
            compression = 'gzip',
            compression_opts = 9,
        )

    def add_SFH(self, SFH):
        f = h5py.File(self.filename, 'a')
        f[str(self.redshift)][str(self.delta)].create_dataset(
            "SFH",
            dtype = "float",
            data = SFH,
            compression = 'gzip',
            compression_opts = 9,
        )

def error_function(what):
    print("Something happened!")
    raise ValueError("Check your function", what)

def saving_function(class_list):
    print("accessed saving function")
    if hasattr(class_list, '__len__'):
        hdf = h5py.File(class_list[0].filename, 'a')
    else:
        hdf = h5py.File(class_list.filename, 'a')
        class_list = [class_list]
    for file in class_list:
        delta = file.delta
        redshift = file.redshift
        print(delta)
        try:
            hdf[str(redshift)].create_group(str(delta))
        except ValueError:
            print("Something happened, here's the error. First, all of the keys")
            #print(hdf[str(redshift)].keys())
            print("Now what I want to add")
            print(str(delta))
            print("And the redshift")
            print(str(redshift))
            #raise ValueError
            print("skipping for now but beware")
            continue
        hdf[str(redshift)][str(delta)].attrs['delta'] = delta

        if hasattr(file, 'proc_number'):
            hdf[str(redshift)][str(delta)].attrs['proc_number'] = file.proc_number
        if hasattr(file, 'iter_number'):
            hdf[str(redshift)][str(delta)].attrs['iter_number'] = file.iter_number

        hdf[str(redshift)][str(delta)].create_dataset(
            "SFH",
            dtype="float",
            data=file.SFH,
            #compression='gzip',
            #compression_opts=9,
        )

    #    hdf[str(redshift)][str(delta)].create_dataset(
    #        "uv_lf",
    #        dtype="float",
    #        data=file.uv_lf,
    #        #compression='gzip',
    #        #compression_opts=9,
    #    )

        hdf[str(redshift)][str(delta)].create_dataset(
            "nion",
            dtype="float",
            data=file.nion,
            #compression='gzip',
            #compression_opts=9,
        )

    #    hdf[str(redshift)][str(delta)].create_dataset(
    #        "beta",
    #        dtype="float",
    #        data=file.beta,
    #        #compression='gzip',
    #        #compression_opts=9,
    #    )

        hdf[str(redshift)][str(delta)].create_dataset(
            "L_LyC",
            dtype="float",
            data=file.L_lyc,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "L_uv",
            dtype="float",
            data=file.L_uv,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "L_lw",
            dtype="float",
            data=file.L_lw,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "L_x",
            dtype="float",
            data=file.L_x,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "metallicity",
            dtype="float",
            data=file.metallicity,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "SFR",
            dtype="float",
            data=file.SFR,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "stellar_masses",
            dtype="float",
            data=file.stellar_masses,
            #compression='gzip',
            #compression_opts=9,
        )

        hdf[str(redshift)][str(delta)].create_dataset(
            "halo_masses",
            dtype="float",
            data=file.halo_masses,
            #compression='gzip',
            #compression_opts=9,
        )
    hdf.close()

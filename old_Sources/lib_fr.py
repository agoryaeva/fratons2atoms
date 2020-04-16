#<------------------------------------------------------->#
#<----       Fratons2atoms Library                   ---->#
#<---    Alexandra M. Goryaeva, M. C. Marinica ---------->#
#---------------------------------------------------------#

import os
import sys
import shutil
import numpy as np
import h5py
from matplotlib import pyplot as plt
import natsort
from ase import atoms
from ase.io import read, iread, write
from scipy import ndimage as ndi
from scipy.signal import argrelextrema



def read_hdf5(file, h5_key, path=None):
    if path == None:
        file_r = file
    else:
        file_r = path + file

    if not os.path.exists(file_r):
        print("The directory %s  doesn t exists" % str(file_r))
        exit(0)
    db = h5py.File(file_r, 'r')
    #print("The keys of the HDF5 file...............:", db.keys())
    dset = db[h5_key]
    np_dset = np.asarray(dset, dtype=float)

    return np_dset;





def select_type_files(path, ftype):

    if not os.path.exists(path):
        print("The directory %s  doesn t exists" % str(path))
        exit(0)

    files = os.listdir(path)
    fsorted = natsort.natsorted(files)
    number=len(ftype)
    files_selected = []
    for fname in fsorted:
        if fname[-number:] == ftype:
            files_selected.append(fname)

    return files_selected


# - - - - - - - - - - - - - - - - - -


def make_zeroes(number):
    """ Function to create n zeroes before the file_id in the poscar filename"""
    if number < 10:
        zeroes = '00000'
    elif (number >= 10) and (number < 100):
        zeroes = '0000'
    elif (number >= 100) and (number < 1000):
        zeroes = '000'
    elif (number >= 1000) and (number < 10000):
        zeroes = '00'
    return(zeroes)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)



def write_atoms_xyz(np_dset, threshold, prefix_file, index_at, out_dir=None):
    if out_dir == None:
        nfile = "at_" + prefix_file + ".xyz"
    else:
        nfile = out_dir + "at_" + prefix_file + ".xyz"

    cell = np.zeros((3,3))
    cell[0] = [np_dset.shape[0],          0        ,        0           ]
    cell[1] = [     0            , np_dset.shape[1],        0           ]
    cell[2] = [     0           ,           0        , np_dset.shape[2] ]


    #I write only the elements that have the density bigger than a thershold ...
    #idx_sel = np.where( np_dset > threshold)
    #if  index_at != None:
    idx_sel = index_at
    #else:
    """
    #    idx_sel = np.where( abs(np_dset) > threshold)
    f=open(nfile,'w')
    f.write("%d \n"%len((idx_sel[0])))
    f.write('config_type=%s Lattice="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"' % ('draft', cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0], cell[2][1], cell[2][2]))
    f.write(' Properties=species:S:1:pos:R:3:density:R:1:')
    f.write('pbc="T T T" dipole="0.0 0.0 0.0" \n')
    for i in range(len((idx_sel[0]))):
        f.write("Fe  %7.1f %7.1f %7.1f %15.10f \n"%( idx_sel[0][i], idx_sel[1][i], idx_sel[2][i], np_dset[idx_sel[0][i]][idx_sel[1][i]][idx_sel[2][i]] ))
    f.close()
    """

    f=open(nfile,'w')
    f.write("%d \n"%len((idx_sel)))
    f.write('config_type=%s Lattice="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"' % ('draft', cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0], cell[2][1], cell[2][2]))
    f.write(' Properties=species:S:1:pos:R:3:density:R:1:')
    f.write('pbc="T T T" dipole="0.0 0.0 0.0" \n')
    for i in range(len((idx_sel))):
        f.write("Fe  %7.1f %7.1f %7.1f %15.10f \n"%( idx_sel[i][0], idx_sel[i][1], idx_sel[i][2], np_dset[idx_sel[i][0]][idx_sel[i][1]][idx_sel[i][2]] ))
    f.close()




    return



def write_atoms_smooth_xyz(np_dset, threshold, prefix_file, index_at, r_at, out_dir=None):
    if out_dir == None:
        nfile = "at_" + prefix_file + ".xyz"
    else:
        nfile = out_dir + "at_" + prefix_file + ".xyz"

    cell = np.zeros((3,3))
    cell[0] = [np_dset.shape[0],          0        ,        0           ]
    cell[1] = [     0            , np_dset.shape[1],        0           ]
    cell[2] = [     0           ,           0        , np_dset.shape[2] ]


    #I write only the elements that have the density bigger than a thershold ...
    #idx_sel = np.where( np_dset > threshold)
    #if  index_at != None:
    idx_sel = index_at
    #else:
    """
    #    idx_sel = np.where( abs(np_dset) > threshold)
    f=open(nfile,'w')
    f.write("%d \n"%len((idx_sel[0])))
    f.write('config_type=%s Lattice="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"' % ('draft', cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0], cell[2][1], cell[2][2]))
    f.write(' Properties=species:S:1:pos:R:3:density:R:1:')
    f.write('pbc="T T T" dipole="0.0 0.0 0.0" \n')
    for i in range(len((idx_sel[0]))):
        f.write("Fe  %7.1f %7.1f %7.1f %15.10f \n"%( idx_sel[0][i], idx_sel[1][i], idx_sel[2][i], np_dset[idx_sel[0][i]][idx_sel[1][i]][idx_sel[2][i]] ))
    f.close()
    """

    f=open(nfile,'w')
    f.write("%d \n"%len((idx_sel)))
    f.write('config_type=%s Lattice="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"' % ('draft', cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0], cell[2][1], cell[2][2]))
    f.write(' Properties=species:S:1:pos:R:3:density:R:1:')
    f.write('pbc="T T T" dipole="0.0 0.0 0.0" \n')
    for i in range(len((idx_sel))):
        f.write("Fe  %10.6f %10.6f %10.6f %15.10f \n"%( r_at[i][0], r_at[i][1], r_at[i][2], np_dset[idx_sel[i][0]][idx_sel[i][1]][idx_sel[i][2]] ))
    f.close()




    return







def write_fratons_xyz(np_dset, prefix_file, out_dir=None):
    if out_dir == None:
        nfile = "fr_" + prefix_file + ".xyz"
    else:
        nfile = out_dir + "fr_" + prefix_file + ".xyz"


    cell = np.zeros((3,3))
    cell[0] = [np_dset.shape[0],          0        ,        0           ]
    cell[1] = [     0            , np_dset.shape[1],        0           ]
    cell[2] = [     0           ,           0        , np_dset.shape[2] ]

    idx_sel = np.where(abs(np_dset) > 0.0)
    f = open(nfile, 'w')
    f.write("%d \n" % len((idx_sel[0])))
    f.write('config_type=%s Lattice="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"' % ('draft', cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0], cell[2][1], cell[2][2]))
    f.write(' Properties=species:S:1:pos:R:3:density:R:1:')
    f.write('pbc="T T T" dipole="0.0 0.0 0.0" \n')
    for i in range(len((idx_sel[0]))):
        f.write("Fe  %7.1f %7.1f %7.1f %15.10f \n" % (idx_sel[0][i], idx_sel[1][i], idx_sel[2][i], np_dset[idx_sel[0][i]][idx_sel[1][i]][idx_sel[2][i]]))
    f.close()

    return

def get_best_guess(data, d_grid):


    size = 1 + 2 * d_grid
    footprint = np.ones((size, size, size))
    footprint[d_grid, d_grid, d_grid] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint, mode='wrap')
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def entropy (density, temperature):
    factor = 8.617342*1e-5*temperature
    entro_dens = density*np.log(density) + (1.0 - density)*np.log(1.0 - density)

    return factor*entro_dens;

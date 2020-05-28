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




# - - - - - - - - - - - - - - - - - -

def read_hdf5(file, path=None):
    if path == None:
        file_r = file
    else:
        file_r = path + file

    if not os.path.exists(file_r):
        print("The directory %s  doesn t exists" % str(file_r))
        exit(0)
    db = h5py.File(file_r, 'r')
    #print("The keys of the HDF5 file...............:", db.keys())
    h5_key = list(db.keys())[0]
    dset = db[h5_key]
    np_dset = np.asarray(dset, dtype=float)

    return np_dset;



# - - - - - - - - - - - - - - - - - -


def select_type_files(path, ftype, fkey):
    """ Function to select files of sertain type with sertain name pattern
        Example: to select the files with names drp__XXXXXXXXX.h5 in directory results/,
                one should provide:
                        path = 'results/'
                        ftype = '.h5'
                        fkey = 'drp_'
        The function will select all the files that begin with fkey and end with ftype in the indicated directory """

    if not os.path.exists(path):
        print("The directory %s  doesn t exists" % str(path))
        exit(0)

    files = os.listdir(path)
    fsorted = natsort.natsorted(files)
    nlast=len(ftype)
    nfirst=len(fkey)
    files_selected = []
    for fname in fsorted:
        if (fname[-nlast:] == ftype) and (fkey in fname):
            files_selected.append(fname)

    return files_selected



# - - - - - - - - - - - - - - - - - -

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# - - - - - - - - - - - - - - - - - -

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


def set_pbc_cell(np_dset):

    cell = np.zeros((3,3))
    cell[0] = [np_dset.shape[0],          0        ,        0           ]
    cell[1] = [     0            , np_dset.shape[1],        0           ]
    cell[2] = [     0           ,           0        , np_dset.shape[2] ]
    return cell


# - - - - - - - - - - - - - - - - - -

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


# - - - - - - - - - - - - - - - - - -


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



# - - - - - - - - - - - - - - - - - -

def get_best_guess(data, d_grid):

    size = 1 + 2 * d_grid
    footprint = np.ones((size, size, size))
    footprint[d_grid, d_grid, d_grid] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint, mode='wrap')
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values

# - - - - - - - - - - - - - - - - - -

def entropy (density, temperature):
    factor = 8.617342*1e-5*temperature
    entro_dens = density*np.log(density) + (1.0 - density)*np.log(1.0 - density)

    return factor*entro_dens;



# - - - - - - - - - - - - - - - - - -

def periodize_configuration(configuration, r_cut, cell):
    """applying PBC conditions on a rectangular box
    Parameters
        configuration: np.array of shape (n_atoms, 3), coordinates of the atoms to be periodized
        r_cut: float, cutoff radius
        cell: np.array of shape (3,3), dimensions of the periodic rectangle
    Returns
        periodized_configuration: np.array of shape (n_atoms_periodized, 3)
        initial_atom_ids: np.array of shape (n_atoms_periodized, )
            ids of the periodized atoms in the initial configuration
    """
    periodized_configuration = []
    initial_atom_ids = []
    dimensions=np.zeros((3))
    for i in range(3):
       dimensions[i]=cell[i][i]
    """
    x_translation = np.array([[dimensions[0], 0, 0]], dtype=configuration.dtype)
    y_translation = np.array([[0, dimensions[1], 0]], dtype=configuration.dtype)
    z_translation = np.array([[0, 0, dimensions[2]]], dtype=configuration.dtype)
    """

    x_translation=cell[0][:]
    y_translation=cell[1][:]
    z_translation=cell[2][:]

    mask_true = np.ones(configuration.shape[0], dtype=bool)
    in_or_out=  []
    for i_x, mask_x in [(-1., configuration[:, 0] > (dimensions[0] - r_cut)), (0., mask_true), (1., configuration[:, 0] < r_cut)]:
        for i_y, mask_y in [(-1., configuration[:, 1] > (dimensions[1] - r_cut)), (0., mask_true), (1., configuration[:, 1] < r_cut)]:
            for i_z, mask_z in [(-1., configuration[:, 2] > (dimensions[2] - r_cut)), (0., mask_true), (1., configuration[:, 2] < r_cut)]:

                mask = mask_x * mask_y * mask_z
                initial_atom_ids.append(np.nonzero(mask)[0])
                periodized_configuration.append(configuration[mask] + i_x*x_translation + i_y*y_translation + i_z*z_translation)
                if ((i_x==0) & (i_y==0) & (i_z==0)):
                   #print(np.nonzero(mask)[0])
                   in_or_out.append(np.nonzero(mask)[0])
                else:
                   y=np.nonzero(mask)[0]
                   y[y >= 0 ] = -1
                   in_or_out.append(y)


    periodized_configuration = np.concatenate(periodized_configuration, axis=0)
    initial_atom_ids = np.concatenate(initial_atom_ids, axis=0)
    in_or_out=np.concatenate(in_or_out, axis=0)
    """
    test=in_or_out[in_or_out >= 0]
    test2=in_or_out[in_or_out < 0]
    print(len(test), len(test2), len(test) + len(test2))
    """
    return periodized_configuration, initial_atom_ids, in_or_out

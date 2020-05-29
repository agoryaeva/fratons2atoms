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
from sklearn.neighbors import KDTree
from numba import jit


def clean_coords (coords, coords_unperturbed, cell, r_cut, dist_to_remove):

    periodized_coords, initial_atom_ids, in_or_out = periodize_configuration(coords, r_cut, cell)
    tree = KDTree(periodized_coords, leaf_size=10)
    dist, ind = tree.query(periodized_coords, k=2)
    toot = dist[ (0 < dist) & (dist < dist_to_remove)]
    ind_toot=ind[(0 < dist) & (dist < dist_to_remove)]
    ind_unique=ind_toot
    ind_deleted=[]

    for indx in ind_toot:
        if not(indx in ind_deleted):
            i_del = ind[indx][-1]
            ind_deleted.append(i_del)
            ind_unique=np.delete(ind_unique,np.where(ind_unique == i_del), axis=0)
            #debug print(indx, ind_unique)
    #ind_buffer
    #print(toot)
    """
    print("The initial index of atoms in double....:", ind_toot)
    print("The final   index of selected atoms.....:", len(ind_unique) , ind_unique )
    print("The final   index of deleted  atoms.....:", len(ind_deleted), ind_deleted)
    """
    for id in ind_unique:
        if in_or_out[id] >= 0:
            id_r=in_or_out[id]
            coords[id_r] = 0.5*(coords[id_r] + periodized_coords[ind[id][-1]])
    delete_real=[]
    for id in ind_deleted:
        if in_or_out[id] >= 0:
         delete_real.append(in_or_out[id])
    #print(delete_real)
    """
    print("coords before", coords.shape)
    """
    coords_clean=np.delete(coords,delete_real,axis=0)
    coords_unperturbed_clean=np.delete(coords_unperturbed,delete_real,axis=0)
    """
    print("coords_clean after", coords_clean.shape)
    """
    return coords_clean,coords_unperturbed_clean ;

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

def read_hdf5_old(file, h5_key, path=None):
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



def get_best_guess_new(data, d_grid):
    #-----------------Max Filter--------------------
    footprint = ndi.generate_binary_structure(3,7)
    # all fratons at local max are set to 1 all the others are 0
    filtered = ndi.maximum_filter(data, footprint=footprint, mode='wrap') == data

    #------------------Erosion----------------------
    # preparing the erosion of the selections ...
    # The visual definition of erosion from Wiki: https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    """
    Suppose A is a 13 x 13 matrix and B is a 3 x 3 matrix:

    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 0 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1               1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1               1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1               1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1

    Assuming that the origin B is at its center, for each pixel in A superimpose the origin of B, if B is completely contained by A the pixel is retained, else deleted.

    Therefore the Erosion of A by B is given by this 13 x 13 matrix.

    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 1 1 1 0 0 0 1 1 1 1 0
    0 1 1 1 1 0 0 0 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 1 1 1 1 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    """

    idx = largest_indices(-data, data.shape[0]**3)
    values = data [idx]
    no_low =  int(data.shape[0]**3/4)
    low_mean= np.sum(values[:no_low])/no_low
    #print(values, len(values), low_mean)

    #background = (data == 0)
    background = (data <= low_mean)
    eroded_background = ndi.binary_erosion(background, structure=footprint, border_value=1)
    # xor operation between the fileter and background erosion.
    mask_local_maxima = filtered ^ eroded_background
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
    return periodized_configuration, initial_atom_ids, in_or_out


@jit(nopython=True)
def get_weigthed_average(coords, coords_weighted, np_dset, np_dset_gaussian, mult, nult, m_grid_size):
    #coords_weighted = np.array(coords, dtype=float)

    m = m_grid_size
    for ic in range(len(coords)):
        val = coords[ic]
        #print(val, np_dset[val[0], val[1], val[2]])
        factor = 0.0
        tmp_i = 0.0
        tmp_j = 0.0
        tmp_k = 0.0
        for m1 in range(-m,m+1):
            for m2 in range(-m,m+1):
                for m3 in range(-m,m+1):
                    i1 = val[0] + m1
                    i1uf = val[0] + m1
                    if i1 >= np_dset.shape[0]:
                        i1 = i1 - np_dset.shape[0]
                    if  i1  < 0 :
                        i1 = i1 + np_dset.shape[0]

                    i2 = val[1] + m2
                    i2uf = val[1] + m2
                    if i2 >= np_dset.shape[1]:
                        i2 = i2 - np_dset.shape[1]
                    if  i2  < 0:
                        i2 = i2 + np_dset.shape[1]

                    i3 = val[2] + m3
                    i3uf = val[2] + m3
                    if i3 >= np_dset.shape[2]:
                        i3 = i3 - np_dset.shape[2]
                    if  i3 <  0:
                        i3 = i3 + np_dset.shape[2]
                    val_dens = np_dset_gaussian[i1, i2, i3]

                    sign_dens=np.sign(val_dens)
                    abs_dens=abs(val_dens)
                    factor = factor + sign_dens*abs_dens**mult
                    dist = (float(i1uf)**2 + float(i2uf)**2 + float(i3uf)**2 )**nult
                    tmp_i = tmp_i   + sign_dens*abs_dens**mult * float(i1uf)
                    tmp_j = tmp_j   + sign_dens*abs_dens**mult * float(i2uf)
                    tmp_k = tmp_k   + sign_dens*abs_dens**mult * float(i3uf)

                    """
                    factor = factor + val_dens**mult
                    dist = (float(i1uf)**2 + float(i2uf)**2 + float(i3uf)**2 )**nult
                    tmp_i = tmp_i   + val_dens**mult * float(i1uf)
                    tmp_j = tmp_j   + val_dens**mult * float(i2uf)
                    tmp_k = tmp_k   + val_dens**mult * float(i3uf)
                    """
        coords_weighted[ic] = np.array([tmp_i, tmp_j, tmp_k])/factor
    return coords_weighted;

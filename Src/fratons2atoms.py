#<------------------------------------------------------->#
#<----               Fratons2atoms                   ---->#
#<---        A. M. Goryaeva, M. C. Marinica    ---------->#
#---------------------------------------------------------#
#---------------------------------------------------------#
#                                                         #
#   Extraction of XYZ coordinates from atomic density:    #
#                  - reading h5 files                     #
#                  - applying 3 filters                   #
#                  - writing xyz files                    #
#                                                         #
#---------------------------------------------------------#


import argparse
import numpy as np
import h5py
import os
import sys
from sklearn.neighbors import KDTree
from scipy.ndimage import gaussian_filter
from lib_fr import *
from numba import jit

usage="""
     USAGE:
     -------------------------------------------------
       python3 fratons2atoms.py  directory_name/

       where directory_name/ is the path to *.h5 files to convert to xyz

"""

if (len(sys.argv)-1 != 1):
    print("\n\tERROR: Wrong number of arguments !!! ")
    print(usage)
    exit(0)

""""
#if one day the input change using parsers ....
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fratons2atoms')
    parser.add_argument('--dir', nargs='+', help="where directory_name/ is to path to *.h5 files to convert to xyz", required=True)
    parser.add_argument('--sigma_gaussian', type=float,  help="First Gausian filter. The variance", default=1.1)
    parser.add_argument('--size_window', type=int,  help="The size of the window in best_guess function", default=2)
    parser.add_argument('--mult', type=float,  help="The power for averaging fratons coordinates into atoms", default=2.0)
    parser.add_argument('--m_grid', nargs='+', help="The size of grid around central fraton to smooth", default=2)
    args = parser.parse_args()


    in_dir=args.dir
    sigma_gaussian=args.sigma_gaussian
    size_window=args.size_window
    mult=args.mult
    m_grid_size=args.m_grid_size
"""

#--------------Inputs-----------------

# Directory with *.h5 files to analyse
in_dir = sys.argv[1]

#First Gaussian filter: the standard value:
sigma_gaussian=0.5

#the size of the window in best_guess function:
size_window=1 # 3

#skewnes of the filter ... typical values 1, 2, 3 ... now 2.
mult=2.0
nult=1.0
# the size of grid around central fraton to smooth: 2 is nice: 0 - no filter, 1 - why not ; 2 or 3 nice choices.
m_grid_size = 2  #3

#r_cut in order to patch for PBC. r_cut>=max(a0_FCC, a0_bcc). E.g. in this case 8 is OK.
r_cut=16 # 16
#all the dimer of atoms at distances smaller that this distance will be replaced by only one atom in the middle of  the dimer:
dist_to_remove=3.8 #7.5

write_fratoms=False
#--------------------------------------------


out_dir_at='%s/xyz_at/' %(in_dir)
out_dir_at_smooth='%s/xyz_at_smooth/' %(in_dir)

if write_fratoms==True:
    out_dir_fr='%s/xyz_fr/' %(in_dir)
    os.system('rm -rf %s' % out_dir_fr)
    os.system('mkdir -p %s' % out_dir_fr)

ftype='.h5'    # type of files to read
fkey='drp__'   # prefix of files to read
key = 'drp'


os.system('rm -rf %s' % out_dir_at)
os.system('rm -rf %s' % out_dir_at_smooth)
os.system('mkdir -p %s' % out_dir_at)
os.system('mkdir -p %s' % out_dir_at_smooth)

files2read = select_type_files(in_dir, ftype, fkey)
print ('     ')
print (' ----------------------- DIRECTORY: %s ------------------------' %in_dir)
print ('  >>  Number of files to read: ............................... ', len(files2read))
print ('  >>  No   name file         at1    at1_clean   err_at1 err_end at2_clean   at2   err_end')


density=[]
o2=[]
coor = []
coor2 = []
for i in range(int(len(files2read))):
#for i in range(1):
    f = files2read[i]

    file_h5 = in_dir + f
    np_dset = read_hdf5(file_h5)
    #print(np_dset.shape)

    # first filter ....: define sigma : but probably should be desactivated. O(N_fratons) operation
    np_dset_gaussian = gaussian_filter(np_dset, sigma=sigma_gaussian, order=0, mode='wrap')

    # convolution home-made filter. O(N_fratons) operation
    np_dset_gaussian = convolution_sharpen(np_dset_gaussian)

    """
    #multiple shaprpen not tested version ... but priobably work if only shifted ...
    np_dset_gaussian0 = convolution_sharpen(np_dset)
    np_dset_gaussian = convolution_sharpen(np_dset_gaussian0)
    """
    #print(np_dset_gaussian.shape)
    #np_dset_gaussian = np_dset



    #second filter: define d_grid ; O(N_fratons) operation
    coords, sel_values = get_best_guess(np_dset_gaussian, d_grid=size_window)
    #coords, sel_values = get_best_guess_new(np_dset_gaussian, d_grid=size_window)
    #print("new", coords.shape)

    np_dset_gaussian = np_dset
    o2.append(len(sel_values))

    # add for PBC and clean coordinates for small distances: O(4*N_atoms * N_close_atoms**2) operation
    cell=set_pbc_cell(np_dset)
    coords00= coords
    coords_ini, coords_unperturbed_ini = clean_coords (coords, coords, cell, r_cut, dist_to_remove)

    """
    coords_ini = coords
    coords_unperturbed_ini=coords
    """

    #test if are some remaining small distances ... ; O(N_atoms * ln(N_atoms)*(k-1)) operation
    tree = KDTree(coords_ini, leaf_size=10)
    dist, ind = tree.query(coords_ini, k=2)
    toot = dist[(0 < dist) & (dist < dist_to_remove)]
    coor.append(int(len(toot)))
    """
    print("test1", len(toot))
    if np.any(dist[:,1]==0.0):
        print("WARNING COORDINATES coords CONTAMINATED")
        exit(0)
    """
    name=f.split('.')[0]

    #third ... filter: define mult & nult, O(4*N_atoms*mult**3) operation
    coords_weighted = np.array(coords, dtype=float)
    coords_weighted=get_weigthed_average(coords, coords_weighted, np_dset, np_dset_gaussian, mult, nult, m_grid_size)

    #clean the final coorrdinates:  O(N_atoms * N_close_atoms**2) operation
    coords_end, coords_unperturbed_end  = clean_coords (coords_weighted, coords00, cell, r_cut, dist_to_remove)
    removed_end = -len(coords_end) + len(coords_weighted)
    removed_ini = -len(coords_ini) + len(coords)
    """
    coords_end = coords_weighted
    coords_unperturbed_end=coords
    """
    #test if are some remaining small distances ...O(N_atoms * ln(N_atoms*(k-1)) operation
    tree2= KDTree(coords_end, leaf_size=10)
    dist, ind = tree2.query(coords_end, k=2)
    toot2 = dist[(0 < dist) & (dist < dist_to_remove)]
    coor2.append(int(len(toot2)))

    print("%7d %s %8d %8d %8d %8d %8d %8d %8d" % (i+1, f, len(coords), len(coords_ini), removed_ini, len(toot), len(coords_end), removed_end, len(toot2)))


    #writing brute coordinates after 2 filters
    write_atoms_xyz(np_dset, 0.0, name[5:], index_at=coords_ini,   out_dir=out_dir_at)

    #writing smooth  coordinates after 3 filters
    write_atoms_smooth_xyz(np_dset, 0.0, name[5:], index_at= coords_unperturbed_end, r_at= coords_end,   out_dir=out_dir_at_smooth)
    #write_atoms_smooth_xyz(np_dset, 0.0, name[5:], index_at= coords, r_at= coords_weighted,   out_dir=out_dir_at_smooth)


    if write_fratoms==True:
        write_fratons_xyz(np_dset_gaussian, name[5:], out_dir=out_dir_fr)

print ('     ')
print ('  >>  Non-smooth coordinates are written to: ................. ', out_dir_at)
print ('  >>  Smooth coordinates are written to: ..................... ', out_dir_at_smooth)
print ('     ')
print ('----------------------- ENJOY!!! ------------------------')
print ('     ')

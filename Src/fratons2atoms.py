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
from datetime import datetime

usage="""
     USAGE:
     -------------------------------------------------
       python3 fratons2atoms.py  --dir=directory_name/ --a0=X.XX

       where:
             directory_name/ is the path to *.h5 files to convert to xyz
             a0 is a unit cell parameter in fraton units 
                in case of 2 phases, the minimum a0 should be provided:
                e.g., for a0_bcc=6.5 and a0_fcc=8 -> a0=6.5 should be provided
                
       example:
                python3 fratons2atoms.py  --dir=my_calculation/ --a0=6.5
                  

"""



#if one day the input change using parsers ....
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fratons2atoms')
    parser.add_argument('--dir', type=str, help="the path to *.h5 files to convert to xyz", required=True)
    parser.add_argument('--a0', type=float, help="cell parameter of the system",required=True)
    
    # The following options can be used to set custom input parameters, different from default values
    parser.add_argument('--sigma', type=float,  help="The variance sigma of the Gaussian filter")
    parser.add_argument('--mask',  type=str,  help="Kernel mask: none, 333 or 555", choices=['none', '333', '555'])
    parser.add_argument('--size_window', type=int,  help="The size of the window in best_guess function. Good choises: 1 for low resolution calculations; 2 for higher resolution")
    parser.add_argument('--mu', type=float,  help="The power for averaging fratons coordinates into atoms. Good choise: 2.0") # if int(parser.parse_args().a0) < 6 else 2.0)
    parser.add_argument('--nu', type=float,  help="The power for averaging fratons coordinates into atoms. Good choise: 1.0")
    parser.add_argument('--m_grid', help="The size of grid around central fraton to smooth. Good choises: 2 for low resolution calculations; 3 for higher resolution")

    args = parser.parse_args()


    in_dir=args.dir
    a0=args.a0



#--------------Inputs-----------------

write_fratoms=False    # set True if writing of row atomic densities in xyz format desired. Warning: files will be heavy

# OPTIONS FOR a0_bcc=6.5 and a0_fcc=8.0
if a0<10:
    # F1: Gaussian
    sigma_gaussian=0.5 if args.sigma==None  else args.sigma         # Standard deviation of the Gaussian. For low resolution recommended values sigma <=1
    mask='none' if args.mask==None  else args.mask                  # 'none', '333' or '555' -> The effect of mask is not well tested
    # F2: Choose best candidates for atoms
    size_window=1 if args.size_window==None  else args.size_window  # 1 or 2, not 3!!
    # F3: Smooth coordinates
    mult=2.0 if args.mu==None  else args.mu                         # 0.0 -> no smoothing; 1.0-> very light 2.0 -> recommended value 3.O -> a lot of smoothing
    nult=1.0 if args.nu==None  else args.nu                         # 0.0 -> no second term in smoothong ; 1.0 -> moderate effect; 2.0 -> stronger effect
    m_grid_size=2 if args.m_grid==None  else args.m_grid            # 2 is a good choise for low resolution

       

# OPTIONS FOR a0_bcc=13 and a0_fcc=16.0
if 10<= a0 <17 :
    # F1: Gaussian
    sigma_gaussian=1.1 if args.sigma==None  else args.sigma        # Standard deviation of the Gaussian. 
    mask='none' if args.mask==None  else args.mask                 # 'none', '333' or '555' -> The effect of mask is not well tested
    # F2: Choose best candidates for atoms
    size_window=3 if args.size_window==None else args.size_window  # 2, 3 or 4!!
    # F3: Smooth coordinates
    mult=2.0 if args.mu==None  else args.mu                        # 0.0 -> no smoothing; 1.0-> very light 2.0 -> recommended value 3.O -> a lot of smoothing
    nult=1.0 if args.nu==None  else args.nu                        # 0.0 -> no second term in smoothong ; 1.0 -> moderate effect; 2.0 -> stronger effect
    m_grid_size=3 if args.m_grid==None  else args.m_grid           # 2 or 3 is a good choise
 


# OPTIONS FOR HIGHER RESOLUTION CALCULATIONS, i.e. a0 > 16
if a0 >= 17:
    print (' ------------------- !!!! WARNING !!!! ---------------------------')
    print ('                This value of a0 was not tested !                ' )
    print ('    The default parameters for conversion are likely not optimal ' )
    print ('  It is recommended to make tests to find appropriate parameters ' )
    print (' -----------------------------------------------------------------')
    # F1: Gaussian
    sigma_gaussian=1.1 if args.sigma==None  else args.sigma
    mask='none' if args.mask==None  else args.mask  # 'none', '333' or '555'
    # F2: Choose best candidates for atoms
    size_window=3 if args.size_window==None  else args.size_window
    # F3: Smooth coordinates
    mult=2.0 if args.mu==None  else args.mu
    nult=1.0 if args.nu==None  else args.nu
    m_grid_size=3 if args.m_grid==None  else args.m_grid
    

# CORRECTION OF SMALL DISTANCES
dist_to_remove=0.6*a0       # 2 atoms that are distant by less than than dist_to_remove will be identified as 1 atom in the middle of these fake atoms

# SOME TECHNICAL INPUTS -> NO CHANGE HERE
r_cut=a0*1.24               # The thinkness of layer to add at periodic boundary conditions in order to estimate close neighbours at the borders of the box

# FILES TO READ:
ftype='.h5'    # type of files to read
fkey='drp__'   # prefix of files to read


# WRITE OUTPUTS TO
out_dir_at='%sxyz_at/' %(in_dir)
out_dir_at_smooth='%sxyz_at_smooth/' %(in_dir)
os.system('rm -rf %s' % out_dir_at)
os.system('rm -rf %s' % out_dir_at_smooth)
os.system('mkdir -p %s' % out_dir_at)
os.system('mkdir -p %s' % out_dir_at_smooth)

if write_fratoms==True:
    out_dir_fr='%sxyz_fr/' %(in_dir)
    os.system('rm -rf %s' % out_dir_fr)
    os.system('mkdir -p %s' % out_dir_fr)



start=datetime.now()
#--------------------------------------------


print (' #####################################################################')
print (' #                                                                   #')
print (' #                           Ftatons2atoms                           #')
print (' #                                                                   #')
print (' #####################################################################')
print ('')
print ('                   Conversion for min a0=%.3f' %a0 )
print ('')
print ('       - Filter 1:      standard deviation = %.2f' %sigma_gaussian )
print ('                        mask = %s' %mask )
print ('       - Filter 2:      size window  = %d' %size_window)
print ('       - Filter 3:      power mu  = %.2f' %mult)
print ('                        power nu  = %.2f' %nult)
print ('                        grid  m   = %d' %m_grid_size)
print ('       - Correction:    critical distance  = %.3f' %dist_to_remove)



files2read = select_type_files(in_dir, ftype, fkey)
print ('     ')
print (' --------------------------------------------------------------')
print ('                        DIRECTORY: %s ' %in_dir)
print ('                     Number of files to read: %d ' %len(files2read))
print (' --------------------------------------------------------------')
print ('     ')
print ('     No    File             N at1    N at2   N close      err   N at3    N close     err')

density=[]
o2=[]
coor = []
coor2 = []
for i in range(int(len(files2read))):
    f = files2read[i]

    file_h5 = in_dir + f
    np_dset = read_hdf5(file_h5)
    
    # first filter ....: define sigma : but probably should be desactivated. O(N_fratons) operation
    np_dset_gaussian = gaussian_filter(np_dset, sigma=sigma_gaussian, order=0, mode='wrap')

    # convolution home-made filter with masks. O(N_fratons) operation
    if mask !='none':
        np_dset_gaussian = convolution_sharpen(np_dset_gaussian, mask)

    """
    #multiple shaprpen not tested version ... but priobably work if only shifted ...
    np_dset_gaussian0 = convolution_sharpen(np_dset)
    np_dset_gaussian = convolution_sharpen(np_dset_gaussian0)
    """
    #print(np_dset_gaussian.shape)
    #np_dset_gaussian = np_dset



    #second filter: define d_grid -> O(N_fratons) operation
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

    #test if there are some remaining small distances -> O(N_atoms * ln(N_atoms)*(k-1)) operation
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

    #third filter: define mult & nult -> O(4*N_atoms*mult**3) operation
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
    #test if there are some remaining small distances -> O(N_atoms * ln(N_atoms*(k-1)) operation
    tree2= KDTree(coords_end, leaf_size=10)
    dist, ind = tree2.query(coords_end, k=2)
    toot2 = dist[(0 < dist) & (dist < dist_to_remove)]
    coor2.append(int(len(toot2)))

    print("%7d %s %8d %8d %8d %8d %8d %8d %8d" % (i+1, f, len(coords), len(coords_ini), removed_ini, len(toot), len(coords_end), removed_end, len(toot2)))
    #print("%7d %s %8d %8d %8d %8d %8d " % (i+1, f, len(coords), len(coords_ini), removed_ini, len(coords_end), removed_end))


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
print (datetime.now()-start)

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



import numpy as np
import h5py
import os
import sys
from sklearn.neighbors import KDTree
from scipy.ndimage import gaussian_filter
from lib_fr import *


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



#--------------Inputs-----------------

# Directory with *.h5 files to analyse
in_dir = sys.argv[1]
#in_dir='MD_02/'

#First Gaussian filter: the standard value:
sigma_gaussian=1.1

#the size of the window in best_guess function:
size_window=2

#skewnes of the filter ... typical values 1, 2, 3 ... now 2.
mult=2.0

# the size of grid around central fraton to smooth: 2 is nice: 0 - no filter, 1 - why not ; 2 or 3 nice choices. 
m_grid_size = 2


#--------------------------------------------


out_dir_at='%s/xyz_at/' %(in_dir)
out_dir_at_smooth='%s/xyz_at_smooth/' %(in_dir)
key = 'drp'
ftype='.h5'
os.system('rm -rf %s' % out_dir_at)
os.system('rm -rf %s' % out_dir_at_smooth)
os.system('mkdir -p %s' % out_dir_at)
os.system('mkdir -p %s' % out_dir_at_smooth)

files2read = select_type_files(in_dir, ftype)
print ('     ')
print (' ----------------------- DIRECTORY: %s ------------------------' %in_dir)
print ('  >>  Number of files to read: ............................... ', len(files2read))


density=[]
#free=np.loadtxt('MD_01/free_energy.dat')
#debug print(free[:,0])



o1=[]
o2=[]
o3=[]
o4=[]
o5=[]
o6=[]


#for i in range(int(len(files2read)/10)):
#    f=files2read[10*i]

coor = []
coor2 = []
fig = plt.figure()
ax = fig.add_subplot(111)
ims = []
#for i in range(int(len(files2read))):
for i in range(int(len(files2read))):
#for i in range(1):
    f = files2read[i]

    file_h5 = in_dir + f
    np_dset = read_hdf5(file_h5, key)

    # first filter ....: define sigma
    np_dset_gaussian = gaussian_filter(np_dset, sigma=sigma_gaussian, order=0, mode='wrap')

    #np_dset_gaussian = gaussian_filter(np_dset, sigma=2.1, order=0, mode='wrap')

    #second filter: define d_grid
    coords, sel_values = get_best_guess(np_dset_gaussian, d_grid=size_window)
    np_dset_gaussian = np_dset
    o2.append(len(sel_values))


    tree = KDTree(coords, leaf_size=10)
    dist, ind = tree.query(coords, k=7)
    toot = dist[(0 < dist) & (dist < 4.1)]
    coor.append(int(len(toot)))


    name=f.split('.')[0]

    #third ... filter: define m
    coords_weighted = np.array(coords, dtype=float)
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
                    factor = factor + val_dens**mult
                    tmp_i = tmp_i +  val_dens**mult * float(i1uf)
                    tmp_j = tmp_j +  val_dens**mult * float(i2uf)
                    tmp_k = tmp_k +  val_dens**mult * float(i3uf)
                    #print(i1,i2,i3)

        coords_weighted[ic] = np.array([tmp_i, tmp_j, tmp_k])/factor


    tree2= KDTree(coords_weighted, leaf_size=10)
    dist, ind = tree2.query(coords_weighted, k=7)
    toot2 = dist[(0 < dist) & (dist < 4.1)]
    coor2.append(int(len(toot2)))

    print("%7d %s %8d %8d %8d" % (i+1, f, o2[-1], int(len(toot)),int(len(toot2))  ))
    
        
    #writing brute coordinates after 2 filters
#    if write_non-smooth_xyz = True:
    write_atoms_xyz(np_dset, 0.0, name[5:], index_at=coords,   out_dir=out_dir_at)

    #writing smooth  coordinates after 3 filters
    write_atoms_smooth_xyz(np_dset, 0.0, name[5:], index_at=coords, r_at=coords_weighted,  out_dir=out_dir_at_smooth)
    
print ('     ')    
print ('  >>  Non-smooth coordinates are written to: ................. ', out_dir_at)
print ('  >>  Smooth coordinates are written to: ..................... ', out_dir_at_smooth)
print ('     ')
print ('----------------------- ENJOY!!! ------------------------')
print ('     ')


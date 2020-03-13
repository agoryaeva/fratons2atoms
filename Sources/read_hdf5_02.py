#<-------------------AMG, MCM 2020----------------------->#
#                  reading of h5 files                    #
#                writting in xyz format                   #
#---------------------------------------------------------#


import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
import sys
from lib_fr import *


usage = """
     USAGE:
     -------------------------------------------------
     python read_hdf5_01 threshold

     threshold - is the threshold in density from which the atoms are written
"""



if (len(sys.argv) != 2):
    print("\n\tERROR: Wrong number of arguments !!! ")
    print(usage)
    exit(0)

threshold = float(sys.argv[1])

in_dir = 'MD_01/'
out_dir_fr = 'MD_out_fr/'
out_dir_at = 'MD_out_at/'
key = 'drp'
ftype = ".h5"
os.system('rm -rf %s' % out_dir_at)
os.system('rm -rf %s' % out_dir_fr)
os.system('mkdir -p %s' % out_dir_at)
os.system('mkdir -p %s' % out_dir_fr)

#file_h5 = 'MD_01/drp__00200000.h5'
files2read = select_type_files(in_dir, ftype)
print('     ')
print('  >>  Number of files to read: ............................... ', len(files2read))


density = []
free = np.loadtxt('MD_01/free_energy.dat')
#debug print(free[:,0])

neg_sum=[]
#for i in range(int(len(files2read) / 10)):
for i in range(int(len(files2read))):
#for i in range(100,101):
    #print(i, files2read[10 * i])
    #f = files2read[10 * i]
    f = files2read[i]
    file_h5 = in_dir + f
    np_dset = read_hdf5(file_h5, key)
    nsum = np.sum(np_dset [ np_dset < 0 ])

    idx = largest_indices(np_dset, np_dset.shape[0]**3)
    values = np_dset[idx]
    epsilon = 1.e-1
    np_dset_shifted = np_dset + abs(np.min(np_dset))
    le = entropy(np_dset[np_dset > epsilon].flatten(), 300.0)
    total_entropy = np.sum(le)

    le = entropy(np_dset_shifted[np_dset > epsilon].flatten(), 300.0)
    total_entropy_shifted = np.sum(le)
    density.append(np.sum(np_dset))
    neg_sum.append(nsum)
    print(f,np.sum(np_dset [ np_dset < 0 ]), total_entropy, total_entropy_shifted, np.max(np_dset), np.min(np_dset))
#Setting the thershold beyond that the atoms are plotted ...
#    print("If I take a imposet threshold given by %2f the number of atoms .......:" % threshold, np_dset[np_dset > threshold].shape)
    name = f.split('.')[0]
    #write_fratons_xyz(np_dset, name[5:], out_dir=out_dir_fr)
    #coords, sel_values = get_best_guess(np_dset, d_grid=2)
    #write_atoms_xyz(np_dset, 0.0, name[5:], index_at=coords, out_dir=out_dir_at)



fig = plt.figure()
ax = fig.add_subplot(111)
"""
ax.plot(neg_sum, '-', color='lightsteelblue', markersize=1, label='')
ax.legend(loc='best', fontsize=10, frameon=False)
ax.set_xlabel(r" i simulation ", fontsize=20)
ax.set_ylabel(r"$\sum (\rho  < 0)$   ", fontsize=20) #, color='slateblue')
#plt.ylabel(r" $\Sigma$ ($\rho$  < 0)")
"""

ax.plot(values, 'o', color='lightsteelblue', markersize=1, label='')
ax.legend(loc='best', fontsize=10, frameon=False)
ax.set_xlabel(r" i fraton ", fontsize=20)
ax.set_ylabel(r"$\rho_i$  ", fontsize=20) #, color='slateblue')




"""
fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(free[:, 0], 'o', color='lightsteelblue', markersize=1, label='free')
ax.legend(loc='best', fontsize=10, frameon=False)
ax = fig.add_subplot(132)
ax.plot(free[:, 1], 'o', color='lightsteelblue', markersize=1, label='internal')
ax.legend(loc='best', fontdsize=10, frameon=False)

ax = fig.add_subplot(133)
ax.plot(free[:, 2], 'o', color='lightsteelblue', markersize=1, label='entropy')
ax.legend(loc='best', fontsize=10, frameon=False)
#ax.set_xlabel(r" site index ", fontsize=20)
#ax.set_ylabel(r" Density  ", fontsize=20) #, color='slateblue')
"""
plt.show()

#<------------------------------------------------------->#
#<----               Fratons2atoms                   ---->#
#<---    Alexandra M. Goryaeva, M. C. Marinica ---------->#
#---------------------------------------------------------#



import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
import sys
from lib_fr import *
from sklearn.neighbors import KDTree
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.ndimage import gaussian_filter

usage="""
     USAGE:
     -------------------------------------------------
     python read_hdf5_01 threshold

     threshold - is the threshold in density from which the atoms are written
"""


"""
if (len(sys.argv) != 2):
    print("\n\tERROR: Wrong number of arguments !!! ")
    print(usage)
    exit(0)

threshold = float(sys.argv[1])
"""

in_dir='MD_01/'
out_dir_fr='MD_out_fr/'
out_dir_at='MD_out_at/'
key = 'drp'
ftype='.h5'
os.system('rm -rf %s' % out_dir_at)
os.system('rm -rf %s' % out_dir_fr)
os.system('mkdir -p %s' % out_dir_at)
os.system('mkdir -p %s' % out_dir_fr)

#file_h5 = 'MD_01/drp__00200000.h5'
files2read = select_type_files(in_dir, ftype)
print ('     ')
print ('  >>  Number of files to read: ............................... ', len(files2read))


density=[]
free=np.loadtxt('MD_01/free_energy.dat')
#debug print(free[:,0])



o1=[]
o2=[]
o3=[]
o4=[]
o5=[]
o6=[]


#for i in range(int(len(files2read)/10)):
#    f=files2read[10*i]

coor=[]
fig=plt.figure()
ax=fig.add_subplot(111)
ims=[]
for i in range(int(len(files2read))):
#for i in range(1):
    f = files2read[i]

    file_h5 = in_dir + f
    np_dset = read_hdf5(file_h5, key)
    # first filter ...
    np_dset_gaussian = gaussian_filter(np_dset, sigma=2.1, order=0, mode='wrap')



    #second filter ...
    coords, sel_values = get_best_guess(np_dset_gaussian, d_grid=2)



    name=f.split('.')[0]
    write_fratons_xyz(np_dset_gaussian, name[5:], out_dir=out_dir_fr)
    write_atoms_xyz(np_dset, 0.0, name[5:], index_at=coords,   out_dir=out_dir_at)




ax=fig.add_subplot(121)

ax.plot(o2, '-', color='darkblue', markersize=1, label='o=2' )
ax.plot(coor, '-', color='skyblue', markersize=1, label='very close' )
plt.show()
"""




ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=500)
#writer = PillowWriter(fps=20)
#ani.save("demo2.gif", writer=writer)
ani.save("demo2.gif", writer='imagemagick', fps=50)
plt.show()



ax.plot(o1, '-', color='skyblue', markersize=1, label='o=1' )
ax.plot(o3, '-', color='m', markersize=1, label='o=3' )
ax.plot(o4, '-', color='lightpink', markersize=1, label='o=4' )
#ax.plot(o5, '-', color='y', markersize=1, label='o=5' )
"""

exit(0)

ax=fig.add_subplot(122)

ax.hist(dist[dist >0].flatten(), 40, density=True, facecolor='g', alpha=0.75)
ax.set_xlim(2,8)
ax.legend(loc='best', fontsize=10, frameon=False)
#ax=fig.add_subplot(132)
#ax.plot(free[:,1], 'o', color='lightsteelblue', markersize=1, label='internal' )
#ax.legend(loc='best', fontsize=10, frameon=False)

#ax=fig.add_subplot(133)
#ax.plot(free[:,3], 'o', color='lightsteelblue', markersize=1, label='entropy' )
#ax.legend(loc='best', fontsize=10, frameon=False)
#ax.set_xlabel(r" site index ", fontsize=20)
#ax.set_ylabel(r" Density  ", fontsize=20) #, color='slateblue')

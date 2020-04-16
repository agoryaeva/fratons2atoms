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

usage="""
     USAGE:
     -------------------------------------------------
     python read_hdf5_01 threshold

     threshold - is the threshold in density from which the atoms are written



if (len(sys.argv) != 2):
    print("\n\tERROR: Wrong number of arguments !!! ")
    print(usage)
    exit(0)

threshold = float(sys.argv[1])

"""


def update_hist(num,data):

    #ax.hist(dist[dist >0].flatten(), 40, density=True, facecolor='g', alpha=0.75)
    #ax.set_xlim(2,8)
    plt.cla()
    plt.xlim(2,8)
    plt.ylim(0,1000)
    plt.text(3,800, 'num = %d' % num)
    #plt.hist(data[num], 40, density=True, facecolor='g', alpha=0.75)
    plt.hist(data[num],60, facecolor='chartreuse')





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



o2=[]


#for i in range(int(len(files2read)/10)):
#    f=files2read[10*i]

coor=[]
data=[]
for i in range(int(len(files2read))):
#for i in range(4):
    f = files2read[i]

    file_h5 = in_dir + f
    np_dset = read_hdf5(file_h5, key)
    coords, sel_values = get_best_guess(np_dset, d_grid=2)
    o2.append(len(sel_values))
    neg_dist = np_dset [ np_dset < 0 ]
    tree = KDTree(coords, leaf_size=10)
    #print(coords[:])
    dist, ind = tree.query(coords, k=7)
    toot = dist[(0 < dist) & (dist < 4.1)]
    coor.append(int(len(toot)))
    data.append(dist.flatten())
    #print("%7d %s %8d %8d" % (i, f, o2[-1], int(len(toot))))
    print("%8d %s %8d %8d %8d %.7f" % (i, f, o2[-1], int(len(toot)), int(len(neg_dist)), np.sum(neg_dist)  ))
    #print(dist[dist >0].flatten())
    #print(f,np.sum(np_dset))
    #density.append(np.sum(np_dset))
#Setting the thershold beyond that the atoms are plotted ...
#    print("If I take a imposet threshold given by %2f the number of atoms .......:" % threshold, np_dset[np_dset > threshold].shape)
    name=f.split('.')[0]
    #write_fratons_xyz(np_dset, name[5:], out_dir=out_dir_fr)
    #write_atoms_xyz(np_dset, name[5:], out_dir=out_dir_at)

"""
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=500)
#writer = PillowWriter(fps=20)
#ani.save("demo2.gif", writer=writer)
ani.save("demo2.gif", writer='imagemagick', fps=50)
"""
number_of_frame = 100
fig=plt.figure()
hist = plt.hist(data[0])
#ax=fig.add_subplot(111)
animation = animation.FuncAnimation(fig, update_hist, number_of_frame, fargs=(data, ) )
animation.save("demo2.gif", writer='imagemagick', fps=2)
plt.show()

"""
ax=fig.add_subplot(121)

print(o2)
ax.plot(o2, '-', color='darkblue', markersize=1, label='o=2' )
ax.plot(coor, '-', color='skyblue', markersize=1, label='very close' )
"""


"""
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

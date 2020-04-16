#<------------------------------------------------------->#
#<----       Fratons2atoms Library                   ---->#
#<---    Alexandra M. Goryaeva, M. C. Marinica ---------->#
#---------------------------------------------------------#


import numpy as np
import  h5py
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage as ndi



def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)









db=h5py.File('MD_01/drp__00200000.h5', 'r')
#db=h5py.File('MD_01/drp__00204000.h5', 'r')
print("The keys of the HDF5 file...............:", db.keys())
dset = db ['drp']
print("The contenet of the DB..................:", dset)

#--------------If the selection is done by the Npart fratons with the highest density ...
#Because I didn't know I have taken the sum of the densities ...
print("The sum of densities is ................:",np.sum(dset))
Npart = int(np.sum(dset))
np_dset = np.asarray(dset, dtype=float)
print("Shape of the data ................:", np_dset.shape)

Npart=31192
idx = largest_indices(np_dset, Npart)
print("The number of atoms should be...............................:",Npart)
print("And the threshold enbling the selection of Npart atoms......:",np_dset[idx][-1] )

#If I take as thershold 0.7




threshold = 0.9
print("If I take a imposet threshold given by %2f the number of atoms .......:"%threshold,np_dset [np_dset > threshold ].shape)



# I want to print the sorted density against the atomic index ... for all fratoms ...
idx = largest_indices(np_dset, np_dset.shape[0]**3)
values = np_dset[idx]

fig=plt.figure()
ax=fig.add_subplot(121)
ax.plot(values, 'o', color='lightsteelblue', markersize=1, label='densities vs site index' )

coords, sel_values = local_maxima_3D(np_dset, order=1)

ax=fig.add_subplot(122)
ax.plot(sel_values, 'o', color='lightsteelblue', markersize=1, label='local maxima' )
ax.set_xlabel(r" site index ", fontsize=20)
ax.set_ylabel(r" Density  ", fontsize=20) #, color='slateblue')

plt.show()

print(len(sel_values))
print(sel_values)

exit(0)

indx = np.where(np_dset > 0.95)

X = np.asarray(indx, dtype=float).T
print(X)
Y = np_dset[indx]

print(X.shape)
print(Y.shape)

kmeans = KMeans(n_clusters=15000, random_state=0, max_iter=10)
wt_kmeansclus = kmeans.fit(X,sample_weight = Y)
predicted_kmeans = kmeans.predict(X, sample_weight = Y)
centers = wt_kmeansclus.cluster_centers_

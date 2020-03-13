#<------------------------------------------------------->#
#<----               Fratons2atoms                   ---->#
#<---    Alexandra M. Goryaeva, M. C. Marinica ---------->#
#---------------------------------------------------------#

import numpy as np
import  h5py
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage



def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)





#db=h5py.File('MD_01/drp__00200000.h5', 'r')
db=h5py.File('MD_01/drp__00204000.h5', 'r')
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
ax=fig.add_subplot(111)


ax.plot(values, 'o', color='lightsteelblue', markersize=1, label='densities vs site index' )
ax.set_xlabel(r" site index ", fontsize=20)
ax.set_ylabel(r" Density  ", fontsize=20) #, color='slateblue')

plt.show()


img = np_dset

# Get local maximum values of desired neighborhood
# I'll be looking in a 5x5x5 area
img2 = ndimage.maximum_filter(img, size=(8, 8, 8))

print(img2)

# Threshold the image to find locations of interest
# I'm assuming 6 standard deviations above the mean for the threshold
img_thresh = img2.mean() + img2.std() * 6

# Since we're looking for maxima find areas greater than img_thresh

labels, num_labels = ndimage.label(img2 > img_thresh)

# Get the positions of the maxima
coords = ndimage.measurements.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))

# Get the maximum value in the labels
values = ndimage.measurements.maximum(img, labels=labels, index=np.arange(1, num_labels + 1))


print(values)

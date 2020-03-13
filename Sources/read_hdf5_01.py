#<------------------------------------------------------->#
#<----       Fratons2atoms Library                   ---->#
#<---    Alexandra M. Goryaeva, M. C. Marinica ---------->#
#---------------------------------------------------------#

import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
import sys
from lib_fr import *


usage="""
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

#file_h5 = 'MD_01/drp__00200000.h5'
file_h5 = 'MD_01/drp__00203840.h5'
key = 'drp'


np_dset = read_hdf5(file_h5,key)
#Setting the thershold beyond that the atoms are plotted ...
print("If I take a imposet threshold given by %2f the number of atoms .......:" % threshold, np_dset[np_dset > threshold].shape)


write_atoms_xyz(np_dset, threshold, 'first')
write_fratons_xyz(np_dset, 'first')

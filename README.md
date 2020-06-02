# fratons2atoms

python package to convert atomic densities ("fratons") from quasiparticle calculations to atomc positions


Purposes:
-----------------
Conversion of binary h5 files with atomic densities  ("fratons") to xyz coordinates that can be visualized and analyzed with ovito and serve as an input for atomistic calculations

Description
----------------

Usage:
-----------------
### Running the script with default options:

```
> python3 fratons2atoms.py <path> <a0>
```
  
With this command the script `fratons2atoms.py` will read the h5 files in the `<path>` directory and set the default parameters for the convertion based on the given `<a0>` in fraton units. In case the simulation cell contains more than one phase with different `<a0>`, the smallest one should be provided. 
  
  ##### Examaple: 
  
  To convert the calculations of the transition between fcc with a0=8.0 and bcc with a0=6.5 that are stored in Examples/BCC-6.5_FCC_8/ 
  
 ``` 
 > python3 fratons2atoms.py  Examples/BCC-6.5_FCC_8/  6.5
 ``` 
### Running the script with custom parameters:

  ##### Examaple: 

### Notes on choosing parameters:


Installation:
-----------------
To install from the terminal, use:

```
> git clone https://github.com/agoryaeva/fratons2atoms.git
```

Authors:
----------
A. M. Goryaeva, M.-C. Marinica


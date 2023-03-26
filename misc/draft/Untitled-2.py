# %matplotlib inline 
from scipy import *
from scipy.io import *
from numpy import *
from matplotlib.pyplot  import *

import os
path = os.path.abspath(os.path.join(os.getcwd())).replace("\\", "/")
path = path + "/models/bunny1000_2000/"
print(path)
P = mmread(path + "P1.mtx")
R = mmread(path + "R.mtx")

fig2 = figure("Figure 2")
spy(P, markersize=0.1)
show()
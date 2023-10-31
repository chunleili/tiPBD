# %matplotlib inline
from scipy import *
from scipy.io import *
from numpy import *
from matplotlib.pyplot import *

import os

# path = os.path.abspath(os.path.join(os.getcwd())).replace("\\", "/")
# path = path + "/model/bunny1k2k/"
# print(path)
# P = mmread(path + "P1.mtx")
R = mmread("R.mtx")

fig = figure("pattern of R")
spy(R, markersize=0.1)
show()

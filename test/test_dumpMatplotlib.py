import pickle   
import matplotlib.pyplot as plt

path = "D:/Dev/tiPBD/result/case180-0308-bunny-interval300/A/plot.pickle"

import pickle
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show()
plt.show()
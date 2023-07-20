import numpy as np

np.savez("draft2.npz", 111, 123)
a = np.load("draft2.npz")
print(a["arr_0"], a["arr_1"])

import pickle
import taichi as ti

data = {"a": [1, 2.0, 3, 4 + 6], "b": ("character string", b"byte string"), "c": {None, True, False}}

with open("data.pickle", "wb") as f:
    pickle.dump(data, f)

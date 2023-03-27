import numpy as np
from engine.metadata import meta
def debug_info(field):
    if meta.debug == True:
        field_np = field.to_numpy()
        print("---------------------")
        print("name: ", field._name )
        print("shape: ",field_np.shape)
        print("min, max: ", field_np.min(), field_np.max())
        print(field_np)
        print("---------------------")
        np.savetxt("debug.txt", field_np.flatten(), fmt="%.4f", delimiter="\t")
        return field_np
    else:
        return None
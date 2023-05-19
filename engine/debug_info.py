import numpy as np


def debug_info(field, name="", dont_print_cli=False):
    field_np = field.to_numpy()
    if name == "":
        name = field._name
    print("---------------------")
    print("name: ", name)
    print("shape: ", field_np.shape)
    print("min, max: ", field_np.min(), field_np.max())
    if not dont_print_cli:
        print(field_np)
    print("---------------------")
    if field_np.ndim > 2:
        np.savetxt(f"result/debug_{name}.csv", field_np.reshape(-1, field_np.shape[-1]), fmt="%.2f", delimiter="\t")
    else:
        np.savetxt(f"result/debug_{name}.csv", field_np, fmt="%.2f", delimiter="\t")
    return field_np

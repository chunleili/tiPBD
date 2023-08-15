import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def conjugate_gradient(A, b, x0, max_iterations=100, tolerance=1e-12):
    r_norm_list = []

    # initial residual and direction
    x = x0
    r = b - A @ x
    p = r

    r_norm = np.linalg.norm(r)
    print(f"initial r:{r_norm}")

    r_norm_list.append(r_norm)

    for iter in range(max_iterations):
        # compute alpha
        Ap = A @ p
        rTr = np.dot(r.T, r)
        pTAp = p.T @ Ap
        alpha = rTr / pTAp

        # update x
        x = x + alpha * p
        r_new = r - alpha * Ap

        r_norm = np.linalg.norm(r)
        r_norm_list.append(r_norm)
        print(f"iter:{iter}, r:{r_norm}")
        if np.linalg.norm(r) < tolerance:
            print(f"Converged at r: {r_norm}")
            return x, r_new, r_norm_list

        rTr_new = np.dot(r_new.T, r_new)
        beta = rTr_new / rTr

        p_new = r_new + beta * p

        # x = x_new.copy()
        p = p_new.copy()
        r = r_new.copy()

    print(f"Reach max iterations r:{r_norm}")
    return x, r_new, r_norm_list


# prepare data
n = 100  # Size of the matrix
random_matrix = np.random.rand(n, n)
symmetric_matrix = random_matrix @ random_matrix.T  # Generate a symmetric matrix
positive_definite_matrix = (
    symmetric_matrix + np.eye(n) * 0.1
) * 10  # Add a small diagonal to make it positive definite
print("Random Symmetric Positive Definite Matrix:")
print(positive_definite_matrix)
A = positive_definite_matrix
b = np.random.rand(n)

t = perf_counter()
x1 = np.linalg.solve(A, b)
t1 = perf_counter() - t

t = perf_counter()
x0 = np.random.rand(b.shape[0]) * 1.0
x2, r, r_norm_list = conjugate_gradient(A, b, x0)
t2 = perf_counter() - t


print("x1: ", x2)
print("x2: ", x2)

print("t1: ", t1)
print("t2: ", t2)

# plot the residual decrease with iterations
plt.plot(np.array(r_norm_list))
plt.xlabel("iter")
plt.ylabel("residual")
plt.yscale("log")
plt.show()

print(f"max diff with direct solver: {np.linalg.norm(x1-x2, np.inf)}")
assert np.allclose(x1, x2, atol=1e-4), f"max diff: {np.linalg.norm(x1-x2, np.inf)}"
print("Value is verified")

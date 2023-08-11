import numpy as np


def solve_sor(A, b, x0, omega, max_iterations, tol):
    n = A.shape[0]
    x = x0.copy()

    # D = np.diag(A)
    # L = np.tril(A, k=-1)
    # U = np.triu(A, k=1)
    # Lw = np.linalg.inv(D + omega * L) @ (- omega * U + (1 - omega) * D )
    # spectral_radius_Lw = max(abs(np.linalg.eigvals(Lw)))
    # print(f"spectral radius of Lw: {spectral_radius_Lw:.2f}")

    for iteration in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (
                b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1 :], x[i + 1 :])
            )

        residual = np.linalg.norm(A @ x_new - b)

        print(f"iter: {iteration}, res: {residual:.2e}")

        if residual < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return x_new

        x = x_new

    print("Did not converge within the maximum number of iterations.")
    return x


def test_sor():
    # Example usage
    A = np.array([[4, -1.0, 0.0, -1.0], [-1.0, 4, -1.0, 0.0], [0.0, -1.0, 4, -1.0], [-1.0, 0.0, -1.0, 4]])

    # spectral radius of A
    spectral_radius = max(abs(np.linalg.eigvals(A)))
    print(f"spectral radius of A: {spectral_radius:.2f}")

    b = np.array([10.0, 10.0, 10.0, 10.0])
    x0 = np.zeros(4)
    omega = 1.2  # SOR relaxation parameter
    max_iterations = 100
    tolerance = 1e-6

    solution = solve_sor(A, b, x0, omega, max_iterations, tolerance)
    print("Solution:", solution)

    x_true = np.linalg.solve(A, b)
    print("True Solution By DirectSolver:", x_true)


test_sor()

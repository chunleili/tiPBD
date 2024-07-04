# include <vector>
# include <array>
# include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Create rigid body modes from coordinate vector.
// To be used as near-nullspace vectors with aggregation coarsening
// for 2D or 3D elasticity problems.
// The output matrix B may be transposed on demand
// (to be used as a set of deflation vectors).
std::vector<double> rigid_body_modes(int ndim, const std::vector<double> &coo, std::vector<double> &B, bool transpose = false) {
    size_t n = coo.size();
    int nmodes = (ndim == 2 ? 3 : 6);
    B.resize(n * nmodes, 0.0);

    const int stride1 = transpose ? 1 : nmodes;
    const int stride2 = transpose ? n : 1;

    double sn = 1 / sqrt(n);

    if (ndim == 2) {
        for(size_t i = 0; i < n; ++i) {
            size_t nod = i / ndim;
            size_t dim = i % ndim;

            double x = coo[nod * 2 + 0];
            double y = coo[nod * 2 + 1];

            // Translation
            B[i * stride1 + dim * stride2] = sn;

            // Rotation
            switch(dim) {
                case 0:
                    B[i * stride1 + 2 * stride2] = -y;
                    break;
                case 1:
                    B[i * stride1 + 2 * stride2] = x;
                    break;
            }
        }
    } else if (ndim == 3) {
        for(size_t i = 0; i < n; ++i) {
            size_t nod = i / ndim;
            size_t dim = i % ndim;

            double x = coo[nod * 3 + 0];
            double y = coo[nod * 3 + 1];
            double z = coo[nod * 3 + 2];

            // Translation
            B[i * stride1 + dim * stride2] = sn;

            // Rotation
            switch(dim) {
                case 0:
                    B[i * stride1 + 3 * stride2] = y;
                    B[i * stride1 + 5 * stride2] = z;
                    break;
                case 1:
                    B[i * stride1 + 3 * stride2] = -x;
                    B[i * stride1 + 4 * stride2] = -z;
                    break;
                case 2:
                    B[i * stride1 + 4 * stride2] =  y;
                    B[i * stride1 + 5 * stride2] = -x;
                    break;
            }
        }
    }

    // Orthonormalization
    std::array<double, 6> dot;
    for(int i = ndim; i < nmodes; ++i) {
        std::fill(dot.begin(), dot.end(), 0.0);
        for(size_t j = 0; j < n; ++j) {
            for(int k = 0; k < i; ++k)
                dot[k] += B[j * stride1 + k * stride2] * B[j * stride1 + i * stride2];
        }
        double s = 0.0;
        for(size_t j = 0; j < n; ++j) {
            for(int k = 0; k < i; ++k)
                B[j * stride1 + i * stride2] -= dot[k] * B[j * stride1 + k * stride2];
            s += B[j * stride1 + i * stride2] * B[j * stride1 + i * stride2];
        }
        s = sqrt(s);
        for(size_t j = 0; j < n; ++j)
            B[j * stride1 + i * stride2] /= s;
    }


    // print B
    for(size_t i = 0; i < n; ++i) {
        for(int j = 0; j < nmodes; ++j) {
            std::cout << B[i * stride1 + j * stride2] << " ";
        }
        std::cout << std::endl;
    }

    return B;
}




PYBIND11_MODULE(rigid_body_modes, m) {
    m.def("rigid_body_modes", &rigid_body_modes, "rigid_body_modes");
}

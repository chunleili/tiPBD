
#pragma once

#include "Vec.h"
#include "CSR.h"
#include <array>

namespace fastmg
{

struct MGLevel {
    CSR<float> A;
    CSR<float> R;
    CSR<float> P;
    Vec<float> residual;
    Vec<float> b;
    Vec<float> x;
    Vec<float> h;
    Vec<float> outh;
    CSR<float> Dinv;
    CSR<float> Aoff;
    float scale_RAP=0.0;
    float jacobi_omega=2.0/3.0;
    std::array<float,3> chebyshev_coeff;
    Vec<int> colors; // color index of each node
    int color_num; // number of colors, max(colors)+1
};

} // namespace fastmg

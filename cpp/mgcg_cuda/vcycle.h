#pragma once

#include <vector>
#include "cusparse_wrappers.h"

namespace fastmg
{
struct MGLevel;
struct Smoother;

struct VCycle : CusparseWrappers {
    VCycle(std::vector<MGLevel> &levels,
     std::shared_ptr<Smoother> smoother
    ) : levels(levels), smoother(smoother){}

    size_t coarse_solver_type = 0; //0:direct solver by cusolver (cholesky), 1: one sweep smoother
    void run(Vec<float> &xf, Vec<float> &bf); // A@xf = bf or A@z=r
    
    
    private:
    std::vector<MGLevel> &levels;
    std::shared_ptr<Smoother> smoother;
    Buffer buff;

    void coarse_solve(const CSR<float> &A, Vec<float> &x, const Vec<float> &b);

};

} // namespace fastmg
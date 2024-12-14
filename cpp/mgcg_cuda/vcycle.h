#pragma once

#include <vector>
#include "cusparse_wrappers.h"

namespace fastmg
{
struct MGLevel;
struct Smoother;

struct VCycle : CusparseWrappers {
    VCycle(std::vector<MGLevel> &levels,
     std::shared_ptr<Smoother> smoother,
      Vec<float> &z, Vec<float> &r, Buffer &buff
    ) : levels(levels), smoother(smoother), z(z), r(r), buff(buff){}

    size_t coarse_solver_type = 1; //0:direct solver by cusolver (cholesky), 1: one sweep smoother

    void run();

    Vec<float> &z;
    Vec<float> &r;
    Buffer &buff;

private:
    std::vector<MGLevel> &levels;
    std::shared_ptr<Smoother> smoother;

    void vcycle_down();
    void vcycle_up();
    void coarse_solve();

};

} // namespace fastmg
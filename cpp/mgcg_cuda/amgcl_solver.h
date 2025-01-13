#pragma once
#include "common.h"
#include "SpMatData.h"
#include "linear_solver.h"

struct AmgclSolver:LinearSolver
{
    std::vector<SpMatData> Ps;
    // std::vector<std::shared_ptr<amgcl::backend::crs<float, ptrdiff_t, ptrdiff_t>>> Ps;
    virtual Field1f solve(SpMatData* A, Field1f& b);

    bool should_setup = true;
};


#pragma once
#include "common.h"
#include "SpMatData.h"
#include "linear_solver.h"

struct AmgclSolver:LinearSolver
{
    std::vector<SpMatData> Ps;
    virtual Field1f solve(SpMatData* A, Field1f& b);
};


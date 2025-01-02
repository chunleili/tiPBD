#pragma once
#include "fastmg.h"
#include "common.h"
#include "linear_solver.h"

namespace fastmg
{
/// A wrapper for the fastmg solver, which use the amgcl as the setup phase 
struct AmgCudaSolver:LinearSolver
{
    AmgCudaSolver(){}
    std::vector<SpMatData*> m_Ps;
    std::vector<SpMatData> m_Ps_data;
    Field1f solve(SpMatData* A, Field1f& b, bool should_setup=true);
    std::vector<SpMatData*> setup(SpMatData* A);
};
}

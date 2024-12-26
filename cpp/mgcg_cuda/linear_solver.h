#pragma once
#include "common.h"
#include "SpMatData.h"

struct LinearSolver
{
    float rtol = 1e-5;
    int maxiter = 100;
    std::vector<float> residuals;
    std::vector<float> solution;
    int niter;
    
    virtual void run(SpMatData* A, Field1f& b, bool should_setup=true)=0;
};


struct EigenSolver:LinearSolver
{
    virtual void run(SpMatData* A, Field1f& b, bool should_setup=true);
};

struct AmgclSolver:LinearSolver
{
    std::vector<SpMatData> Ps;
    virtual void run(SpMatData* A, Field1f& b, bool should_setup=true);
};


namespace fastmg
{
/// wrapper for the AmgCuda
struct AmgCudaSolver:LinearSolver
{
    std::vector<SpMatData> Ps;
    bool should_setup = true;
    virtual void run(SpMatData* A, Field1f& b, bool should_setup=true);
};
}
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
    bool verbose = false;
    
    virtual Field1f solve(SpMatData* A, Field1f& b)=0;
};

#include "vcycle.h"
#include "smoother.h"
#include "mglevel.h"
#include "cusparse_wrappers.h"

namespace fastmg
{
    /// @brief Run a V-cycle
    /// Input: levels must have all As, Ps, Rs , xf and bf. Especially xf and bf are finest solution and finest right-hand side. 
    /// Output: xf will be the solution of the linear system.
    /// Az=r Ax=b
    void  VCycle::run(Vec<float> &x0, Vec<float> &bf)
    {
        int nl = levels.size();
        x0.resize(levels[0].A.ncols);
        zero(x0);
        copy(levels[0].r, bf);
        copy(levels[0].x, x0);
        for (int l = 0; l < nl - 1; ++l)
        {
            zero(levels[l].x);
      
            smoother->smooth(l, levels[l].x, levels[l].r); 
            
            // r_l+1 = Rl @ (r_l - A_l @ x_l)
            b_Ax(levels[l].A, levels[l].x, levels[l].r, levels[l].r); 
            spmv(levels[l + 1].r, 1, levels[l].R, levels[l].r, 0, buff); // r_{l+1} = R@r_l
        }

        coarse_solve(levels[nl - 1].A, levels[nl - 1].x, levels[nl - 1].r);

        for (int l = nl - 2; l >= 0; --l)
        {
            spmv(levels[l].x, 1, levels[l].P, levels[l + 1].x, 1, buff); // xl += Pl@x_{l+1}
            smoother->smooth(l, levels[l].x, levels[l].r);
        }

        copy(x0, levels[0].x);
    }


    void  VCycle::coarse_solve(const CSR<float> &A, Vec<float> &x, const Vec<float> &b) {
        int nl = levels.size();
        x.resize(A.ncols);
        zero(x);
        if (coarse_solver_type==0)
        {
            spsolve(x, A, b);
        }
        else if (coarse_solver_type==1)
        {
            smoother->smooth(nl-1, x, b);
        }
    }

} // namespace fastmg


#include "vcycle.h"
#include "smoother.h"
#include "mglevel.h"
#include "cusparse_wrappers.h"

namespace fastmg
{
    void  VCycle::run() {
        vcycle_down();
        coarse_solve();
        vcycle_up();
    }

    void  VCycle::vcycle_down() {
        int nlvs = levels.size();
        for (int lv = 0; lv < nlvs-1; ++lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : z;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : r;

            smoother->smooth(lv, x, b);

            copy(levels.at(lv).residual, b);
            spmv(levels.at(lv).residual, -1, levels.at(lv).A, x, 1, buff); // residual = b - A@x

            levels.at(lv).b.resize(levels.at(lv).R.nrows);
            spmv(levels.at(lv).b, 1, levels.at(lv).R, levels.at(lv).residual, 0, buff); // coarse_b = R@residual

            levels.at(lv).x.resize(levels.at(lv).b.size());
            zero(levels.at(lv).x);
        }
    }

    void  VCycle::vcycle_up() {
        int nlvs = levels.size();
        for (int lv = nlvs-2; lv >= 0; --lv) {
            Vec<float> &x = lv != 0 ? levels.at(lv - 1).x : z;
            Vec<float> &b = lv != 0 ? levels.at(lv - 1).b : r;
            spmv(x, 1, levels.at(lv).P, levels.at(lv).x, 1, buff); // x += P@coarse_x
            smoother->smooth(lv, x, b);
        }
    }

    void  VCycle::coarse_solve() {
        int nlvs = levels.size();
        auto const &A = levels.at(nlvs - 1).A;
        auto &x = levels.at(nlvs - 2).x;
        auto &b = levels.at(nlvs - 2).b;
        if (coarse_solver_type==0)
        {
            spsolve(x, A, b);
        }
        else if (coarse_solver_type==1)
        {
            smoother->smooth(nlvs-1, x, b);
        }
    }

} // namespace fastmg

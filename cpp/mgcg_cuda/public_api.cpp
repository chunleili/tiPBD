
#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


extern "C" DLLEXPORT void fastmg_setup_smoothers(int type) {
    fastmg->smoother->setup_smoothers(type);
}


extern "C" DLLEXPORT void fastmg_set_smoother_niter(const size_t niter) {
    fastmg->smoother->set_smoother_niter(niter);
}

extern "C" DLLEXPORT void fastmg_set_colors(const int *c, int n, int color_num, int lv) {
    fastmg->smoother->set_colors(c, n, color_num, lv);
}

extern "C" DLLEXPORT void fastmg_use_radical_omega(int flag) {
    fastmg->smoother->use_radical_omega = bool(flag);
}


extern "C" DLLEXPORT void fastmg_set_coarse_solver_type(int t) {
    fastmg->vcycle->coarse_solver_type = t;
}


extern "C" DLLEXPORT void fastmg_set_A0_from_fastFillCloth() {
    fastmg->set_A0_from_fastFill(fastFillCloth);
}

extern "C" DLLEXPORT void fastmg_set_A0_from_fastFillSoft() {
    fastmg->set_A0_from_fastFill(fastFillSoft);
}



extern "C" DLLEXPORT void fastmg_new() {
    if (!fastmg)
        fastmg = new FastMG{};
}

extern "C" DLLEXPORT void fastmg_setup_nl(size_t numlvs) {
    fastmg->create_levels(numlvs);
}


extern "C" DLLEXPORT void fastmg_RAP(size_t lv) {
    fastmg->compute_RAP(lv);
}


extern "C" DLLEXPORT int fastmg_get_nnz(size_t lv) {
    int nnz = fastmg->get_nnz(lv);
    std::cout<<"nnz: "<<nnz<<std::endl;
    return nnz;
}

extern "C" DLLEXPORT int fastmg_get_matsize(size_t lv) {
    int n = fastmg->get_nrows(lv);
    std::cout<<"matsize: "<<n<<std::endl;
    return n;
}

extern "C" DLLEXPORT void fastmg_fetch_A(size_t lv, float* data, int* indices, int* indptr) {
    fastmg->fetch_A(lv, data, indices, indptr);
}

extern "C" DLLEXPORT void fastmg_fetch_A_data(float* data) {
    fastmg->fetch_A_data(data);
}

extern "C" DLLEXPORT void fastmg_solve() {
    fastmg->solve();
}

extern "C" DLLEXPORT void fastmg_set_data(const float* x, size_t nx, const float* b, size_t nb, float rtol, size_t maxiter) {
    fastmg->set_data(x, nx, b, nb, rtol, maxiter);
}

extern "C" DLLEXPORT size_t fastmg_get_data(float *x, float *r) {
    size_t niter = fastmg->get_data(x, r);
    return niter;
}

extern "C" DLLEXPORT void fastmg_set_A0(float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastmg->set_A0(data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}

// only update the data of A0
extern "C" DLLEXPORT void fastmg_update_A0(const float* data_in)
{
    fastmg->update_A0(data_in);
}

extern "C" DLLEXPORT void fastmg_set_P(int lv, float* data, int* indices, int* indptr, int rows, int cols, int nnz)
{
    fastmg->set_P(lv, data, nnz, indices, nnz, indptr, rows + 1, rows, cols, nnz);
}



extern "C" DLLEXPORT void fastmg_scale_RAP(float s, int lv) {
    fastmg->set_scale_RAP(s, lv);
}


extern "C" DLLEXPORT void fastmg_solve_only_smoother() {
    fastmg->solve_only_smoother();
}


extern "C" DLLEXPORT void fastmg_solve_only_jacobi() {
    fastmg->solve_only_jacobi();
}

extern "C" DLLEXPORT void fastmg_solve_only_directsolver() {
    fastmg->solve_only_directsolver();
}




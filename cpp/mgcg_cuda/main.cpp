#include "common.h"
#include "fastfill.h" 
#include "meshio_tetgen.h"
#include "physdata.h"

using std::shared_ptr;
using std::make_shared;
// using namespace fastmg;

using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;


struct SpMatData
{
    float *data;
    int *indices;
    int *indptr;
    int nrows;
    int ncols;
    int nnz;
};




struct LinearSolver
{
    Eigen::VectorXf x0;
    Eigen::VectorXf x;
    float rtol = 1e-5;
    int maxiter = 100;
    std::vector<float> residuals;
    int niter;
    
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Upper> solver;


    Eigen::VectorXf& LinearSolver::run(SpMat& A, Eigen::VectorXf& b, bool should_setup=true);
};


Eigen::VectorXf& LinearSolver::run(SpMat& A, Eigen::VectorXf& b, bool should_setup)
{
    // Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> A_sp(
    //     A.nrows, A.ncols, A.nnz, A.indptr, A.indices, A.data);

    x = solver.compute(A).solve(b);
    
    if(solver.info()!=Eigen::Success) {
        std::cerr << "Solver failed" << std::endl;
    }
    return x;
}


// struct AmgCuda:LinearSolver
// {
//     FastMG *fastmg;
//     AmgCuda(){fastmg = new FastMG();}

//     void run(SpMatData& A, Field1f& b, bool should_setup=true)
//     {
//         if (should_setup)
//         {
//             fastmg->setup();
//         }

//         fastmg->set_A0(A.data, A.nnz, A.indices, A.nnz, A.indptr, A.nrows + 1, A.nrows, A.ncols, A.nnz);
//         x0.clear()
//         x0.resize(b.size())
//         fastmg->set_data(x0.data(), x0.size(), b.data(), b.size(), rtol, maxiter);
//         fastmg->solve();

//         x.clear();
//         x.resize(b.size());
//         residuals.resize(maxiter);
//         niter = fastmg->get_data(x.data(), residuals.data())
//         niter+=1;
//         residuals.resize(niter);

//         printf("    inner iter: %d", niter);
//         printf("    residual: %.6e->%.6e",residuals[0], residuals[niter-1]);
//     }
// };


Eigen::Matrix<float, 3, 9> make_matrix(float x, float y, float z) {
    Eigen::Matrix<float, 3, 9> mat;
    mat << x, 0, 0, y, 0, 0, z, 0, 0,
        0, x, 0, 0, y, 0, 0, z, 0,
        0, 0, x, 0, 0, y, 0, 0, z;
    return mat;
}



Vec43f compute_gradient(const Mat3f &U, const Eigen::Vector3f &S, const Mat3f &V, const Mat3f &B) {
    float sum_sigma = std::sqrt((S(0) - 1) * (S(0) - 1) + (S(1) - 1) * (S(1) - 1) + (S(2) - 1) * (S(2) - 1));
    // (dcdS00, dcdS11, dcdS22)
    Eigen::Vector3f dcdS = 1.0 / sum_sigma * Eigen::Vector3f(S(0) - 1, S(1) - 1, S(2) - 1);
    // Compute (dFdx)^T
    Eigen::Matrix<float, 3, 9> dFdp1T = make_matrix(B(0, 0), B(0, 1), B(0, 2));
    Eigen::Matrix<float, 3, 9> dFdp2T = make_matrix(B(1, 0), B(1, 1), B(1, 2));
    Eigen::Matrix<float, 3, 9> dFdp3T = make_matrix(B(2, 0), B(2, 1), B(2, 2));
    // Compute (dsdF)
    Eigen::Vector3f dsdF00(U(0, 0) * V(0, 0), U(0, 1) * V(0, 1), U(0, 2) * V(0, 2));
    Eigen::Vector3f dsdF10(U(1, 0) * V(0, 0), U(1, 1) * V(0, 1), U(1, 2) * V(0, 2));
    Eigen::Vector3f dsdF20(U(2, 0) * V(0, 0), U(2, 1) * V(0, 1), U(2, 2) * V(0, 2));
    Eigen::Vector3f dsdF01(U(0, 0) * V(1, 0), U(0, 1) * V(1, 1), U(0, 2) * V(1, 2));
    Eigen::Vector3f dsdF11(U(1, 0) * V(1, 0), U(1, 1) * V(1, 1), U(1, 2) * V(1, 2));
    Eigen::Vector3f dsdF21(U(2, 0) * V(1, 0), U(2, 1) * V(1, 1), U(2, 2) * V(1, 2));
    Eigen::Vector3f dsdF02(U(0, 0) * V(2, 0), U(0, 1) * V(2, 1), U(0, 2) * V(2, 2));
    Eigen::Vector3f dsdF12(U(1, 0) * V(2, 0), U(1, 1) * V(2, 1), U(1, 2) * V(2, 2));
    Eigen::Vector3f dsdF22(U(2, 0) * V(2, 0), U(2, 1) * V(2, 1), U(2, 2) * V(2, 2));

    // Compute (dcdF)
    Eigen::Vector<float, 9> dcdF(  dsdF00.dot(dcdS),
                                dsdF10.dot(dcdS), 
                                dsdF20.dot(dcdS),
                                dsdF01.dot(dcdS), 
                                dsdF11.dot(dcdS), 
                                dsdF21.dot(dcdS),
                                dsdF02.dot(dcdS), 
                                dsdF12.dot(dcdS), 
                                dsdF22.dot(dcdS));

    Eigen::Vector3f g1;
    Eigen::Vector3f g2;
    Eigen::Vector3f g3;
    Eigen::Vector3f g0;
    g1 = dFdp1T * dcdF;
    g2 = dFdp2T * dcdF;
    g3 = dFdp3T * dcdF;
    g0 = -g1 - g2 - g3;

    return Vec43f{g0, g1, g2, g3};
}


// TODO: Implement in cuda kernel
void compute_C_and_gradC_imply(Field3f &pos_mid, Field4i &vert, FieldMat3f &B, Field1f &constraints, Field43f &gradC) {
    for (int t = 0; t < vert.size(); t++) {
        int p0 = vert[t][0];
        int p1 = vert[t][1];
        int p2 = vert[t][2];
        int p3 = vert[t][3];
        Vec3f x0 = pos_mid[p0];
        Vec3f x1 = pos_mid[p1];
        Vec3f x2 = pos_mid[p2];
        Vec3f x3 = pos_mid[p3];
        Mat3f D_s;
        D_s <<
            x1[0] - x0[0], x2[0] - x0[0], x3[0] - x0[0],
            x1[1] - x0[1], x2[1] - x0[1], x3[1] - x0[1],
            x1[2] - x0[2], x2[2] - x0[2], x3[2] - x0[2];
        Mat3f F = D_s * B[t];

        Eigen::JacobiSVD<Mat3f, Eigen::NoQRPreconditioner> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3f U = svd.matrixU();
        Mat3f V = svd.matrixV();
        Eigen::Vector3f S = svd.singularValues();
        constraints[t] = std::sqrt((S[0] - 1) * (S[0] - 1) + (S[1] - 1) * (S[1] - 1) + (S[2] - 1) * (S[2] - 1));
        gradC[t] = compute_gradient(U, S, V, B[t]);
    }
}





// struct FastFillSoftWrapper:FastFillSoft
// {
//     FastFillSoftWrapper() {}
//     std::pair<SpMatData, Eigen::VectorXf> run(Field3f &pos, Field43f &gradC);
// };

// std::pair<SpMatData, Eigen::VectorXf> FastFillSoftWrapper::run(Field3f &pos, Field43f &gradC)
// {
//     SpMatData A;
//     Eigen::VectorXf b;
//     float* p = reinterpret_cast<float*>(pos.data());
//     float* g = reinterpret_cast<float*>(gradC.data());
//     FastFillSoft::run(p, g);
//     return std::make_pair(A,b);
// }





struct SoftBody
{
    LinearSolver* m_linsol; 
    PhysData* m_d;
    // FastFillSoftWrapper* m_ff;
    std::vector<float> residuals;
    SpMat A;
    SoftBody(PhysData* d);
    ~SoftBody();
    void semi_euler(PhysData* m_d);
    void update_vel(PhysData* m_d);
    Field3f&  dlam2dpos(PhysData* m_d, Eigen::VectorXf& dlam);
    void compute_b(PhysData* m_d);
    void compute_C_and_gradC(PhysData* m_d);
    void copy_pos_mid();
    void update_pos(PhysData* m_d, Field3f& pos, Field3f& dpos);
    SpMat fillA(PhysData* m_d);
    void project_arap_oneiter(PhysData* m_d);
    void substep(int maxiter=10);
};


SoftBody::SoftBody(PhysData* d) : m_d(d)
{
    // physics data
    // m_d is already initialized in the initializer list

    // create the fastfill
    // m_ff = new FastFillSoftWrapper;

    // create the linear solver
    m_linsol = new LinearSolver();

};




SoftBody::~SoftBody()
{
    // delete m_ff;
    delete m_linsol;
}



void SoftBody::compute_b(PhysData* m_d) {
    Field1f& alpha_tilde = m_d->alpha_tilde;
    Field1f& lam = m_d->lam;
    Field1f& constraints = m_d->constraints;
    Eigen::VectorXf& b = m_d->b;
    int NCONS = m_d->NCONS;

    for (int i=0; i<NCONS; i++) {
        b[i] = - alpha_tilde.at(i) * lam.at(i) - constraints.at(i);
    }
}


void SoftBody::compute_C_and_gradC(PhysData* m_d) {
    Field3f& pos_mid = m_d->pos_mid;
    Field4i& vert = m_d->vert;
    FieldMat3f& B = m_d->B;
    Field1f& constraints = m_d->constraints;
    Field43f& gradC = m_d->gradC;

    compute_C_and_gradC_imply(pos_mid, vert, B, constraints, gradC);
}



Field3f& SoftBody::dlam2dpos(PhysData* m_d, Eigen::VectorXf& dlam)
{
    Field43f& gradC = m_d->gradC;
    Field4i& vert = m_d->vert;
    Field1f& inv_mass = m_d->inv_mass;
    Field1f& lam = m_d->lam;
    Field3f& dpos = m_d->dpos;

    for(int i=0; i<vert.size(); i++)
    {
        int idx0 = vert[i][0];
        int idx1 = vert[i][1];
        int idx2 = vert[i][2];
        int idx3 = vert[i][3];

        lam[i] += dlam[i];
        dpos.at(idx0) += inv_mass[idx0] * dlam[i] * gradC[i][0];
        dpos.at(idx1) += inv_mass[idx1] * dlam[i] * gradC[i][1];
        dpos.at(idx2) += inv_mass[idx2] * dlam[i] * gradC[i][2];
        dpos.at(idx3) += inv_mass[idx3] * dlam[i] * gradC[i][3];
    }
    return dpos;
}




void SoftBody::update_pos(PhysData* m_d, Field3f& pos, Field3f& dpos)
{
    Field1f& inv_mass = m_d->inv_mass;

    for(int i=0; i<pos.size(); i++)
    {
        if (inv_mass[i] != 0.0)
        {
            pos[i] += m_d->omega * dpos[i];
        }
    }
}



void SoftBody::substep(int maxiter)
{
    semi_euler(m_d);
    std::fill(m_d->lam.begin(), m_d->lam.end(), 0.0);
    for(int i=0; i<maxiter; i++)
    {
        project_arap_oneiter(m_d);
    }
    update_vel(m_d);
}


void SoftBody::copy_pos_mid() {
    Field3f& pos = m_d->pos;
    Field3f& pos_mid = m_d->pos_mid;

    pos_mid = pos;
}


SpMat SoftBody::fillA(PhysData* m_d) {
    

    return A;
}



void SoftBody::project_arap_oneiter(PhysData* m_d) {
    copy_pos_mid();
    compute_C_and_gradC(m_d);
    compute_b(m_d);
    A = fillA(m_d);
    // auto &[A,b]=m_ff->run(m_d->pos, m_d->gradC);
    m_d->dlam = m_linsol->run(A,m_d->b);
    m_d->dpos = dlam2dpos(m_d, m_d->dlam);
    update_pos(m_d, m_d->pos, m_d->dpos);
}


void SoftBody::semi_euler(PhysData *m_d)
{
    Field3f& pos = m_d->pos;
    Field3f& vel = m_d->vel;
    Field3f& old_pos = m_d->old_pos;
    Field1f& inv_mass = m_d->inv_mass;
    Vec3f& gravity = m_d->gravity;
    float h = m_d->delta_t;

    for (int i = 0; i < pos.size(); i++)
    {
        if (inv_mass[i] != 0.0)
        {
            vel[i] += h * gravity;
            old_pos[i] = pos[i];
            pos[i] += h * vel[i];
        }
    }
}

void SoftBody::update_vel(PhysData *m_d)
{
    Field3f& pos = m_d->pos;
    Field3f& vel = m_d->vel;
    Field3f& old_pos = m_d->old_pos;
    Field1f& inv_mass = m_d->inv_mass;
    float h = m_d->delta_t;

    for (int i = 0; i < pos.size(); i++)
    {
        if (inv_mass[i] != 0.0)
        {
            vel[i] = (pos[i] - old_pos[i]) / h;
        }
    }
}


std::pair<Field3f, Field4i> readmesh()
{
    std::string file="D:/Dev/tiPBD/data/model/bunny_small/bunny_small";

    auto [pos,vert,face] = read_tetgen(file);
    // change pos to vector of vec3f
    Field3f pos3f(pos.size()/3);
    for(int i=0; i<pos.size(); i+=3)
    {
        pos3f.at(i/3) = Vec3f(pos[i], pos[i+1], pos[i+2]);
    }
    // change vert to vector of vec4i
    Field4i vert4i(vert.size()/4);
    for(int i=0; i<vert.size(); i+=4)
    {
        vert4i.at(i/4) = Vec4i{vert[i], vert[i+1], vert[i+2], vert[i+3]};
    }
    return std::move(std::make_pair(pos3f, vert4i));
}



// example usage
int main()
{
    auto [pos,vert] = readmesh();

    PhysData d(pos.size(), vert.size(), pos, vert);
    SoftBody sb(&d);

    for(int i=0; i<100; i++)
    {
        sb.substep();
    }

    return 0;
}


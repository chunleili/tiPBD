#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_set>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/SparseExtra"


using std::vector;
using std::array;
using Eigen::Map;
using Eigen::Vector3f;
using Eigen::VectorXf;
using Eigen::Matrix3f;
using SpMat = Eigen::SparseMatrix<float>;
using Triplet = Eigen::Triplet<float>;

// typedefs
using Vec3f = Eigen::Vector3f;
using Vec2i = std::array<int, 2>; // using Vec2i = Eigen::Vector2i;
using Vec3i = std::array<int, 3>; // using Vec3i = Eigen::Vector3i;
using Vec4i = std::array<int, 4>; // using Vec4i = Eigen::Vector4i;
using Vec43f = std::array<Vec3f, 4>; // using Vec4i = Eigen::Vector4i;
using Field1f = vector<float>;
using Field3f = vector<Vec3f>;
using Field3i = vector<Vec3i>;
using Field4i = vector<Vec4i>;
using Field2i = vector<Vec2i>;
using Field1i = vector<int>;
using Field43f = vector<Vec43f>;
using FieldXi = vector<vector<int>>;
using FieldMat3f = vector<Eigen::Matrix3f>;
using Field43f = vector<Vec43f>;




// function declarations
Eigen::Matrix<float, 3, 9> make_matrix(float x, float y, float z);
Vec43f compute_gradient(const Eigen::Matrix3f &U, const Eigen::Vector3f &S, const Eigen::Matrix3f &V, const Eigen::Matrix3f &B);
void compute_C_and_gradC_imply(Field3f &pos_mid, Field4i &vert, FieldMat3f &B, Field1f &constraints, Field43f &gradC);



class SolveConstraintsMgxpbd {
public:
    SolveConstraintsMgxpbd() {}
    
    size_t NV=0;
    size_t NCONS=0;
    Field3f pos;
    Field1f alpha_tilde;
    Field1f rest_len;
    Field4i vert;
    Field1f inv_mass;
    Field1f constraints;
    FieldMat3f B;
    Field3f pos_mid;
    Field1f lambda;
    Field1f dlambda;
    Field3f dpos;
    float residual=1e6;

    Field43f gradC;
    Field1f b;

    void resize_fields(size_t NV, size_t NCONS)
    {
        this->NV = NV;
        this->NCONS = NCONS;
        pos.resize(NV, Vec3f(0.0, 0.0, 0.0));
        vert.resize(NCONS, Vec4i{0, 0, 0, 0});
        inv_mass.resize(NV, 0.0);
        rest_len.resize(NCONS, 0.0);
        lambda.resize(NCONS, 0.0);
        constraints.resize(NCONS, 0.0);
        pos_mid.resize(NV, Vec3f(0.0, 0.0, 0.0));
        dpos.resize(NV, Vec3f(0.0, 0.0, 0.0));
        gradC.resize(NCONS, Vec43f{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});
        alpha_tilde.resize(NCONS, 0.0);
        dlambda.resize(NCONS, 0.0);
        B.resize(NCONS, Matrix3f::Zero());
        b.resize(NCONS, 0.0);
    }

    void solve_constraints_mgxpbd() {
        std::copy(pos.begin(), pos.end(), pos_mid.begin());
        compute_C_and_gradC();
        compute_b();
    }


    void compute_C_and_gradC() {
        // TODO: Implement in cuda kernel
        compute_C_and_gradC_imply(pos_mid, vert, B, constraints, gradC);
    }

    void compute_b() {
        for (int i=0; i<NCONS; i++) {
            b[i] = - alpha_tilde[i] * lambda[i] - constraints[i];
        }
    }

/* -------------------------------------------------------------------------- */
/*                               implementations                              */
/* -------------------------------------------------------------------------- */
    Eigen::Matrix<float, 3, 9> make_matrix(float x, float y, float z) {
        Eigen::Matrix<float, 3, 9> mat;
        mat << x, 0, 0, y, 0, 0, z, 0, 0,
            0, x, 0, 0, y, 0, 0, z, 0,
            0, 0, x, 0, 0, y, 0, 0, z;
        return mat;
    }



    Vec43f compute_gradient(const Eigen::Matrix3f &U, const Eigen::Vector3f &S, const Eigen::Matrix3f &V, const Eigen::Matrix3f &B) {
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
            Eigen::Matrix3f D_s;
            D_s <<
                x1[0] - x0[0], x2[0] - x0[0], x3[0] - x0[0],
                x1[1] - x0[1], x2[1] - x0[1], x3[1] - x0[1],
                x1[2] - x0[2], x2[2] - x0[2], x3[2] - x0[2];
            Eigen::Matrix3f F = D_s * B[t];

            Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();
            Eigen::Vector3f S = svd.singularValues();
            constraints[t] = std::sqrt((S[0] - 1) * (S[0] - 1) + (S[1] - 1) * (S[1] - 1) + (S[2] - 1) * (S[2] - 1));
            gradC[t] = compute_gradient(U, S, V, B[t]);
        }
    }

}; // end of SolveConstraintsMgxpbd class





/* -------------------------------------------------------------------------- */
/*                         For interaction with python                        */
/* -------------------------------------------------------------------------- */
static SolveConstraintsMgxpbd *SolveConstraintsMgxpbd_instance = nullptr;

#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT void fastmg_solve_constraints_mgxpbd_new() {
        if (!SolveConstraintsMgxpbd_instance)
        SolveConstraintsMgxpbd_instance = new SolveConstraintsMgxpbd{};
}

extern "C" DLLEXPORT float fastmg_solve_constraints_mgxpbd_run(
    size_t NV_in,
    size_t NCONS_in,
    const float* pos_in, 
    const float* alphat_tilde_in,
    const float* rest_len_in,    
    const int* vert_in,
    const float* inv_mass_in,        //NOT USED NOW
    const float  delta_t_in,     //only for alpha_tilde
    const float* B, //NOT USED NOW
    const float* lambda_in,
    float* dlambda_out,
    float* dpos_out,
    float* constraints_out, //TODO:internal, for now
    float* gradC_out,       //TODO:internal, for now
    float* b_out           //TODO:internal, for now
) {
    SolveConstraintsMgxpbd* p = SolveConstraintsMgxpbd_instance; // alias

    if (NV_in != p->NV || NCONS_in != p->NCONS) {
        p->NV = NV_in;
        p->NCONS = NCONS_in;
        p->resize_fields(NV_in, NCONS_in);
    }
    
    // copy input data FIXME: a better is to pass the pointer directly
    std::copy(pos_in, pos_in + NV_in*3, p->pos[0].data());
    std::copy(alphat_tilde_in, alphat_tilde_in + NCONS_in, p->alpha_tilde.data());
    std::copy(rest_len_in, rest_len_in + NCONS_in, p->rest_len.data());
    std::copy(vert_in, vert_in + NCONS_in*4, p->vert[0].data());
    std::copy(inv_mass_in, inv_mass_in + NV_in, p->inv_mass.data());
    std::copy(B, B + NCONS_in*9, p->B[0].data());
    std::copy(lambda_in, lambda_in + NCONS_in, p->lambda.data());
    


    p->solve_constraints_mgxpbd();

    // copy output data FIXME: a better is to pass the pointer
    std::copy(p->dlambda.data(), p->dlambda.data() + NCONS_in, dlambda_out);
    std::copy(p->dpos[0].data(), p->dpos[0].data() + NV_in*3, dpos_out);
    std::copy(p->constraints.data(), p->constraints.data() + NCONS_in, constraints_out);
    std::copy(p->gradC[0][0].data(), p->gradC[0][0].data() + NCONS_in*12, gradC_out);
    std::copy(p->b.data(), p->b.data() + NCONS_in, b_out);

    // printf("residual: %f\n", p->residual);
    float residual = p->residual;
    return residual;
}
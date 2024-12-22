#pragma once
#include "common.h"
#include "solve_soft.h"


// function declarations(static)
Eigen::Matrix<float, 3, 9> make_matrix(float x, float y, float z);
Vec43f compute_gradient(const Mat3f &U, const Eigen::Vector3f &S, const Mat3f &V, const Mat3f &B);

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
        if (constraints[t] > 1e-6) {
            gradC[t] = compute_gradient(U, S, V, B[t]);
        }
    }
}

void SolveSoft::compute_C_and_gradC(){
    compute_C_and_gradC_imply(this->pos_mid, this->vert, this->B, this->constraints, this->gradC);
}

void SolveSoft::solve() {
    std::copy(pos.begin(), pos.end(), pos_mid.begin());
    compute_C_and_gradC();
    compute_b();
}   

void SolveSoft::compute_b() {
    for (int i=0; i<NCONS; i++) {
        b[i] = - alpha_tilde[i] * lam[i] - constraints[i];
    }
}


void SolveSoft::resize_fields(size_t NV, size_t NCONS)
{
    this->NV = NV;
    this->NCONS = NCONS;
    pos.resize(NV, Vec3f(0.0, 0.0, 0.0));
    vert.resize(NCONS, Vec4i{0, 0, 0, 0});
    inv_mass.resize(NV, 0.0);
    rest_len.resize(NCONS, 0.0);
    lam.resize(NCONS, 0.0);
    constraints.resize(NCONS, 0.0);
    pos_mid.resize(NV, Vec3f(0.0, 0.0, 0.0));
    dpos.resize(NV, Vec3f(0.0, 0.0, 0.0));
    gradC.resize(NCONS, Vec43f{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});
    alpha_tilde.resize(NCONS, 0.0);
    dlambda.resize(NCONS, 0.0);
    B.resize(NCONS, Mat3f::Zero());
    b.resize(NCONS, 0.0);
}


// /* -------------------------------------------------------------------------- */
// /*                         For interaction with python                        */
// /* -------------------------------------------------------------------------- */
// static SolveSoft *solveSoft = nullptr;

// #if _WIN32
// #define DLLEXPORT __declspec(dllexport)
// #else
// #define DLLEXPORT
// #endif

// extern "C" DLLEXPORT void solveSoft_new() {
//         if (!solveSoft)
//         solveSoft = new SolveSoft{};
// }


// extern "C" DLLEXPORT void solveSoft_set_data(
//     size_t NV_in,
//     size_t NCONS_in,
//     const float* pos_in, 
//     const float* alphat_tilde_in,
//     const float* rest_len_in,    
//     const int* vert_in,
//     const float* inv_mass_in,        //NOT USED NOW
//     const float  delta_t_in,     //only for alpha_tilde
//     const float* B, //NOT USED NOW
//     const float* lambda_in
// ) {
//     // SolveSoft* p = solveSoft; // alias

//     if (NV_in != solveSoft->NV || NCONS_in != solveSoft->NCONS) {
//         solveSoft->NV = NV_in;
//         solveSoft->NCONS = NCONS_in;
//         solveSoft->resize_fields(NV_in, NCONS_in);
//     }
    
//     // copy input data FIXME: a better is to pass the pointer directly
//     std::copy(pos_in, pos_in + NV_in*3, solveSoft->pos[0].data());
//     std::copy(alphat_tilde_in, alphat_tilde_in + NCONS_in, solveSoft->alpha_tilde.data());
//     std::copy(rest_len_in, rest_len_in + NCONS_in, solveSoft->rest_len.data());
//     std::copy(vert_in, vert_in + NCONS_in*4, solveSoft->vert[0].data());
//     std::copy(inv_mass_in, inv_mass_in + NV_in, solveSoft->inv_mass.data());
//     std::copy(B, B + NCONS_in*9, solveSoft->B[0].data()); //Eigen is col major
//     for (int i=0; i<NCONS_in; i++) {
//         // copy from col major to row major
//         solveSoft->B[i].transposeInPlace();
//     }

//     std::copy(lambda_in, lambda_in + NCONS_in, solveSoft->lam.data());
// }


// extern "C" DLLEXPORT float solveSoft_get_data(
//     float* dlambda_out,
//     float* dpos_out,
//     float* constraints_out, //TODO:internal, for now
//     float* gradC_out,       //TODO:internal, for now
//     float* b_out           //TODO:internal, for now
// ) {
//     int NV_in = solveSoft->NV;
//     int NCONS_in = solveSoft->NCONS;
//     // copy output data FIXME: a better is to pass the pointer
    
//     std::copy(solveSoft->dlambda.data(), solveSoft->dlambda.data() + NCONS_in, dlambda_out);
//     std::copy(solveSoft->dpos[0].data(), solveSoft->dpos[0].data() + NV_in*3, dpos_out);
//     std::copy(solveSoft->constraints.data(), solveSoft->constraints.data() + NCONS_in, constraints_out);
//     std::copy(solveSoft->gradC[0][0].data(), solveSoft->gradC[0][0].data() + NCONS_in*12, gradC_out);
//     std::copy(solveSoft->b.data(), solveSoft->b.data() + NCONS_in, b_out);

//     // printf("residual: %f\n", solveSoft->residual);
//     float residual = solveSoft->residual;
//     return residual;
// }


// extern "C" DLLEXPORT float solveSoft_run(
//     const float* pos_in
// ) {
//     int NV = solveSoft->NV;
//     std::copy(pos_in, pos_in + NV*3, solveSoft->pos[0].data());

//     solveSoft->solve_constraints_mgxpbd();

//     float residual = solveSoft->residual;
//     return residual;
// }
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
using Field1f = Eigen::VectorXf;
using Field3f = vector<Vec3f>;
using Field3i = vector<Vec3i>;
using Field4i = vector<Vec4i>;
using Field2i = vector<Vec2i>;
using Field1i = vector<int>;
using Field43f = vector<array<Vec3f, 4>>;
using FieldXi = vector<vector<int>>;
using FieldMat3f = vector<Eigen::Matrix3f>;
using Field43f = vector<array<Vec3f, 4>>;

// function declarations
Eigen::Matrix<float, 3, 9> make_matrix(float x, float y, float z);
Vec43f compute_gradient(const Eigen::Matrix3f &U, const Eigen::Vector3f &S, const Eigen::Matrix3f &V, const Eigen::Matrix3f &B);
void compute_C_and_gradC_imply(Field3f &pos_mid, Field4i &tet_indices, FieldMat3f &B, Field1f &constraints, Field43f &gradC);



class SolveConstraintsMgxpbd {
public:
    SolveConstraintsMgxpbd(size_t nv, size_t ncons) : NV(nv), NCONS(ncons) {
        resize_fields();
    }
    Field3f pos;
    Field4i tet_indices;
    Field1f rest_len;
    Field1f inv_mass;
    Field1f lagrangian;
    Field1f dlambda;
    Field1f constraints;
    FieldMat3f B;
    Field3f pos_mid;
    Field3f dpos;
    Field43f gradC;
    Field1f alpha_tilde;
    Eigen::VectorXf b;
    size_t NV;
    size_t NCONS;

    


    void resize_fields()
    {
        pos.resize(NV, Vec3f(0.0, 0.0, 0.0));
        tet_indices.resize(NCONS, Vec4i{0, 0, 0, 0});
        rest_len.resize(NCONS, 0.0);
        inv_mass.resize(NV, 0.0);
        lagrangian.resize(NCONS, 0.0);
        constraints.resize(NCONS, 0.0);
        pos_mid.resize(NV, Vec3f(0.0, 0.0, 0.0));
        dpos.resize(NV, Vec3f(0.0, 0.0, 0.0));
        gradC.resize(NCONS, array<Vec3f, 4>{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});
        alpha_tilde.resize(NCONS, 0.0);
        dlambda.resize(NCONS, 0.0);
        B.resize(NCONS, Matrix3f::Zero());
        b.resize(NV*3, 0.0);
    }

    void solve_constraints_mgxpbd() {
        std::copy(pos.begin(), pos.end(), pos_mid.begin());
        compute_C_and_gradC();
        compute_b();
    }


    void compute_C_and_gradC() {
        // TODO: Implement in cuda kernel
        compute_C_and_gradC_imply(pos_mid, tet_indices, B, constraints, gradC);
    }

    void compute_b() {
        VectorXf temp = alpha_tilde * lagrangian;
        b = -constraints - temp;
    }

};


Eigen::Matrix<float, 3, 9> make_matrix(float x, float y, float z) {
    Eigen::Matrix<float, 3, 9> mat;
    mat << x, 0, 0, y, 0, 0, z, 0, 0,
           0, x, 0, 0, y, 0, 0, z, 0,
           0, 0, x, 0, 0, y, 0, 0, z;
    return mat;
}



Vec43f compute_gradient(const Eigen::Matrix3f &U, const Eigen::Vector3f &S, const Eigen::Matrix3f &V, const Eigen::Matrix3f &B) {
    float sum_sigma = std::sqrt((S(0, 0) - 1) * (S(0, 0) - 1) + (S(1, 1) - 1) * (S(1, 1) - 1) + (S(2, 2) - 1) * (S(2, 2) - 1));
    // (dcdS00, dcdS11, dcdS22)
    Eigen::Vector3f dcdS = 1.0 / sum_sigma * Eigen::Vector3f(S(0, 0) - 1, S(1, 1) - 1, S(2, 2) - 1);
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


void compute_C_and_gradC_imply(Field3f &pos_mid, Field4i &tet_indices, FieldMat3f &B, Field1f &constraints, Field43f &gradC) {
    for (int t = 0; t < tet_indices.size(); t++) {
        int p0 = tet_indices[t][0];
        int p1 = tet_indices[t][1];
        int p2 = tet_indices[t][2];
        int p3 = tet_indices[t][3];
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

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(F);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        Eigen::Vector3f S = svd.singularValues();
        constraints[t] = std::sqrt((S[0] - 1) * (S[0] - 1) + (S[1] - 1) * (S[1] - 1) + (S[2] - 1) * (S[2] - 1));
        gradC[t] = compute_gradient(U, S, V, B[t]);
    }
}

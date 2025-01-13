#include "common.h"
#include "fastfill.h" 
#include "meshio_tetgen.h"
#include "meshio_obj.h"
#include "physdata.h"
#include "linear_solver.h"
#include "amgcl_solver.h"
#include "eigen_solver.h"


using namespace fastmg;


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
        if (constraints[t]>1e-6) //CAUTION! When the constraint is too small, there is no deformation at all, the gradient will be in any direction! Sigma=(1,1,1) There will be singularity issue!
            gradC[t] = compute_gradient(U, S, V, B[t]);
    }
}





struct FastFillSoftWrapper:FastFillSoft
{
    SpMatData *m_hostA;
    FastFillSoftWrapper()  = delete;
    FastFillSoftWrapper(PhysData *d, SpMatData*A);
    SpMatData* run(Field3f &pos, Field43f &gradC);
};


FastFillSoftWrapper::FastFillSoftWrapper(PhysData *d, SpMatData*A) : FastFillSoft(d), m_hostA(A) {
    this->d_alpha_tilde.assign(d->alpha_tilde.data(), d->NCONS);
    this->d_inv_mass.assign(d->inv_mass.data(), d->NV);
};



SpMatData* FastFillSoftWrapper::run(Field3f &pos, Field43f &gradC)
{
    float* p = pos[0].data();
    float* g = gradC[0][0].data();
    FastFillSoft::run(p, g);
    FastFillSoft::fetch_A(*m_hostA);
    FastFillSoft::get_ii(m_hostA->ii);
    return m_hostA;
}





struct SoftBody
{
    LinearSolver* m_linsol; 
    PhysData* m_d;
    FastFillSoftWrapper* m_ff;
    std::vector<float> residuals;
    SpMatData *m_hostA;
    SoftBody(PhysData* d);
    ~SoftBody();
    void semi_euler(PhysData* m_d);
    void update_vel(PhysData* m_d);
    Field3f&  dlam2dpos(PhysData* m_d, Field1f& dlam);
    void compute_b(PhysData* m_d);
    void compute_C_and_gradC(PhysData* m_d);
    void copy_pos_mid();
    void update_pos(PhysData* m_d, Field3f& pos, Field3f& dpos);
    void fillA(PhysData* m_d);
    void project_arap_oneiter(PhysData* m_d);
    void substep(int maxiter=10);
};


SoftBody::SoftBody(PhysData* d) : m_d(d)
{
    // physics data
    // m_d is already initialized in the initializer list

    // create the system matrix
    m_hostA = new SpMatData();
    
    // create the fastfill
    m_ff = new FastFillSoftWrapper(m_d, m_hostA);

    // create the linear solver
    m_linsol = new AmgclSolver();

};




SoftBody::~SoftBody()
{
    delete m_linsol;
    delete m_ff;
    delete m_hostA;
}



void SoftBody::compute_b(PhysData* m_d) {
    Field1f& alpha_tilde = m_d->alpha_tilde;
    Field1f& lam = m_d->lam;
    Field1f& constraints = m_d->constraints;
    Field1f& b = m_d->b;
    float& dual = m_d->dual;
    int NCONS = m_d->NCONS;

    dual = 0.0;
    for (int i=0; i<NCONS; i++) {
        b[i] = - alpha_tilde.at(i) * lam.at(i) - constraints.at(i);
        dual += b[i] * b[i];
    }
    dual = std::sqrt(dual);
}


void SoftBody::compute_C_and_gradC(PhysData* m_d) {
    Field3f& pos_mid = m_d->pos_mid;
    Field4i& vert = m_d->vert;
    FieldMat3f& B = m_d->B;
    Field1f& constraints = m_d->constraints;
    Field43f& gradC = m_d->gradC;

    compute_C_and_gradC_imply(pos_mid, vert, B, constraints, gradC);
}



Field3f& SoftBody::dlam2dpos(PhysData* m_d, Field1f& dlam)
{
    Field43f& gradC = m_d->gradC;
    Field4i& vert = m_d->vert;
    Field1f& inv_mass = m_d->inv_mass;
    Field1f& lam = m_d->lam;
    Field3f& dpos = m_d->dpos;

    dpos.clear();
    dpos.resize(m_d->NV, Vec3f(0.0, 0.0, 0.0));

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
    for(m_d->ite=0; m_d->ite<maxiter; m_d->ite++)
    {
        printf("iter = %d\n", m_d->ite);
        project_arap_oneiter(m_d);
    }
    update_vel(m_d);
}


void SoftBody::copy_pos_mid() {
    Field3f& pos = m_d->pos;
    Field3f& pos_mid = m_d->pos_mid;

    // pos_mid = pos;
    std::copy(pos.begin(), pos.end(), pos_mid.begin());
}



void SoftBody::fillA(PhysData* m_d) {
    m_hostA = m_ff->run(m_d->pos_mid, m_d->gradC);
    return ;
}



void SoftBody::project_arap_oneiter(PhysData* m_d) {
    copy_pos_mid();
    compute_C_and_gradC(m_d);
    compute_b(m_d); //also compute dual
    fillA(m_d);
    m_d->dlam = m_linsol->solve(m_hostA,m_d->b);
    m_d->dpos = dlam2dpos(m_d, m_d->dlam);
    update_pos(m_d, m_d->pos, m_d->dpos);
    printf("    dual = %f\n", m_d->dual);
}


void SoftBody::semi_euler(PhysData *m_d)
{
    Field3f& pos = m_d->pos;
    Field3f& vel = m_d->vel;
    Field3f& old_pos = m_d->old_pos;
    Field3f& predict_pos = m_d->predict_pos;
    Field3f& force = m_d->force;
    Field1f& inv_mass = m_d->inv_mass;
    Vec3f& gravity = m_d->gravity;
    float delta_t = m_d->delta_t;
    float damping_coeff = m_d->damping_coeff;

    for (int i = 0; i < pos.size(); i++)
    {
        if (inv_mass[i] != 0.0)
        {
            old_pos[i] = pos[i];
            vel[i] += damping_coeff* delta_t * (gravity+force[i]);
            pos[i] += delta_t * vel[i];
            predict_pos[i] = pos[i];
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


std::tuple<Field3f, Field4i, Field3i> readmesh(std::string file)
{
    auto [pos,vert,face] = read_tetgen(file);
    // reshape pos to vector of Vec3f
    Field3f pos3f(pos.size()/3);
    for(int i=0; i<pos.size(); i+=3)
    {
        pos3f.at(i/3) = Vec3f(pos[i], pos[i+1], pos[i+2]);
    }
    // reshape vert to vector of Vec4i
    Field4i vert4i(vert.size()/4);
    for(int i=0; i<vert.size(); i+=4)
    {
        vert4i.at(i/4) = Vec4i{vert[i], vert[i+1], vert[i+2], vert[i+3]};
    }
    // reshape face to vector of Vec3i
    Field3i face3i(face.size()/3);
    for(int i=0; i<face.size(); i+=3)
    {
        face3i.at(i/3) = Vec3i{face[i], face[i+1], face[i+2]};
    }
    return std::move(std::make_tuple(pos3f, vert4i, face3i));
}


void reinit(Field3f& pos)
{
    for(int i=0; i<pos.size(); i++)
    {
        pos[i]*= 1.5; //enlarge
    }
}



void write_mesh(int frame_num, Field3f &pos, Field3i &face)
{
    // tic();
    std::filesystem::path p(__FILE__);
    std::filesystem::path prj_path = p.parent_path().parent_path().parent_path();
    auto proj_dir_path = prj_path.string();
    std::string mesh_dir = proj_dir_path + "/result/latest/mesh/";

    std::string out_mesh_name = mesh_dir + std::to_string(frame_num) + ".obj";
    write_obj(out_mesh_name, pos, face);
    // toc("output mesh");
}

// example usage
int run_softbody(std::string mesh="D:/Dev/tiPBD/data/model/bunny_small/bunny_small", int endframe=100, float mu=1e6, float delta_t=3e-3, bool verbose=false)
{
    printf("Reading mesh...\n");
    auto [pos,vert, face] = readmesh(mesh);
    // all physical data in in this class
    printf("Initializing softbody...\n");
    PhysData d(pos, vert, mu, delta_t);
    SoftBody sb(&d);
    sb.m_linsol->verbose = verbose; // print solver info and time
    //enlarge the model to see deformation, in your case, you can skip this step
    reinit(d.pos);
    printf("Start run softbody...\n");
    for(d.frame=1; d.frame<endframe; d.frame++)
    {
        printf("\n-------------\nframe = %d\n", d.frame);
        sb.substep();
        write_mesh(d.frame, d.pos, face);
    }

    return 0;
}


int main(int argc, char *argv[]) {
    fprintf(stdout, "Usage: %s <output_file> <endframe> <mu> <delta_t> <verbose>\n", argv[0]);
    fprintf(stdout, "Example: %s D:/Dev/tiPBD/data/model/bunny_small/bunny_small 100 1e6 3e-3 0\n", argv[0]);
    fprintf(stdout, "Example: %s D:/Dev/tiPBD/data/model/bunny85w/bunny85w 2 1e9 3e-3 1\n", argv[0]);
    std::string mesh = argv[1];
    int endframe = std::stoi(argv[2]);
    float mu = std::stof(argv[3]);
    float delta_t = std::stof(argv[4]);
    bool verbose = std::stoi(argv[5]);
    run_softbody(mesh, endframe, mu, delta_t, verbose);
    return 0;
}

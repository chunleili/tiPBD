#pragma once
#include "physdata.h"

PhysData::PhysData(Field3f &pos,Field4i &vert, float mu, float delta_t)
{
    this->NV = pos.size();
    this->NCONS = vert.size();
    resize(this->NV,this->NCONS);
    this->pos = pos;
    this->vert = vert;
    this->init_physics(pos,vert,mu,delta_t);
    std::fill(inv_mass.begin(), inv_mass.end(), 1.0);
    this->gravity = Vec3f(0.0, -9.8, 0.0);
};


void PhysData::init_physics(Field3f &pos, Field4i &vert, float arap_mu, float delta_t)
{
    // init B and rest_volume
    rest_volume.resize(vert.size());
    B.resize(vert.size());
    for(int i=0; i<vert.size(); i++)
    {
        int ia = vert[i][0];
        int ib = vert[i][1];
        int ic = vert[i][2];
        int id = vert[i][3];

        Vec3f p0 = pos[ia];
        Vec3f p1 = pos[ib];
        Vec3f p2 = pos[ic];
        Vec3f p3 = pos[id];

        Mat3f D_m;

        D_m.col(0) = p1 - p0;
        D_m.col(1) = p2 - p0;
        D_m.col(2) = p3 - p0;

        this->B[i] = D_m.inverse();
        this->rest_volume[i] = 1.0 / 6.0 * std::abs(D_m.determinant());
    }

    // init delta_t
    this->delta_t = delta_t;

    // init alpha_tilde
    float inv_mu = 1.0 / arap_mu;
    float inv_h2 = 1.0 / (delta_t * delta_t);
    if(NCONS==0)
        throw std::runtime_error("NCONS==0!");
    this->alpha_tilde.resize(NCONS);
    for (int i = 0; i < alpha_tilde.size(); i++)
    {
        if (rest_volume[i]<=0.0)
            throw std::runtime_error("rest_volume <= 0.0, please check the input tet mesh  or init the rest_volume(by init_B) first!");
        this->alpha_tilde[i] = inv_h2 * inv_mu * 1.0/rest_volume[i];
    }
}


void PhysData::resize(size_t NV, size_t NCONS)
{
    this->NV = NV;
    this->NCONS = NCONS;
    pos.resize(NV);
    vert.resize(NCONS);
    inv_mass.resize(NV);
    rest_len.resize(NCONS);
    lam.resize(NCONS);
    constraints.resize(NCONS);
    pos_mid.resize(NV);
    old_pos.resize(NV);
    dpos.resize(NV);
    alpha_tilde.resize(NCONS);
    gradC.resize(NCONS, Vec43f{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});
    B.resize(NCONS, Mat3f::Zero());
    dlam.resize(NCONS);
    b.resize(NCONS);
    vel.resize(NV);
    force.resize(NV, Vec3f(0.0, 0.0, 0.0));
    predict_pos.resize(NV, Vec3f(0.0, 0.0, 0.0));
}




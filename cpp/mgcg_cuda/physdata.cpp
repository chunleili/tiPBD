#pragma once
#include "physdata.h"

PhysData::PhysData(size_t NV, size_t NCONS, Field3f &pos,Field4i &vert)
{
    // resize(NV,NCONS);
    // this->pos.swap(pos);
    // this->vert = vert;
    // this->init_B(pos,vert);
    // this->delta_t = 1e-3;
    // inv_mass.fill(1.0);
    // alpha_tilde.fill(1e-8/(this->delta_t * this->delta_t));
    // this->gravity = Vec3f(0.0, -9.8, 0.0);
};


void PhysData::init_B(Field3f &pos, Field4i &vert)
{
    // for(int i=0; i<vert.size(); i++)
    // {
    //     int ia = vert(i,0);
    //     int ib = vert(i,1);
    //     int ic = vert(i,2);
    //     int id = vert(i,3);

    //     Vec3f p0 = pos.row(ia);
    //     Vec3f p1 = pos.row(ib);
    //     Vec3f p2 = pos.row(ic);
    //     Vec3f p3 = pos.row(id);

    //     Mat3f D_m;

    //     D_m.col(0) = p1 - p0;
    //     D_m.col(1) = p2 - p0;
    //     D_m.col(2) = p3 - p0;

    //     B[i] = D_m.inverse();
    // }
}


void PhysData::resize(size_t NV, size_t NCONS)
{
    // this->NV = NV;
    // this->NCONS = NCONS;
    // pos.resize(NV);
    // vert.resize(NCONS);
    // inv_mass.resize(NV);
    // rest_len.resize(NCONS);
    // lam.resize(NCONS);
    // constraints.resize(NCONS);
    // pos_mid.resize(NV);
    // dpos.resize(NV);
    // alpha_tilde.resize(NCONS);
    // gradC.resize(NCONS, Vec43f{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});
    // B.resize(NCONS, Mat3f::Zero());
    // dlam.resize(NCONS);
    // b.resize(NCONS);
}




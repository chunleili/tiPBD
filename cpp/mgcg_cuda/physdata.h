#pragma once
#include "common.h"

struct PhysData
{
    size_t NV=0;
    size_t NCONS=0;
    Field3f pos;
    Field3f vel;
    Field1f alpha_tilde;
    Field1f rest_len;
    Field4i vert;
    Field1f inv_mass;
    Field1f constraints;
    FieldMat3f B;
    Field3f pos_mid;
    Field1f lam;
    Field3f dpos;
    Field3f old_pos;
    float delta_t;
    float omega;
    Field43f gradC;
    Vec3f gravity;
    Field1f dlam;
    Field1f b;

    PhysData();
    PhysData(size_t NV, size_t NCONS, Field3f &pos, Field4i &vert);
    ~PhysData() = default;

    void resize(size_t NV, size_t NCONS);
    void init_B(Field3f &pos, Field4i &vert);
    void set_invmass(Field1f& inv_mass){this->inv_mass = inv_mass;};
    void set_alpha_tilde(Field1f& alpha_tilde){this->alpha_tilde = alpha_tilde;};
    void set_dt(float delta_t){this->delta_t = delta_t;};
};




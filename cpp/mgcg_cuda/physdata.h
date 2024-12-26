#pragma once
#include "common.h"

/// @brief Physics data structure
/// @details This structure contains all the necessary data for the physics simulation
/// 
struct PhysData
{
    size_t NV=0;
    size_t NCONS=0; //number of constraints
    Field3f pos;
    Field3f vel;
    Field1f alpha_tilde; // 1.0/stiffness/dt/dt
    Field1f rest_len;
    Field1f rest_volume;
    Field4i vert;       // tetrahedra(p0,p1,p2,p3), size: NCONS x 4
    Field1f inv_mass;
    Field1f constraints;
    FieldMat3f B;       // restmatrix for each tetrahedra, size: NCONS x 3 x 3
    Field3f pos_mid;    
    Field1f lam;        // lagrange multiplier, size: NCONS
    Field3f dpos;       // change in position
    Field3f old_pos;    // last substep position
    float delta_t;      // time step
    float omega=0.25;   // relaxation parameter for dpos
    float mu;           //second lame parameter for arap
    Field43f gradC;     // gradient of the constraint, size: NCONS x 4 x 3
    Vec3f gravity;
    Field1f dlam;
    Field1f b;          // right hand side of the linear system

    PhysData();
    PhysData::PhysData(Field3f &pos,Field4i &vert, float mu=1e6, float delta_t=3e-3);
    ~PhysData() = default;

    void resize(size_t NV, size_t NCONS);
    void init_physics(Field3f &pos, Field4i &vert, float arap_mu, float delta_t);
    void set_invmass(Field1f& inv_mass){this->inv_mass = inv_mass;};
    void set_alpha_tilde(Field1f& alpha_tilde){this->alpha_tilde = alpha_tilde;};
    void set_dt(float delta_t){this->delta_t = delta_t;};
};




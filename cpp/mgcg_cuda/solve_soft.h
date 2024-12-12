#pragma once
#include "common.h"


class SolveSoft {
public:
    SolveSoft() {}
    
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
    float delta_t=1e-3;

    Field43f gradC;
    Field1f b;

    void resize_fields(size_t NV, size_t NCONS);
    void solve();
    
private:
    void compute_C_and_gradC();
    void compute_b();

}; // end of SolveSoft class

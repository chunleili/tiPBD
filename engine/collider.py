import taichi as ti
def add_surface_collider(    point,
                             normal,
                             surface_type="sticky",
                             friction=0.0):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / ti.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        if surface_type == "sticky" and friction != 0:
            raise ValueError('friction must be 0 on sticky surface.')

        dim  = 3

        @ti.kernel
        def collide(vel: ti.template()):
            for I in ti.grouped(vel):
                n = ti.Vector(normal)
                if vel.dot(n) < 0:
                    if ti.static(surface_type == "sticky"):
                        vel[I] = ti.Vector.zero(ti.f32, dim)
                    else:
                        v = vel[I]
                        normal_component = n.dot(v)

                        if ti.static(surface_type == "slip"):
                            # Project out all normal component
                            v = v - n * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - n * min(normal_component, 0)

                        if normal_component < 0 and v.norm() > 1e-30:
                            # Apply friction here
                            v = v.normalized() * max(
                                0,
                                v.norm() + normal_component * friction)

                        vel[I] = v


from engine.util import pos_to_grid_idx
@ti.func
def collision_response(pos, vel, sdf):
    sdf_epsilon = 1e-4
    grid_idx = pos_to_grid_idx(pos)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]

    if sdf_val < sdf_epsilon:
        print("collision")
        pos -= sdf_val * normal
        if vel * normal < 0:
            normal_component = normal.dot(vel)
            vel -=  normal * min(normal_component, 0)

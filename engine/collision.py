import json
import numpy as np
import taichi as ti

# 边界条件类型，从collider json获取
class BCType:
    STICKY = 0  # 粘性碰撞， 即速度为0
    REFLECT = 1  # 反射碰撞， 即法向速度反向

def add_colliders(json_path):
    """ Add colliders from json configs
        json_path: path to json file
        return: list of colliders
    """
    colliders = []
    with open(json_path, 'r') as f:
        data = json.load(f)
        for collider in data['colliders']:
            type = collider['type']
            pos = collider['pos']
            size = collider['size']
            visible = collider.get('visible', True)
            bc_type_str = collider.get('bc_type', 'sticky').lower()  # 默认为sticky
            bc_type = BCType.STICKY if bc_type_str == 'sticky' else BCType.REFLECT
            
            if type == "sphere":
                colliders.append(SphereCollider(pos, size, visible=visible, bc_type=bc_type))
            elif type == "cylinder":
                colliders.append(CylinderCollider(pos, size, visible=visible, bc_type=bc_type))
            else:
                raise ValueError(f"Unknown collider type: {type}")
    return colliders


class Collider:
    def __init__(self, type="sphere", pos=[0,0,0], size=0.1, visible=True, bc_type=BCType.STICKY):
        self.type = type
        self.pos = pos
        self.size = size
        self.visible = visible
        self.bc_type = bc_type


class SphereCollider(Collider):
    def __init__(self, pos=[0,0,0], size=0.1, visible=True, bc_type=BCType.STICKY):
        super().__init__(type="sphere", pos=pos, size=size, visible=visible, bc_type=bc_type)

class CylinderCollider(Collider):
    def __init__(self, pos=[0,0,0], size=0.1, visible=True, bc_type=BCType.STICKY):
        super().__init__(type="cylinder", pos=pos, size=size, visible=visible, bc_type=bc_type)


# ground collision response
@ti.kernel
def ground_collision_kernel(pos: ti.template(), old_pos:ti.template(), ground_pos: ti.f32, inv_mass: ti.template()):
    for i in pos:
        if inv_mass[i] != 0.0:
            if pos[i][1] < ground_pos:
                pos[i] = old_pos[i]
                pos[i][1] = ground_pos


# Position Based Collision Response

@ti.kernel
def sphere_collision_kernel(pos: ti.template(), old_pos:ti.template(), sphere_pos: ti.template(), sphere_radius: ti.f32, inv_mass: ti.template(),  dt: ti.f32, vel: ti.template(), is_colliding: ti.template(), bc_type:ti.i32):
    for i in ti.grouped(pos):
        if inv_mass[i] != 0.0:
            offset_to_center = pos[i] - sphere_pos
            dist = offset_to_center.norm()
            if dist <= sphere_radius:
                is_colliding[i] = 1
                
                # 计算碰撞点的法向量（从球心指向碰撞点的单位向量）
                normal = offset_to_center / dist
                
                # 将点移动到球面上
                pos[i] = sphere_pos + normal * sphere_radius
                
                # 更新速度
                vel[i] = (pos[i] - old_pos[i])/dt

                # 计算碰撞后的速度
                if bc_type==0:  # STICKY
                    vel[i]=0.0
                elif bc_type==1:   # REFLECT  
                    vel_normal = ti.math.dot(vel[i], normal) * normal
                    vel_tangent = vel[i] - vel_normal
                    vel[i] = vel_tangent - vel_normal  # 反转法向速度分量
                

@ti.kernel
def cylinder_collision_kernel(pos: ti.template(), old_pos:ti.template(), cylinder_pos: ti.template(), cylinder_radius: ti.f32, inv_mass: ti.template(),  dt: ti.f32, vel: ti.template(), is_colliding: ti.template(), bc_type:ti.i32):
    for i in ti.grouped(pos):
        if inv_mass[i] != 0.0:
            # 计算点到圆柱轴线的最短距离（在xy平面上）
            p = pos[i] - cylinder_pos
            # 投影到xy平面
            p_xy = ti.Vector([p[0], p[1]])  # 改为xy平面
            dist = p_xy.norm()
            
            if dist <= cylinder_radius:
                is_colliding[i] = 1
                
                # 计算碰撞点的法向量（从轴线指向碰撞点的单位向量，在xy平面上）
                normal_xy = p_xy / dist
                # 构建3D法向量（z分量为0）
                normal = ti.Vector([normal_xy[0], normal_xy[1], 0.0])  # z分量为0
                
                # 将点移动到圆柱表面
                pos[i] = cylinder_pos + ti.Vector([normal[0] * cylinder_radius, normal[1] * cylinder_radius, p[2]])  # 保持z坐标不变
                
                # 更新速度
                vel[i] = (pos[i] - old_pos[i])/dt

                if bc_type==0:  # STICKY
                    vel[i]=0.0
                elif bc_type==1: # REFLECT
                    vel_normal = ti.math.dot(vel[i], normal) * normal
                    vel_tangent = vel[i] - vel_normal
                    vel[i] = vel_tangent - vel_normal  # 反转法向速度分量


def visualize_colliders(colliders, out_dir):
    """可视化碰撞体，既可以独立输出每个碰撞体，也可以合并输出所有碰撞体
    Args:
        colliders: 碰撞体列表
        out_dir: 输出目录
    """
    from engine.mesh_io import create_sphere_mesh,create_cylinder_mesh, write_mesh
    import os
    import logging

    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    logging.info(f"Visualizing colliders to {out_dir}")
    logging.info(f"Collider count: {len(colliders)}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for idx, collider in enumerate(colliders):
        if not collider.visible:
            continue
        if collider.type == "sphere":
            logging.info(f"Collider {idx}: Sphere at {collider.pos} with radius {collider.size}\nOutput to {out_dir}/mesh/static_collider_{idx}.obj")
            vertices, triangles = create_sphere_mesh(collider.pos, collider.size)
            # 独立输出
            write_mesh(out_dir + f"/mesh/static_collider_{idx}", vertices, triangles)
            # 收集合并数据
            all_vertices.append(vertices)
            all_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)
        elif collider.type == "cylinder":
            logging.info(f"Collider {idx}: Cylinder at {collider.pos} with radius {collider.size}\nOutput to {out_dir}/mesh/static_collider_{idx}")
            vertices, triangles = create_cylinder_mesh(collider.pos, collider.size)
            # 独立输出
            write_mesh(out_dir + f"/mesh/static_collider_{idx}", vertices, triangles)
            # 收集合并数据
            all_vertices.append(vertices)
            all_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)
    
    # 合并输出
    if all_vertices:
        combined_vertices = np.concatenate(all_vertices)
        combined_triangles = np.concatenate(all_triangles)
        write_mesh(out_dir + "/mesh/static_colliders_combined", combined_vertices, combined_triangles)
        logging.info(f"Combined colliders output to {out_dir}/mesh/static_colliders_combined.ply")

            
import taichi as ti
from taichi.math import vec2, vec3, dot, clamp, length, sign, sqrt, min, max

# ref: https://iquilezles.org/articles/distfunctions/
@ti.func
def sphere(pos, radius):
    return (pos.norm() - radius)

@ti.func
def box(p,b):
    q = abs(p) - b
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0)
    

@ti.func
def torus(p,t:vec2):
    q = vec2(length(p.xz)-t.x,p.y)
    return length(q)-t.y

@ti.func
def plane(pos, normal, height):
    return pos.dot(normal) + height

@ti.func
def cone(p, c, h):
    q = h*vec2(c.x/c.y,-1.0)
    w = vec2( length(p.xz), p.y )
    a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 )
    b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 )
    k = sign( q.y )
    d = min(dot( a, a ),dot(b, b))
    s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  )
    return sqrt(d)*sign(s)

@ti.func
def union(a, b):
    return min(a, b)

@ti.func
def intersection(a, b):
    return max(a, b)

@ti.func
def subtraction(a, b):
    return max(-a, b)


@ti.func
def triangle(p:vec3, a:vec3, b:vec3, c:vec3):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = ba.cross(ac)

    res = nor.dot(pa) * nor.dot(pa) / nor.norm_sqr()
    if sqrt(sign(ba.cross(nor).dot(pa)) + sign(cb.cross(nor).dot(pb)) + sign(ac.cross(nor).dot(pc))) < 2.0:
        res = min(min((ba * clamp(ba.dot(pa) / ba.dot(ba), 0.0, 1.0) - pa).norm(),
                      (cb * clamp(cb.dot(pb) / cb.dot(cb), 0.0, 1.0) - pb).norm()),
                  (ac * clamp(ac.dot(pc) / ac.dot(ac), 0.0, 1.0) - pc).norm())
    return res


@ti.func
def collision_response_sdf(pos:ti.template(), sdf):
    sdf_epsilon = 1e-4
    grid_idx = ti.Vector([pos.x * sdf.resolution, pos.y * sdf.resolution, pos.z * sdf.resolution], ti.i32)
    grid_idx = ti.math.clamp(grid_idx, 0, sdf.resolution - 1)
    normal = sdf.grad[grid_idx]
    sdf_val = sdf.val[grid_idx]
    assert 1 - 1e-4 < normal.norm() < 1 + 1e-4, f"sdf normal norm is not one: {normal.norm()}" 
    if sdf_val < sdf_epsilon:
        pos -= sdf_val * normal

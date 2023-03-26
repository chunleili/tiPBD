from read_tet import read_tet_mesh
import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

def get_barycentric_coord(p,a,b,c,d):
    vap = p-a
    vbp = p-b
    vab = b-a
    vac = c-a
    vad = d-a
    vbc = c-b
    vbd = d-b
    va6 = np.dot(np.cross(vbp,vbd),vbc)
    vb6 = np.dot(np.cross(vap,vac),vad)
    vc6 = np.dot(np.cross(vap,vad),vab)
    vd6 = np.dot(np.cross(vap,vab),vac)
    v6 = 1/np.dot(vab,np.cross(vac,vad))
    is_in_tet_flag = (va6>=0 and vb6>=0 and vc6>=0 and vd6>=0 and va6+vb6+vc6+vd6<=1)
    return  is_in_tet_flag, np.array([va6*v6,vb6*v6,vc6*v6,vd6*v6])


def is_in_tet(p, pos, tet_indices):
    a, b, c, d = tet_indices
    p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
    A = np.array([p1-p0, p2-p0, p3-p0]).transpose()
    b = p - p0
    x = np.linalg.solve(A, b)
    return (all(x >= 0) and sum(x) <= 1), x

def compute_mapping(coarse_pos,coarse_tet_indices, fine_pos, fine_tet_indices):
    coarse_nv = coarse_pos.shape[0]
    coarse_nt = coarse_tet_indices.shape[0]
    fine_nv = fine_pos.shape[0]
    fine_nt = fine_tet_indices.shape[0]
    coarse_in_fine_tet_indx = np.zeros(coarse_nv, dtype=np.int32)
    coarse_in_fine_tet_indx.fill(-1)
    coarse_in_fine_tet_coord = np.zeros((coarse_nv, 3), dtype=np.float64)

    fine_in_coarse_tet_indx = np.zeros(fine_nv, dtype=np.int32)
    fine_in_coarse_tet_indx.fill(-1)
    fine_in_coarse_tet_coord = np.zeros((fine_nv, 3), dtype=np.float64)

    for i in range(coarse_nv):
        for t in range(fine_nt):
            flag, x =  is_in_tet(coarse_pos[i], fine_pos, fine_tet_indices[t])
            flag1, x1 = get_barycentric_coord(coarse_pos[i], fine_pos[fine_tet_indices[t][0]], fine_pos[fine_tet_indices[t][1]], fine_pos[fine_tet_indices[t][2]], fine_pos[fine_tet_indices[t][3]])
            # if(flag==flag1):
            #     print("flag is equal")
            # else:
            #     print("flag is not equal")
            if (x==x1):
                print("x is equal")
            

            if flag:
                coarse_in_fine_tet_indx[i] = t
                coarse_in_fine_tet_coord[i] = x
                break
            
    for i in range(fine_nv):
        for t in range(coarse_nt):
            flag, x =  is_in_tet(fine_pos[i], coarse_pos, coarse_tet_indices[t])
            if flag:
                fine_in_coarse_tet_indx[i] = t
                fine_in_coarse_tet_coord[i] = x
                break
    return coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord

"""
    Compute restriction operator R: 
            x_c = R @ x_f, x_c is coarse vertex positions, x_f is fine vertex positions
    Parameters:
    n: number of fine vertices
    m: number of coarse vertices
    Output:
        R_coo: restriction operator in coo format
"""
def compute_R(n, m, coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_tet_indices):
    row = np.zeros(4 * m,dtype=np.int32)
    col = np.zeros(4 * m,dtype=np.int32)
    val = np.zeros(4 * m,dtype=np.float64)
    for i in range(m):
        row[4 * i: 4 * i + 4] = [i, i, i, i]
        fine_idx = coarse_in_fine_tet_indx[i]
        a, b, c, d = fine_tet_indices[fine_idx]
        u, v, w = coarse_in_fine_tet_coord[i]
        col[4 * i: 4 * i + 4] = [a, b, c, d]
        val[4 * i: 4 * i + 4] = [1-u-v-w, u, v, w]
    R_coo = coo_matrix((val, (row, col)), shape=(m, n))
    return R_coo

"""
    Compute prolongation operator P
    n: number of fine vertices
    m: number of coarse vertices
"""
def compute_P(n, m, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord, coarse_tet_indices):
    row = np.zeros(4 * n,dtype=np.int32)
    col = np.zeros(4 * n,dtype=np.int32)
    val = np.zeros(4 * n,dtype=np.float64)
    for i in range(n):
        row[4 * i: 4 * i + 4] = [i, i, i, i]
        coarse_idx = fine_in_coarse_tet_indx[i]
        a, b, c, d = coarse_tet_indices[coarse_idx]
        col[4 * i: 4 * i + 4] = [a, b, c, d]
        u, v, w = fine_in_coarse_tet_coord[i]
        val[4 * i: 4 * i + 4] = [1-u-v-w, u, v, w]
    P_coo = coo_matrix((val, (row, col)), shape=(n, m))
    return P_coo


if __name__ == "__main__":
    is_mapping_computed = False
    fine_mesh = "models/bunny1000_2000/bunny2k"
    coarse_mesh = "models/bunny1000_2000/bunny1k"
    
    coarse_pos, coarse_tet_indices, coarse_face_indices = read_tet_mesh(coarse_mesh)
    fine_pos, fine_tet_indices, fine_face_indices = read_tet_mesh(fine_mesh)
    
    if not is_mapping_computed:
        coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_in_coarse_tet_indx,  fine_in_coarse_tet_coord = compute_mapping(coarse_pos, coarse_tet_indices, fine_pos, fine_tet_indices)
        np.savetxt("models/bunny1000_2000/coarse_in_fine_tet_indx.txt", coarse_in_fine_tet_indx, fmt="%d")
        np.savetxt("models/bunny1000_2000/coarse_in_fine_tet_coord.txt", coarse_in_fine_tet_coord, fmt="%.6f")

        np.savetxt("models/bunny1000_2000/fine_in_coarse_tet_indx.txt", fine_in_coarse_tet_indx, fmt="%d")
        np.savetxt("models/bunny1000_2000/fine_in_coarse_tet_coord.txt", fine_in_coarse_tet_coord, fmt="%.6f")
    else:
        coarse_in_fine_tet_indx = np.loadtxt("models/bunny1000_2000/coarse_in_fine_tet_indx.txt", dtype=np.int32)
        coarse_in_fine_tet_coord = np.loadtxt("models/bunny1000_2000/coarse_in_fine_tet_coord.txt", dtype=np.float64)
        fine_in_coarse_tet_indx = np.loadtxt("models/bunny1000_2000/fine_in_coarse_tet_indx.txt", dtype=np.int32)
        fine_in_coarse_tet_coord = np.loadtxt("models/bunny1000_2000/fine_in_coarse_tet_coord.txt", dtype=np.float64)
        
    n = fine_pos.shape[0]
    m = coarse_pos.shape[0]
    R = compute_R(n, m, coarse_in_fine_tet_indx, coarse_in_fine_tet_coord, fine_tet_indices)
    mmwrite("models/bunny1000_2000/R.mtx", R)
    P = compute_P(n, m, fine_in_coarse_tet_indx, fine_in_coarse_tet_coord, coarse_tet_indices)
    mmwrite("models/bunny1000_2000/P.mtx", P)
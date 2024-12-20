#include "meshio_tetgen.h"
#include <fstream>
#include <sstream>
#include <tuple>

// void read_tetgen(std::string filename)
std::tuple<std::vector<float>, std::vector<int>, std::vector<int>> read_tetgen(std::string filename)
{
    std::vector<int> tet_indices;
    std::vector<int> face_indices;
    std::vector<float> pos;


    std::string ele_file_name = filename + ".ele";
    std::string node_file_name = filename + ".node";
    std::string face_file_name = filename + ".face";


    std::ifstream f(node_file_name);
    std::string line;
    std::getline(f, line);
    std::istringstream iss(line);
    int NV;
    iss >> NV;
    pos.resize(NV * 3);
    for (int i = 0; i < NV; i++)
    {
        std::getline(f, line);
        std::istringstream iss(line);
        int idx;
        float x, y, z;
        iss >> idx >> x >> y >> z;
        pos[i * 3] = x;
        pos[i * 3 + 1] = y;
        pos[i * 3 + 2] = z;
    }

    std::ifstream f2(ele_file_name);
    std::getline(f2, line);
    std::istringstream iss2(line);
    int NT;
    iss2 >> NT;
    tet_indices.resize(NT * 4);
    for (int i = 0; i < NT; i++)
    {
        std::getline(f2, line);
        std::istringstream iss2(line);
        int idx;
        int a, b, c, d;
        iss2 >> idx >> a >> b >> c >> d;
        tet_indices[i * 4] = a;
        tet_indices[i * 4 + 1] = b;
        tet_indices[i * 4 + 2] = c;
        tet_indices[i * 4 + 3] = d;
    }

    std::ifstream f3(face_file_name);
    std::getline(f3, line);
    std::istringstream iss3(line);
    int NF;
    iss3 >> NF;
    face_indices.resize(NF * 3);
    for (int i = 0; i < NF; i++)
    {
        std::getline(f3, line);
        std::istringstream iss3(line);
        int idx;
        int a, b, c;
        iss3 >> idx >> a >> b >> c;
        face_indices[i * 3] = a;
        face_indices[i * 3 + 1] = b;
        face_indices[i * 3 + 2] = c;
    }

    return std::move(std::make_tuple(pos, tet_indices, face_indices));
}




// def read_tetgen(filename):
//     """
//     读取tetgen生成的网格文件，返回顶点坐标、单元索引、面索引

//     Args:
//         filename: 网格文件名，不包含后缀名

//     Returns:
//         pos: 顶点坐标，shape=(NV, 3)
//         tet_indices: 单元索引，shape=(NT, 4)
//         face_indices: 面索引，shape=(NF, 3)
//     """
//     import numpy as np

//     ele_file_name = filename + ".ele"
//     node_file_name = filename + ".node"
//     face_file_name = filename + ".face"

//     with open(node_file_name, "r") as f:
//         lines = f.readlines()
//         NV = int(lines[0].split()[0])
//         pos = np.zeros((NV, 3), dtype=np.float32)
//         for i in range(NV):
//             pos[i] = np.array(lines[i + 1].split()[1:], dtype=np.float32)

//     with open(ele_file_name, "r") as f:
//         lines = f.readlines()
//         NT = int(lines[0].split()[0])
//         tet_indices = np.zeros((NT, 4), dtype=np.int32)
//         for i in range(NT):
//             tet_indices[i] = np.array(lines[i + 1].split()[1:], dtype=np.int32)

//     with open(face_file_name, "r") as f:
//         lines = f.readlines()
//         NF = int(lines[0].split()[0])
//         face_indices = np.zeros((NF, 3), dtype=np.int32)
//         for i in range(NF):
//             face_indices[i] = np.array(lines[i + 1].split()[1:-1], dtype=np.int32)

//     return pos, tet_indices, face_indices


void write_tetgen(std::string filename, std::vector<float> &points, std::vector<int> &tet_indices, std::vector<int> &tri_indices)
{
    std::string node_mesh = filename + ".node";
    std::string ele_mesh = filename + ".ele";
    std::string face_mesh = filename + ".face";
    std::ofstream f(node_mesh);
    f << points.size() / 3 << " 3 0 0\n";
    for (int i = 0; i < points.size() / 3; i++)
    {
        f << i << " " << points[i * 3] << " " << points[i * 3 + 1] << " " << points[i * 3 + 2] << "\n";
    }

    std::ofstream f2(ele_mesh);
    f2 << tet_indices.size() / 4 << " 4 0\n";
    for (int i = 0; i < tet_indices.size() / 4; i++)
    {
        f2 << i << " " << tet_indices[i * 4] << " " << tet_indices[i * 4 + 1] << " " << tet_indices[i * 4 + 2] << " " << tet_indices[i * 4 + 3] << "\n";
    }

    std::ofstream f3(face_mesh);
    f3 << tri_indices.size() / 3 << " 3 0\n";
    for (int i = 0; i < tri_indices.size() / 3; i++)
    {
        f3 << i << " " << tri_indices[i * 3] << " " << tri_indices[i * 3 + 1] << " " << tri_indices[i * 3 + 2] << " -1\n";
    }
}

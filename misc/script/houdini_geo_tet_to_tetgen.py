import os
import json


def read_geo(from_path):
    with open(from_path, "r") as f:
        data = json.load(f)

    # 读取顶点个数等信息
    pointcount = data[5]  # 点个数
    vertexcount = data[7]
    primitivecount = data[9]  # 四面体个数

    # 读取四面体的索引
    topology = data[13]
    pointref = topology[1]
    tet_indices = pointref[1]  # 四面体的索引，是一个一维数组

    # 读取顶点的位置
    attributes = data[15]
    pointattributes = attributes[1]
    positions = pointattributes[0][1][7][5]

    return tet_indices, positions, pointcount, vertexcount, primitivecount


def write_tetgen(tet_indices, positions, pointcount, primitivecount, to_path, gen_face=False):
    # 写入tetgen的node文件(也就是顶点的位置)
    node_file = to_path + ".node"
    if os.path.exists(node_file):
        print("remove file: " + node_file)
        os.remove(node_file)
    with open(node_file, "w") as f:
        f.write(str(pointcount) + "  3  0  0\n")
        for i in range(pointcount):
            f.write(
                "   "
                + str(i)
                + "    "
                + str(positions[i][0])
                + "  "
                + str(positions[i][1])
                + "  "
                + str(positions[i][2])
                + "\n"
            )

    # 写入tetgen的ele文件(也就是四面体的索引)
    ele_file = to_path + ".ele"
    if os.path.exists(ele_file):
        print("remove file: " + ele_file)
        os.remove(ele_file)
    with open(ele_file, "w") as f:
        f.write(str(primitivecount) + "  4  0\n")
        for i in range(primitivecount):
            f.write(
                "   "
                + str(i)
                + "    "
                + str(tet_indices[i * 4])
                + "  "
                + str(tet_indices[i * 4 + 1])
                + "  "
                + str(tet_indices[i * 4 + 2])
                + "  "
                + str(tet_indices[i * 4 + 3])
                + "\n"
            )

    # 写入tetgen的face文件(也就是三角面的索引)
    face_file = to_path + ".face"
    if os.path.exists(face_file):
        print("remove file: " + face_file)
        os.remove(face_file)

    if gen_face:
        # 由于本身没有三角面，所以如果想生成face，就自己遍历一遍
        facecount = 0
        for i in range(primitivecount):
            facecount += 4
        with open(face_file, "w") as f:
            f.write(str(facecount) + " 0\n")
            face_i = 0
            for i in range(primitivecount):
                f.write(
                    "    "
                    + str(face_i)
                    + "    "
                    + str(tet_indices[i * 4])
                    + "    "
                    + str(tet_indices[i * 4 + 2])
                    + "    "
                    + str(tet_indices[i * 4 + 1])
                    + "  -1\n"
                )
                face_i += 1
                f.write(
                    "    "
                    + str(face_i)
                    + "    "
                    + str(tet_indices[i * 4])
                    + "    "
                    + str(tet_indices[i * 4 + 3])
                    + "    "
                    + str(tet_indices[i * 4 + 2])
                    + "  -1\n"
                )
                face_i += 1
                f.write(
                    "    "
                    + str(face_i)
                    + "    "
                    + str(tet_indices[i * 4])
                    + "    "
                    + str(tet_indices[i * 4 + 1])
                    + "    "
                    + str(tet_indices[i * 4 + 3])
                    + "  -1\n"
                )
                face_i += 1
                f.write(
                    "    "
                    + str(face_i)
                    + "    "
                    + str(tet_indices[i * 4 + 1])
                    + "    "
                    + str(tet_indices[i * 4 + 2])
                    + "    "
                    + str(tet_indices[i * 4 + 3])
                    + "  -1\n"
                )
                face_i += 1

    print("\n\nwrite tetgen file success! \nnode file: " + node_file + "\nele file: " + ele_file)


if __name__ == "__main__":
    from_path = "model/bunny1000_dilate/bunny1000_dilate.geo"
    to_path = from_path[:-4]

    tet_indices, positions, pointcount, vertexcount, primitivecount = read_geo(from_path)

    write_tetgen(tet_indices, positions, pointcount, primitivecount, to_path, gen_face=True)

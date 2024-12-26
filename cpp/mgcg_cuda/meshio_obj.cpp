#include "meshio_obj.h"

void write_obj(std::string out_mesh_name, Field3f &pos, Field3i& tri_vis)
{
    std::ofstream myfile;
    myfile.open(out_mesh_name);
    for (int i = 0; i < pos.size(); i++)
    {
        myfile << "v " << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << "\n";
    }
    for (int i = 0; i < tri_vis.size(); i++)
    {
        myfile << "f " << tri_vis[i][0] + 1 << " " << tri_vis[i][1] + 1 << " " << tri_vis[i][2] + 1 << "\n";
    }
}
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <iterator>


std::array<float,3> inline normalize(std::array<float,3> v)
{
    float norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return {v[0]/norm, v[1]/norm, v[2]/norm};
}

std::array<float,3> inline normalize_diff(std::array<float,3> &v1,  std::array<float,3> &v2)
{
    std::array<float,3> diff = {v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]};
    float norm = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
    return {diff[0]/norm, diff[1]/norm, diff[2]/norm};
}

float inline dot(std::array<float,3> a, std::array<float,3> b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


struct InitFill()
{
    std::vector<std::array<int,2>> edges;
    std::vector<float> inv_mass;
    std::vector<std::array<float,3>> pos;
    std::vector<std::vector<int>> adjacent_edges;
    std::vector<int> num_adjacent_edge;
    std::vector<std::vector<int>> adjacent_edge_abc;
    std::vector<int> ii, jj;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> data;
}


void init_A_CSR_pattern()
{
    indptr.resize(NE+1);
    indices.resize(num_nonz);
    data.resize(num_nonz);

    indptr[0] = 0;
    for(int i=0; i<NE; i++)
    {
        int num_adj_i = num_adjacent_edge[i];
        indptr[i+1] = indptr[i] + num_adj_i + 1;
        for(int j=0; j<num_adj_i; j++)
        {
            indices[indptr[i]+j] = adjacent_edges[i][j];
        }
        indices[indptr[i+1]-1] = i;
    }
}


void csr_index_to_coo_index()
{
    ii.resize(num_nonz);
    jj.resize(num_nonz);
    for(int i=0; i<NE; i++)
    {
        for(int j=indptr[i]; j<indptr[i+1]; j++)
        {
            ii[j] = i;
            jj[j] = indices[j];
        }
    }
}


void init_adj_edge(std::vector<std::array<int,2>> &edges)
{
    std::unordered_map<int, std::set<int>> vertex_to_edges;
    for(int edge_index=0; edge_index<edges.size(); edge_index++)
    {
        int v1 = edges[edge_index][0];
        int v2 = edges[edge_index][1];
        if (vertex_to_edges.find(v1) == vertex_to_edges.end())
            vertex_to_edges[v1] = std::set<int>();
        if (vertex_to_edges.find(v2) == vertex_to_edges.end())
            vertex_to_edges[v2] = std::set<int>();
        vertex_to_edges[v1].insert(edge_index);
        vertex_to_edges[v2].insert(edge_index);
    }

    adjacent_edges.resize(edges.size());
    for(int edge_index=0; edge_index<edges.size(); edge_index++)
    {
        int v1 = edges[edge_index][0];
        int v2 = edges[edge_index][1];
        std::set<int> adj; //adjacent edges of one edge
        std::set_union(vertex_to_edges[v1].begin(), vertex_to_edges[v1].end(), vertex_to_edges[v2].begin(), vertex_to_edges[v2].end(), std::inserter(adj, adj.begin()));
        adj.erase(edge_index);
        adjacent_edges[edge_index] = std::vector<int>(adj.begin(), adj.end());
    }

    //calc num_adjacent_edge
    for(auto adj:adjacent_edges)
    {
        num_adjacent_edge.push_back(adj.size());
    }

    NE = edges.size();

    adjacent_edge_abc.resize(NE);
    for(int i=0; i<NE; i++)
    {
        // adjacent_edge_abc[i].resize(num_adjacent_edge[i]*3);
        adjacent_edge_abc[i].resize(20*3);
        std::fill(adjacent_edge_abc[i].begin(), adjacent_edge_abc[i].end(), -1);
    }
}

int calc_num_nonz(std::vector<int> &num_adjacent_edge)
{
    int num_nonz = 0;
    for(auto num_adj:num_adjacent_edge)
    {
        num_nonz += num_adj;
    }
    num_nonz += num_adjacent_edge.size();

    return num_nonz;
}


void init_adjacent_edge_abc(std::vector<std::array<int,2>> &edges, std::vector<std::vector<int>> &adjacent_edges, std::vector<int> &num_adjacent_edge, std::vector<std::vector<int>> &adjacent_edge_abc)
{
    for(int i=0; i<edges.size(); i++)
    {
        auto ii0 = edges[i][0];
        auto ii1 = edges[i][1];

        auto num_adj = num_adjacent_edge[i];
        for(int j=0; j<num_adj; j++)
        {
            auto ia = adjacent_edges[i][j];
            if (ia == i)
                continue;
            auto jj0 = edges[ia][0];
            auto jj1 = edges[ia][1];
            auto a = -1;
            auto b = -1;
            auto c = -1;
            if (ii0 == jj0)
            {
                a = ii0;
                b = ii1;
                c = jj1;
            }
            else if (ii0 == jj1)
            {
                a = ii0;
                b = ii1;
                c = jj0;
            }
            else if (ii1 == jj0)
            {
                a = ii1;
                b = ii0;
                c = jj1;
            }
            else if (ii1 == jj1)
            {
                a = ii1;
                b = ii0;
                c = jj0;
            }
            adjacent_edge_abc[i][j*3] = a;
            adjacent_edge_abc[i][j*3+1] = b;
            adjacent_edge_abc[i][j*3+2] = c;
        }
    }
}

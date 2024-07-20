#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;
using py::array_t;

unordered_map<int, vector<int>> init_adj_edge(array_t<int>& edges_) {
    auto edges = edges_.unchecked<2>(); 
    auto NE = edges.shape(0);

    unordered_map<int, unordered_set<int>> vertex_to_edges;
    for (int i = 0; i < NE; i++) {
        int v1 = edges(i,0);
        int v2 = edges(i,1);
        if (vertex_to_edges.find(v1) == vertex_to_edges.end()) {
            vertex_to_edges[v1] = unordered_set<int>();
        }
        if (vertex_to_edges.find(v2) == vertex_to_edges.end()) {
            vertex_to_edges[v2] = unordered_set<int>();
        }
        vertex_to_edges[v1].insert(i);
        vertex_to_edges[v2].insert(i);
    }

    unordered_map<int, vector<int>> all_adjacent_edges;
    for (int i = 0; i < NE; i++) {
        int v1 = edges(i,0);
        int v2 = edges(i,1);
        unordered_set<int> adjacent_edges = vertex_to_edges[v1];
        for (int edge_index : vertex_to_edges[v2]) {
            adjacent_edges.insert(edge_index);
        }
        adjacent_edges.erase(i);
        all_adjacent_edges[i] = vector<int>(adjacent_edges.begin(), adjacent_edges.end());
    }

    return all_adjacent_edges;
}


PYBIND11_MODULE(init_adj, m) {
    m.def("init_adj_edge", &init_adj_edge);
}
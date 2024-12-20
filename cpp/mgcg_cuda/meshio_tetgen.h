#pragma once
#include <vector>
#include <string>
#include <tuple>

void write_tetgen(std::string filename, std::vector<float> &points, std::vector<int> &tet_indices, std::vector<int> &tri_indices);

std::tuple<std::vector<float>, std::vector<int>, std::vector<int>> read_tetgen(std::string filename);

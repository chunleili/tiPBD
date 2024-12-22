#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <string>

#include "fastmg.cu"

template<typename T>
void loadtxt(std::string filename, T &M)
{
//   printf("Loading %s with FieldXi\n", filename.c_str());
  std::ifstream inputFile(filename);
  std::string line;

  if (!inputFile.is_open())
  {
    std::cerr << "Error: Fail to open file" << std::endl;
    return;
  }

  unsigned int rows = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    int val;
    M.resize(rows + 1);
    while (iss >> val)
    {
      M[rows].push_back(val);
    }
    rows++;
  }

}


void main()
{
    fastFill = new FastFill{};
    std::vector<std::vector<int>> edges;
    loadtxt<std::vector<std::vector<int>>>("E:/Dev/tiPBD/cpp/mgcg_cuda/edge.txt", edges);
    for(auto &edge: edges)
    {
        fastFill->edges.push_back({edge[0], edge[1]});
    }

    fastFill->init_adj_edge(fastFill->edges);

    fastFill->init_adjacent_edge_abc();

}
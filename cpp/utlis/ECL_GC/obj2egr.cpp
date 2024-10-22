/*
Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>
#include <tuple>
#include <algorithm>
#include "ECLgraph.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <stdio.h>

ECLgraph g;

using namespace std;

struct Edge {
    int vertex1;
    int vertex2;
};

typedef struct tetrahedron{
    int ELENUM;
    int vertex1;
    int vertex2;
    int vertex3;
    int vertex4;} 
ELE;

int nodeNum, edgeNum;
std::vector<Edge> lineGraphEdges;

bool edge_exist(std::vector<Edge> edges, int vertex1, int vertex2) {
    for (const auto& edge : edges) {
        if ((edge.vertex1 == vertex1 && edge.vertex2 == vertex2) ||
            (edge.vertex1 == vertex2 && edge.vertex2 == vertex1)) {
            return true;
        }
    }
    return false;
}


void objToDualGraph(const std::string& inputFilename) {
    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Failed to open input file." << std::endl;
        return;
    }

    std::vector<Edge> edges;

    std::string line;
    while (std::getline(inputFile, line)) {
        if (line.substr(0, 2) == "f ") {
            std::istringstream iss(line.substr(2));
            int vertexIndex1, vertexIndex2, vertexIndex3;
            if (iss >> vertexIndex1 >> vertexIndex2 >> vertexIndex3) {
                
                // Add the edge only if it doesn't already exist
                if (!edge_exist(edges, vertexIndex1, vertexIndex2)) {
                    edges.push_back({vertexIndex1, vertexIndex2});
                }

               
                if (!edge_exist(edges, vertexIndex2, vertexIndex3)) {
                    edges.push_back({vertexIndex2, vertexIndex3});
                }

            
                if (!edge_exist(edges, vertexIndex3, vertexIndex1)) {
                    edges.push_back({vertexIndex3, vertexIndex1});
                }
            }
        }
    }

    for (int i = 0; i < edges.size(); ++i) {
        for (int j = i + 1; j < edges.size(); ++j) {
            if (edges[i].vertex1 == edges[j].vertex1 || edges[i].vertex1 == edges[j].vertex2 ||
                edges[i].vertex2 == edges[j].vertex1 || edges[i].vertex2 == edges[j].vertex2) {
                lineGraphEdges.push_back({i + 1, j + 1}); // Assuming 1-based indexing
            }
        }
    }

    nodeNum = edges.size();
    edgeNum = lineGraphEdges.size() * 2;
    inputFile.close();

    std::cout << "Conversion completed successfully." << std::endl;
}


bool isNeighbor(ELE e1,ELE e2){
    if(e1.vertex1 == e2.vertex1 || e1.vertex1 == e2.vertex2 || e1.vertex1 == e2.vertex3 || e1.vertex1 == e2.vertex4){
        return true;
    }
    if(e1.vertex2 == e2.vertex1 || e1.vertex2 == e2.vertex2 || e1.vertex2 == e2.vertex3 || e1.vertex2 == e2.vertex4){
        return true;
    }
    if(e1.vertex3 == e2.vertex1 || e1.vertex3 == e2.vertex2 || e1.vertex3 == e2.vertex3 || e1.vertex3 == e2.vertex4){
        return true;
    }
    if(e1.vertex4 == e2.vertex1 || e1.vertex4 == e2.vertex2 || e1.vertex4 == e2.vertex3 || e1.vertex4 == e2.vertex4){
        return true;
    }
    return false;
}

void eleToDualGraph(const std::string& inputFilename) {
    std::ifstream inputFile(inputFilename);
    std::string line;
    int isfirst = 0;
    std::vector<ELE> vertexs;//存储顶点
    while (std::getline(inputFile, line)) {
        isfirst++;
        std::istringstream iss(line);
        std::vector<int> numbers;
        int number;
        while (iss >> number) {
            numbers.push_back(number);
        }
        if(isfirst > 1){
            ELE ele;
            ele.ELENUM = numbers[0] + 1;
            ele.vertex1 = numbers[1];
            ele.vertex2 = numbers[2];
            ele.vertex3 = numbers[3];
            ele.vertex4 = numbers[4];
            vertexs.push_back(ele);
        }

        
    }

    int max_neigh = 0;
    for(int i = 0; i < vertexs.size(); i++){
        int temp = 0;
        for(int j = i + 1;j < vertexs.size();j++){
            
            if(isNeighbor(vertexs[i],vertexs[j])){
                struct Edge edge;
                edge.vertex1 = vertexs[i].ELENUM;
                edge.vertex2 = vertexs[j].ELENUM;
                lineGraphEdges.push_back(edge);
                temp ++;
            }
        }
        if (temp > max_neigh){
            max_neigh = temp;
        }
    }

    std::cout << "max_neigh: " << max_neigh << std::endl;
        // 输出当前行的数字
        /* for (int num : numbers) {
            std::cout << num << " ";
        } */
    std::cout<<lineGraphEdges.size()<<std::endl;
    std::cout << vertexs.size() << std::endl;

        /* for(int i = 0; i < lineGraphEdges.size(); i++){
            std::cout << lineGraphEdges[i].vertex1 << " " << lineGraphEdges[i].vertex2 << std::endl;
        } */
    std::cout << std::endl;

    // 关闭文件
    nodeNum = vertexs.size();
    edgeNum = lineGraphEdges.size() * 2;
    inputFile.close();
}



ECLgraph obj2egr(int mode)
{
  printf("MatrixMarket to ECL Graph Converter (%s)\n", __FILE__);
  printf("Copyright 2016 Texas State University\n");

  if (mode == 1){
    puts("right mode == 1");
    objToDualGraph("input.obj");
  }
  else if (mode == 2){
    puts("right mode == 2");
    eleToDualGraph("input.ele");
  }
  else{
    fprintf(stderr, "ERROR: invalid mode\n\n");
    exit(-1);
  }
  
  g.nodes = nodeNum;
  g.edges = edgeNum;
  g.nindex = (int*)calloc(nodeNum + 1, sizeof(int));
  g.nlist = (int*)malloc(edgeNum * sizeof(int));
  g.eweight = NULL;
  if ((g.nindex == NULL) || (g.nlist == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}

  {
    std::vector<std::pair<int, int>> v;
    for (int i = 0; i < lineGraphEdges.size(); i++){
      v.push_back(std::make_pair(lineGraphEdges[i].vertex1 - 1, lineGraphEdges[i].vertex2 - 1));
      v.push_back(std::make_pair(lineGraphEdges[i].vertex2 - 1, lineGraphEdges[i].vertex1 - 1));
    }
    std::sort(v.begin(), v.end());

    g.nindex[0] = 0;
    for (int i = 0; i < edgeNum; i++) {
      int src = v[i].first;
      int dst = v[i].second;
      g.nindex[src + 1] = i + 1;
      g.nlist[i] = dst;
    }
  } 
  for (int i = 1; i < (nodeNum + 1); i++) {
    g.nindex[i] = std::max(g.nindex[i - 1], g.nindex[i]);
  }

  std::cout<<"nodeNum: "<<nodeNum<<std::endl;


  return g;
}
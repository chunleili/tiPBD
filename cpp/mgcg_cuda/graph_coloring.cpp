#include <iostream>
#include <cstdio>
#include <vector>
#include <set>
#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>

#include <fstream>

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


int nodeNum, edgeNum, Ks;
std::vector<Edge> lineGraphEdges;

std::vector<ELE> vertexs;//存储顶点，从0开始

vector<int> constraintTag(edgeNum);//约束所在的聚类序号，从1开始

int isNeighbor(ELE e1,ELE e2){
    int same_vertex = 0;
    if(e1.vertex1 == e2.vertex1 || e1.vertex1 == e2.vertex2 || e1.vertex1 == e2.vertex3 || e1.vertex1 == e2.vertex4){
        same_vertex++;
    }
    if(e1.vertex2 == e2.vertex1 || e1.vertex2 == e2.vertex2 || e1.vertex2 == e2.vertex3 || e1.vertex2 == e2.vertex4){
        same_vertex++;
    }
    if(e1.vertex3 == e2.vertex1 || e1.vertex3 == e2.vertex2 || e1.vertex3 == e2.vertex3 || e1.vertex3 == e2.vertex4){
        same_vertex++;
    }
    if(e1.vertex4 == e2.vertex1 || e1.vertex4 == e2.vertex2 || e1.vertex4 == e2.vertex3 || e1.vertex4 == e2.vertex4){
        same_vertex++;
    }
    return same_vertex;
}


double distance(ELE node1, ELE node2){
    double distance_temp = 0;
    std::set<int> sum_vertex;
    sum_vertex.insert(node1.vertex1);
    sum_vertex.insert(node1.vertex2);
    sum_vertex.insert(node1.vertex3);
    sum_vertex.insert(node1.vertex4);
    sum_vertex.insert(node2.vertex1);
    sum_vertex.insert(node2.vertex2);
    sum_vertex.insert(node2.vertex3);
    sum_vertex.insert(node2.vertex4);
    //std::cout << "sum_vertex.size(): " << sum_vertex.size() << std::endl;
    distance_temp = 1 - ( (double)(8 - sum_vertex.size()) / (double)sum_vertex.size());
    //std::cout << "distance_temp: " << distance_temp << std::endl;
    return distance_temp; 
}

int sum_distance_min(){
    int flag = 0;
    double min_sum_distance = 0;
    int min_sum_distance_index = 0;
    for (int i = 0; i < nodeNum; i++){
        int i_num = vertexs[i].ELENUM;
        if (constraintTag[i_num] == 0){
            double sum_distance = 0;
            //std::cout << "i: " << i_num << std::endl;
            for (int j = 0; j < nodeNum; j++){
                int j_num = vertexs[j].ELENUM;
                //std::cout << "j: " << j_num << std::endl;
                //std::cout << "constraint_num:" << constraintTag.size() << std::endl;
                if(i_num != j_num && constraintTag[j_num] == 0 && distance(vertexs[i],vertexs[j]) != 1){ 
                    //std::cout << "distance to" << j_num << " :"<< distance(vertexs[i],vertexs[j]) << std::endl;
                    sum_distance += distance(vertexs[i],vertexs[j]);
                }
            }
            //cout <<"sum_distance of " << i_num << ": " << sum_distance<< endl;
            if (flag == 0){
                min_sum_distance = sum_distance;
                min_sum_distance_index = i_num;
                flag = 1;
            }
            else if (sum_distance < min_sum_distance){
                min_sum_distance = sum_distance;
                min_sum_distance_index = i_num;
            }
            
        }
    }
    return min_sum_distance_index;
}


void eleToDualGraph(const std::string& inputFilename) {
    std::ifstream inputFile(inputFilename);
    std::string line;
    int isfirst = 0;
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

    for(int i = 0; i < vertexs.size(); i++){
            for(int j = 0; j < vertexs.size();j++){
                int weight = isNeighbor(vertexs[i],vertexs[j]);
                if(i != j && weight > 0){
                    struct Edge edge;
                    edge.vertex1 = vertexs[i].ELENUM;
                    edge.vertex2 = vertexs[j].ELENUM;
                    lineGraphEdges.push_back(edge);
                }
            }
        }

    // 关闭文件
    nodeNum = vertexs.size();
    edgeNum = lineGraphEdges.size();
    inputFile.close();
}

vector<int> bfs(int start, const unordered_map<int, vector<int>>& graph) {
    vector<int> bfsOrder; // 用于存储BFS遍历顺序
    queue<int> q;         // BFS使用的队列
    unordered_map<int, bool> visited; // 访问标记

    // 从起始节点开始
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        bfsOrder.push_back(current);

        // 遍历当前节点的所有邻居节点
        for (const int& neighbor : graph.at(current)) {
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
    return bfsOrder;
}

void GraphCluster(){

    unordered_map<int, vector<int>> graph; // 使用哈希表表示图的邻接表

    for (int i = 0; i < edgeNum; ++i) {
        int u = lineGraphEdges[i].vertex1;
        int v = lineGraphEdges[i].vertex2;
        graph[u].push_back(v);
    }


    for(int i = 0; i <= nodeNum; i++){
        constraintTag.push_back(0);
    }

    int clusterNum = 0; //  聚类序号
    while (Ks >= 1){
        //cout << "Ks:" << Ks << endl;
        int startNode = sum_distance_min();
        if (startNode == 0){
            break;
        }
        //std::cout << "startNode: " << startNode << std::endl;
        vector<int> traversalOrder = bfs(startNode, graph);

        /*
        cout << "广度优先遍历序列: ";
        for (const int& node : traversalOrder) {
            cout << node << " ";
        }
        cout << endl;
        */

        for (int i = 0; i < traversalOrder.size(); i++){
            int coreNode = traversalOrder[i];
            //cout << "coreNode:" << coreNode << endl;
            if (constraintTag[coreNode] == 0){
                clusterNum++;
                constraintTag[coreNode] = clusterNum;
                //cout << "clusterNum:" << clusterNum << endl;

                // 为邻居节点排序
                struct vertex_dis{
                    int index;
                    double distance;
                };
                vector<vertex_dis> neighbor;
                for (int j = 0; j < graph[coreNode].size(); j++){
                    vertex_dis temp;
                    temp.index = graph[coreNode][j];
                    temp.distance = distance(vertexs[coreNode - 1],vertexs[graph[coreNode][j] - 1]);
                    neighbor.push_back(temp);
                }
                std::sort(neighbor.begin(),neighbor.end(),[](vertex_dis a,vertex_dis b){return a.distance < b.distance;});
                
                //cout << "neighbor_sum:" << neighbor.size() << endl;

                int cluster_sum = 1; //聚类节点数
                for (int j = 0; j < neighbor.size(); j++){
                    int neighborNode = neighbor[j].index;
                    if (constraintTag[neighborNode] == 0 && cluster_sum < Ks){
                        constraintTag[neighborNode] = clusterNum;
                        cluster_sum++;
                    }
                }

                //如果聚类节点数小于Ks，则删除这个聚类，Ks-1
                if (cluster_sum < Ks) {
                    //cout << "cluster_sum < Ks, clusterNum: " << clusterNum << endl; 
                    for (int j = 0; j < neighbor.size(); j++){
                        int neighborNode = neighbor[j].index;
                        if (constraintTag[neighborNode] == clusterNum){
                            constraintTag[neighborNode] = 0;
                        }
                    }
                    clusterNum--;
                    
                }
            }
        }
        Ks--; 
    }
}

std::unordered_map<int, int> greedyColoring(const std::unordered_map<int, std::unordered_set<int>>& graph) {
    std::unordered_map<int, int> result;  // 存储每个节点的颜色
    int numVertices = graph.size();       // 获取图中节点的数量
    std::vector<bool> available(numVertices, true);  // 存储可用颜色

    // 为图中的每个节点着色
    for (const auto& node : graph) {
        int u = node.first;

        // 标记相邻节点的颜色为不可用
        for (int neighbor : graph.at(u)) {
            if (result.find(neighbor) != result.end()) { // 如果相邻节点已经被着色
                available[result[neighbor]] = false;
            }
        }

        // 找到第一个可用的颜色
        int cr;
        for (cr = 0; cr < numVertices; ++cr) {
            if (available[cr]) {
                break;
            }
        }

        // 为当前节点分配颜色
        result[u] = cr;

        // 重置为下一次迭代准备
        std::fill(available.begin(), available.end(), true);
    }
    return result;
}


int run(std::string model, int Ks_in, std::unordered_map<int, int>&result)
{
    cout<<model<<endl;
    eleToDualGraph(model);
    cout<<"Done eleToDualGraph"<<endl;
    // eleToDualGraph("D:\\LearnOpenGL\\XPBD_cluster\\color\\input_liver.ele");

    std::cout << "nodeNum: " << nodeNum << std::endl;
    std::cout << "edgeNum: " << edgeNum << std::endl;
    // for (int i = 0; i < lineGraphEdges.size(); i++) {
    //     std::cout << lineGraphEdges[i].vertex1 << " " << lineGraphEdges[i].vertex2 << std::endl;
    // }
    Ks=Ks_in; //FIXME: global variable may cause problem
    std::cout << "Ks: " << Ks << std::endl;

    cout<<"Doing GraphCluster"<<endl;
    GraphCluster();
    cout<<"Done GraphCluster"<<endl;
    
    // cout << "constraintTag:";
    // for (int i = 1; i <= nodeNum; i++){
    //     cout << constraintTag[i] << " ";
    // }
    
    unordered_map<int, std::unordered_set<int>> graph_Clustered;
    for (int i = 1; i <= nodeNum; i++){
        for (int j = 1; j <= nodeNum; j++){
            if (i != j && isNeighbor(vertexs[i - 1],vertexs[j - 1]) > 0 && constraintTag[i] != constraintTag[j]){
                graph_Clustered[i].insert(j);
            }
        }
    }
    
    cout << endl;
    cout << "graph Clustered!" << endl;

    result = greedyColoring(graph_Clustered);
    // std::cout << "Node Color Distribution:" << std::endl;
    int color_num = 0;
    for (int i = 1; i <= nodeNum; i++){
        if (result[constraintTag[i]] > color_num){
            color_num = result[constraintTag[i]];
        }
        // cout << "Node" << i << "   Cluster Id" << constraintTag[i] << "  Color" << result[constraintTag[i]] <<endl;
    }
    cout << "color_num: " << color_num + 1 << endl;
    return color_num + 1;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model>" << std::endl;
        return 1;
    }
    auto model = argv[1];

    cout<<"input Ks:";
    int Ks_in;
    std::cin >> Ks_in;
    std::unordered_map<int, int> result;
    int color_num = run(model, Ks_in, result);
    
    std::ofstream outfile("color.txt");
    outfile << color_num << std::endl;
    for (int i = 1; i <=nodeNum; i++)
        outfile << result[constraintTag[i]] << std::endl;
    outfile.close();
    
    return 0;
}



#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT int graph_coloring(const char* model, int* color) {
    cout<<"model:"<<model<<endl;
    std::unordered_map<int, int> result;
    int color_num = run(model, 5, result);
    for(int i = 1; i <=nodeNum; i++)
        color[i-1] = result[constraintTag[i]];

    return color_num;
}
/// defining the types and including the necessary libraries

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <array>
#include <algorithm>
#include <unordered_set>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/SparseExtra"

// namespace Eigen{ 
//     typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3fRow;
// }

using std::vector;
using std::array;
using Eigen::Map;
using Eigen::Vector3f;
using Eigen::VectorXf;
using Mat3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

// typedefs
using Vec3f = Eigen::Vector3f;
using Vec2i = std::array<int, 2>; // using Vec2i = Eigen::Vector2i;
using Vec3i = std::array<int, 3>; // using Vec3i = Eigen::Vector3i;
using Vec4i = std::array<int, 4>; // using Vec4i = Eigen::Vector4i;
using Vec43f = std::array<Vec3f, 4>; // using Vec4i = Eigen::Vector4i;
using Field1f = vector<float>;
using Field3f = vector<Vec3f>;
using Field3i = vector<Vec3i>;
using Field4i = vector<Vec4i>;
using Field2i = vector<Vec2i>;
using Field1i = vector<int>;
using Field43f = vector<Vec43f>;
using FieldXi = vector<vector<int>>;
using FieldMat3f = vector<Mat3f>;

// all use eigen types except Field43f and FieldMat3f
// using Vec3f = Eigen::Vector3f;
// using Vec2i = Eigen::Vector2i;
// using Vec3i = Eigen::Vector3i;
// using Vec4i = Eigen::Vector4i;
// using Field1f = Eigen::VectorXf;
// using Field3f = Eigen::Matrix<float, -1, 3, Eigen::RowMajor>;
// using Field3i = Eigen::Matrix<int, -1, 3, Eigen::RowMajor>;
// using Field4i = Eigen::Matrix<int, -1, 4, Eigen::RowMajor>;
// using Field2i = Eigen::Matrix<int, -1, 2, Eigen::RowMajor>;
// using FieldMat3f = vector<Mat3f>;
// using Vec43f = std::array<Vec3f, 4>;
// using Field43f = vector<Vec43f>;
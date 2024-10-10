#include "Softbody_cuda.h"

#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <algorithm>

namespace XPBD {

	__global__ void semi_euler_kernel(int num_vertices, Real* w, Real h, Real* pos,
		Real* vels, Real damping_factor) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < num_vertices) {
			if (w[idx] == 0.0f)
				return;
			vels[3 * idx + 1] += h * -9.8;
			for (int i = 0; i < 3; i++) {
				vels[3 * idx + i] *= (1.0f - damping_factor);
				pos[3 * idx + i] += h * vels[3 * idx + i];
			}
		}
	}

	__global__ void solve_distance_constraint_kernel(int num_edges, Real* w,
		Real* predic_pos, Real* pos,
		int* edges, Real* rest_length,
		int offset, Real* alpha_tilde,
		Real* lambda) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < num_edges) {
			int idx0 = edges[2 * idx], idx1 = edges[2 * idx + 1];
			Vec3f p0(predic_pos[3 * idx0 + 0], predic_pos[3 * idx0 + 1],
				predic_pos[3 * idx0 + 2]);
			Vec3f p1(predic_pos[3 * idx1 + 0], predic_pos[3 * idx1 + 1],
				predic_pos[3 * idx1 + 2]);
			Vec3f normal = p0 - p1;
			Real dis = normal.norm() - rest_length[idx];
			Vec3f n = normal.normalized();
			Real sum_inv_mass = w[idx0] + w[idx1];
			if (sum_inv_mass == 0.0)
				return;
			Real deltaLambda = (dis - lambda[idx + offset] * alpha_tilde[idx + offset]) /
				(sum_inv_mass + alpha_tilde[idx + offset]);
			lambda[idx + offset] += deltaLambda;
			Vec3f corr = 0.1f * deltaLambda * n;
			if (w[idx0] != 0.0) {
				atomicAdd(&pos[3 * idx0 + 0], -w[idx0] * corr.x);
				atomicAdd(&pos[3 * idx0 + 1], -w[idx0] * corr.y);
				atomicAdd(&pos[3 * idx0 + 2], -w[idx0] * corr.z);
			}
			if (w[idx1] != 0.0) {
				atomicAdd(&pos[3 * idx1 + 0], w[idx1] * corr.x);
				atomicAdd(&pos[3 * idx1 + 1], w[idx1] * corr.y);
				atomicAdd(&pos[3 * idx1 + 2], w[idx1] * corr.z);
			}
		}
	}

	__global__ void solve_volume_constraint_kernel(int num_tets, Real* w,
		Real* predic_pos, Real* pos,
		int* tets, Real* rest_volume,
		int offset, Real* alpha_tilde,
		Real* lambda) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < num_tets) {
			int idx0 = tets[4 * idx], idx1 = tets[4 * idx + 1],
				idx2 = tets[4 * idx + 2], idx3 = tets[4 * idx + 3];
			Vec3f p0(predic_pos[3 * idx0 + 0], predic_pos[3 * idx0 + 1],
				predic_pos[3 * idx0 + 2]);
			Vec3f p1(predic_pos[3 * idx1 + 0], predic_pos[3 * idx1 + 1],
				predic_pos[3 * idx1 + 2]);
			Vec3f p2(predic_pos[3 * idx2 + 0], predic_pos[3 * idx2 + 1],
				predic_pos[3 * idx2 + 2]);
			Vec3f p3(predic_pos[3 * idx3 + 0], predic_pos[3 * idx3 + 1],
				predic_pos[3 * idx3 + 2]);

			Real w0 = w[idx0], w1 = w[idx1], w2 = w[idx2], w3 = w[idx3];

			Real volume = (1.0f / 6.0f) * (p1 - p0).cross(p2 - p0).dot(p3 - p0);

			Vec3f grad0 = (p3 - p1).cross(p2 - p1);
			Vec3f grad1 = (p2 - p0).cross(p3 - p0);
			Vec3f grad2 = (p3 - p0).cross(p1 - p0);
			Vec3f grad3 = (p1 - p0).cross(p2 - p0);

			Real sum_w = w0 * grad0.square_norm() + w1 * grad1.square_norm() +
				w2 * grad2.square_norm() + w3 * grad3.square_norm();

			if (fabs(sum_w) < 1e-6)
				return;

			Real c = volume - rest_volume[idx];
			Real delta_lambda =
				(c - lambda[idx + offset] * alpha_tilde[idx + offset]) / (sum_w + alpha_tilde[idx + offset]);
			lambda[idx + offset] += delta_lambda;

			Real stiffness = 0.1f;
			if (w[idx0] != 0.0) {
				atomicAdd(&pos[3 * idx0 + 0],
					-stiffness * delta_lambda * w[idx0] * grad0.x);
				atomicAdd(&pos[3 * idx0 + 1],
					-stiffness * delta_lambda * w[idx0] * grad0.y);
				atomicAdd(&pos[3 * idx0 + 2],
					-stiffness * delta_lambda * w[idx0] * grad0.z);
			}
			if (w[idx1] != 0.0) {
				atomicAdd(&pos[3 * idx1 + 0],
					-stiffness * delta_lambda * w[idx1] * grad1.x);
				atomicAdd(&pos[3 * idx1 + 1],
					-stiffness * delta_lambda * w[idx1] * grad1.y);
				atomicAdd(&pos[3 * idx1 + 2],
					-stiffness * delta_lambda * w[idx1] * grad1.z);
			}
			if (w[idx2] != 0.0) {
				atomicAdd(&pos[3 * idx2 + 0],
					-stiffness * delta_lambda * w[idx2] * grad2.x);
				atomicAdd(&pos[3 * idx2 + 1],
					-stiffness * delta_lambda * w[idx2] * grad2.y);
				atomicAdd(&pos[3 * idx2 + 2],
					-stiffness * delta_lambda * w[idx2] * grad2.z);
			}
			if (w[idx3] != 0.0) {
				atomicAdd(&pos[3 * idx3 + 0],
					-stiffness * delta_lambda * w[idx3] * grad3.x);
				atomicAdd(&pos[3 * idx3 + 1],
					-stiffness * delta_lambda * w[idx3] * grad3.y);
				atomicAdd(&pos[3 * idx3 + 2],
					-stiffness * delta_lambda * w[idx3] * grad3.z);
			}
		}
	}


	

	__global__ void solve_neohookean_constraint_kernel(int num_clusters, Real* w,
		/*Real* predict_pos, */Real* pos,
		int* tets, Mat3f* dm_inv,
		Real* alpha_hydrostatic, Real* alpha_deviatoric
		, int color, int* clusterIndex, int* cluster_vertex, int* colorIndex, int* color_cluster,
		int mode, Real* delta_dis
		) {

		int idx_cluster_zero = threadIdx.x + blockDim.x * blockIdx.x; //范围：0到当前颜色所含聚类数
		//printf("idx:%d\n", idx);
		//printf("color:%d\n", color);
		//printf("idx_color:%d\n", colors[idx]);

		if (idx_cluster_zero < (colorIndex[color + 1] - colorIndex[color])) {
			//printf("idx_cluster:%d\n", idx_cluster);
			int idx_cluster = color_cluster[colorIndex[color] + idx_cluster_zero];
			if (mode == 1){
				int idx = idx_cluster;

				int idx0 = tets[4 * idx], idx1 = tets[4 * idx + 1],
					idx2 = tets[4 * idx + 2], idx3 = tets[4 * idx + 3];
				Real w0 = w[idx0];
				Real w1 = w[idx1];
				Real w2 = w[idx2];
				Real w3 = w[idx3];

				/*Vec3f p0(predict_pos[3 * idx0 + 0], predict_pos[3 * idx0 + 1],
					predict_pos[3 * idx0 + 2]);
				Vec3f p1(predict_pos[3 * idx1 + 0], predict_pos[3 * idx1 + 1],
					predict_pos[3 * idx1 + 2]);
				Vec3f p2(predict_pos[3 * idx2 + 0], predict_pos[3 * idx2 + 1],
					predict_pos[3 * idx2 + 2]);
				Vec3f p3(predict_pos[3 * idx3 + 0], predict_pos[3 * idx3 + 1],
					predict_pos[3 * idx3 + 2]);*/

				Vec3f p0(pos[3 * idx0 + 0], pos[3 * idx0 + 1],
					pos[3 * idx0 + 2]);
				Vec3f p1(pos[3 * idx1 + 0], pos[3 * idx1 + 1],
					pos[3 * idx1 + 2]);
				Vec3f p2(pos[3 * idx2 + 0], pos[3 * idx2 + 1],
					pos[3 * idx2 + 2]);
				Vec3f p3(pos[3 * idx3 + 0], pos[3 * idx3 + 1],
					pos[3 * idx3 + 2]);

				Vec3f p10 = p1 - p0;
				Vec3f p20 = p2 - p0;
				Vec3f p30 = p3 - p0;
				Mat3f Ds0(p10, p20, p30);
				Mat3f F0 = Ds0 * dm_inv[idx];
				Mat3f G = 2.0f * F0 * dm_inv[idx].transpose();
				Vec3f grad1 = G.col(0);
				Vec3f grad2 = G.col(1);
				Vec3f grad3 = G.col(2);
				Vec3f grad0 = -1.0f * (grad1 + grad2 + grad3);
				Real w_sum0 = grad0.square_norm() * w0 + grad1.square_norm() * w1 +
					grad2.square_norm() * w2 + grad3.square_norm() * w3;
				if (w_sum0 == 0.0f)
					return;
				Real c_devia = F0.col(0).square_norm() + F0.col(1).square_norm() + F0.col(2).square_norm() - 3.0f;
				Real  delta_lambda = -c_devia / (w_sum0 + alpha_deviatoric[idx]);
				Real stiffness = 0.1f;

				Vec3f delta_p0 = stiffness * delta_lambda * w[idx0] * grad0;
				Vec3f delta_p1 = stiffness * delta_lambda * w[idx1] * grad1;
				Vec3f delta_p2 = stiffness * delta_lambda * w[idx2] * grad2;
				Vec3f delta_p3 = stiffness * delta_lambda * w[idx3] * grad3;


				// solve hydrostatic constraints
				Mat3f Ds1(p10, p20, p30);
				Mat3f F1 = Ds1 * dm_inv[idx];
				Vec3f f1 = F1.col(0);
				Vec3f f2 = F1.col(1);
				Vec3f f3 = F1.col(2);

				Vec3f df0 = f2.cross(f3);
				Vec3f df1 = f3.cross(f1);
				Vec3f df2 = f1.cross(f2);
				Mat3f df(df0, df1, df2);
				Mat3f G1 = df * dm_inv[idx].transpose();
				grad1 = G1.col(0);
				grad2 = G1.col(1);
				grad3 = G1.col(2);
				grad0 = -1.0f * (grad1 + grad2 + grad3);
				Real w_sum1 = grad0.square_norm() * w0 + grad1.square_norm() * w1 +
					grad2.square_norm() * w2 + grad3.square_norm() * w3;

				if (w_sum1 == 0.0)
					return;

				Real c_hyd = F1.det() - 1.0;
				delta_lambda = -c_hyd / (w_sum1 + alpha_hydrostatic[idx]);

				delta_p0 += stiffness * delta_lambda * w[idx0] * grad0;
				delta_p1 += stiffness * delta_lambda * w[idx1] * grad1;
				delta_p2 += stiffness * delta_lambda * w[idx2] * grad2;
				delta_p3 += stiffness * delta_lambda * w[idx3] * grad3;

				/*Real length_delta_p0 = std::sqrt(delta_p0.x * delta_p0.x + delta_p0.y * delta_p0.y + delta_p0.z * delta_p0.z);
				Real length_delta_p1 = std::sqrt(delta_p1.x * delta_p1.x + delta_p1.y * delta_p1.y + delta_p1.z * delta_p1.z);
				Real length_delta_p2 = std::sqrt(delta_p2.x * delta_p2.x + delta_p2.y * delta_p2.y + delta_p2.z * delta_p2.z);
				Real length_delta_p3 = std::sqrt(delta_p3.x * delta_p3.x + delta_p3.y * delta_p3.y + delta_p3.z * delta_p3.z);

				delta_dis[idx] = (length_delta_p0 + length_delta_p1 + length_delta_p2 + length_delta_p3) / 4.0f;*/

				Real nu = 0.45f;
				Real E = 10000000.0f;

				Real mu = E / (2.0f * (1.0f + nu));
				Real lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));

				Real elastic_energy_devia = 0.5 * mu * c_devia;
				Real elastic_energy_volumetric = 0.5 * lambda * c_hyd * c_hyd; 
				delta_dis[idx] = elastic_energy_devia + elastic_energy_volumetric;


				if (w[idx0] != 0.0) {
					atomicAdd(&pos[3 * idx0 + 0], delta_p0.x);
					atomicAdd(&pos[3 * idx0 + 1], delta_p0.y);
					atomicAdd(&pos[3 * idx0 + 2], delta_p0.z);
				}
				if (w[idx1] != 0.0) {
					atomicAdd(&pos[3 * idx1 + 0], delta_p1.x);
					atomicAdd(&pos[3 * idx1 + 1], delta_p1.y);
					atomicAdd(&pos[3 * idx1 + 2], delta_p1.z);

				}
				if (w[idx2] != 0.0) {
					atomicAdd(&pos[3 * idx2 + 0], delta_p2.x);
					atomicAdd(&pos[3 * idx2 + 1], delta_p2.y);
					atomicAdd(&pos[3 * idx2 + 2], delta_p2.z);
				}
				if (w[idx3] != 0.0) {
					atomicAdd(&pos[3 * idx3 + 0], delta_p3.x);
					atomicAdd(&pos[3 * idx3 + 1], delta_p3.y);
					atomicAdd(&pos[3 * idx3 + 2], delta_p3.z);
				}
				
			}
			else if (mode == 2){
			
				for (int i = clusterIndex[idx_cluster]; i < clusterIndex[idx_cluster + 1]; i++){
					int idx = i;
					//printf("color:%d , idx:%d\n", color, idx);
					// solve deviatoric constraints
					int idx0 = tets[4 * idx], idx1 = tets[4 * idx + 1],
						idx2 = tets[4 * idx + 2], idx3 = tets[4 * idx + 3];
					Real w0 = w[idx0];
					Real w1 = w[idx1];
					Real w2 = w[idx2];
					Real w3 = w[idx3];

					/*Vec3f p0(predict_pos[3 * idx0 + 0], predict_pos[3 * idx0 + 1],
						predict_pos[3 * idx0 + 2]);
					Vec3f p1(predict_pos[3 * idx1 + 0], predict_pos[3 * idx1 + 1],
						predict_pos[3 * idx1 + 2]);
					Vec3f p2(predict_pos[3 * idx2 + 0], predict_pos[3 * idx2 + 1],
						predict_pos[3 * idx2 + 2]);
					Vec3f p3(predict_pos[3 * idx3 + 0], predict_pos[3 * idx3 + 1],
						predict_pos[3 * idx3 + 2]);*/

					Vec3f p0(pos[3 * idx0 + 0], pos[3 * idx0 + 1],
						pos[3 * idx0 + 2]);
					Vec3f p1(pos[3 * idx1 + 0], pos[3 * idx1 + 1],
						pos[3 * idx1 + 2]);
					Vec3f p2(pos[3 * idx2 + 0], pos[3 * idx2 + 1],
						pos[3 * idx2 + 2]);
					Vec3f p3(pos[3 * idx3 + 0], pos[3 * idx3 + 1],
						pos[3 * idx3 + 2]);

					Vec3f p10 = p1 - p0;
					Vec3f p20 = p2 - p0;
					Vec3f p30 = p3 - p0;
					Mat3f Ds0(p10, p20, p30);
					Mat3f F0 = Ds0 * dm_inv[idx];
					Mat3f G = 2.0f * F0 * dm_inv[idx].transpose();
					Vec3f grad1 = G.col(0);
					Vec3f grad2 = G.col(1);
					Vec3f grad3 = G.col(2);
					Vec3f grad0 = -1.0f * (grad1 + grad2 + grad3);
					Real w_sum0 = grad0.square_norm() * w0 + grad1.square_norm() * w1 +
						grad2.square_norm() * w2 + grad3.square_norm() * w3;
					if (w_sum0 == 0.0f)
						return;
					Real c_devia = F0.col(0).square_norm() + F0.col(1).square_norm() + F0.col(2).square_norm() - 3.0f;
					Real  delta_lambda = -c_devia / (w_sum0 + alpha_deviatoric[idx]);
					Real stiffness = 0.1f;

					Vec3f delta_p0 = stiffness * delta_lambda * w[idx0] * grad0;
					Vec3f delta_p1 = stiffness * delta_lambda * w[idx1] * grad1;
					Vec3f delta_p2 = stiffness * delta_lambda * w[idx2] * grad2;
					Vec3f delta_p3 = stiffness * delta_lambda * w[idx3] * grad3;


					// solve hydrostatic constraints
					Mat3f Ds1(p10, p20, p30);
					Mat3f F1 = Ds1 * dm_inv[idx];
					Vec3f f1 = F1.col(0);
					Vec3f f2 = F1.col(1);
					Vec3f f3 = F1.col(2);

					Vec3f df0 = f2.cross(f3);
					Vec3f df1 = f3.cross(f1);
					Vec3f df2 = f1.cross(f2);
					Mat3f df(df0, df1, df2);
					Mat3f G1 = df * dm_inv[idx].transpose();
					grad1 = G1.col(0);
					grad2 = G1.col(1);
					grad3 = G1.col(2);
					grad0 = -1.0f * (grad1 + grad2 + grad3);
					Real w_sum1 = grad0.square_norm() * w0 + grad1.square_norm() * w1 +
						grad2.square_norm() * w2 + grad3.square_norm() * w3;

					if (w_sum1 == 0.0)
						return;

					Real c_hyd = F1.det() - 1.0;
					delta_lambda = -c_hyd / (w_sum1 + alpha_hydrostatic[idx]);

					delta_p0 += stiffness * delta_lambda * w[idx0] * grad0;
					delta_p1 += stiffness * delta_lambda * w[idx1] * grad1;
					delta_p2 += stiffness * delta_lambda * w[idx2] * grad2;
					delta_p3 += stiffness * delta_lambda * w[idx3] * grad3;

					/*Real length_delta_p0 = std::sqrt(delta_p0.x * delta_p0.x + delta_p0.y * delta_p0.y + delta_p0.z * delta_p0.z);
					Real length_delta_p1 = std::sqrt(delta_p1.x * delta_p1.x + delta_p1.y * delta_p1.y + delta_p1.z * delta_p1.z);
					Real length_delta_p2 = std::sqrt(delta_p2.x * delta_p2.x + delta_p2.y * delta_p2.y + delta_p2.z * delta_p2.z);
					Real length_delta_p3 = std::sqrt(delta_p3.x * delta_p3.x + delta_p3.y * delta_p3.y + delta_p3.z * delta_p3.z);

					delta_dis[idx] = (length_delta_p0 + length_delta_p1 + length_delta_p2 + length_delta_p3) / 4.0f;*/

					Real nu = 0.45f;
					Real E = 10000000.0f;

					Real mu = E / (2.0f * (1.0f + nu));
					Real lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
					Real elastic_energy_devia = 0.5 * mu * c_devia;
					Real elastic_energy_volumetric = 0.5 * lambda * c_hyd * c_hyd; 
					delta_dis[idx] = elastic_energy_devia + elastic_energy_volumetric;

					if (w[idx0] != 0.0) {
						atomicAdd(&pos[3 * idx0 + 0], delta_p0.x);
						atomicAdd(&pos[3 * idx0 + 1], delta_p0.y);
						atomicAdd(&pos[3 * idx0 + 2], delta_p0.z);
					}
					if (w[idx1] != 0.0) {
						atomicAdd(&pos[3 * idx1 + 0], delta_p1.x);
						atomicAdd(&pos[3 * idx1 + 1], delta_p1.y);
						atomicAdd(&pos[3 * idx1 + 2], delta_p1.z);

					}
					if (w[idx2] != 0.0) {
						atomicAdd(&pos[3 * idx2 + 0], delta_p2.x);
						atomicAdd(&pos[3 * idx2 + 1], delta_p2.y);
						atomicAdd(&pos[3 * idx2 + 2], delta_p2.z);
					}
					if (w[idx3] != 0.0) {
						atomicAdd(&pos[3 * idx3 + 0], delta_p3.x);
						atomicAdd(&pos[3 * idx3 + 1], delta_p3.y);
						atomicAdd(&pos[3 * idx3 + 2], delta_p3.z);
					}
				}
			}
		}
	}

	__global__ void collision_response_kernel(int num_vertices, Real* w,
		Real* pos, const Real* const boundary) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < num_vertices) {
			if (w[idx] == 0.0)
				return;
			for (int j = 0; j < 3; j++) {
				if (pos[3 * idx + j] >= boundary[j])
					pos[3 * idx + j] = boundary[j];
				if(pos[3 * idx + j] <= boundary[j+3])
					pos[3 * idx + j] = boundary[j+3];
			}
		}
	}

	__global__ void update_vel_kernel(int num_vertices, Real h, Real* pos,
		Real* old_pos, Real* vel, Real* w) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < num_vertices) {
			if (w[idx] != 0.0) {
				for (int i = 0; i < 3; i++)
					vel[3 * idx + i] = (pos[3 * idx + i] - old_pos[3 * idx + i]) / h;
			}
		}
	}

	int num_color = 0;
	int cluster_num = 0;
	int max_clusterNum = 0;



	int mode = 2; // 1: non_cluster 2: cluster

	int itecount = 0;

	void Softbody_cuda::update(Real h, int maxIte, const std::vector<float>& boundary, const Vec3f& moveSphere) {
		auto nv = num_vertices_;
		auto nt = num_tets_;
		auto ne = num_edges_;
		// map OpenGL buffer object for writing from CUDA
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
			(void**)&d_pos, &num_bytes, cuda_vbo_resource));
		// XPBD simulation loop
		int nthreads = 256;
		int nblocks = (nv + nthreads - 1) / nthreads;
		//int ndcblocks = (ne + nthreads - 1) / nthreads;
		int ntcblocks = (nt + nthreads - 1) / nthreads;
		int ncblocks = (max_clusterNum + nthreads - 1) / nthreads;
		checkCudaErrors(cudaMemcpy(d_old_pos, d_pos, sizeof(Real) * 3 * nv, cudaMemcpyDeviceToDevice));
		semi_euler_kernel << <nblocks, nthreads >> > (nv, d_w, h, d_pos, d_vels, damping_factor_);
		//checkCudaErrors(cudaMemcpy(d_predict_pos, d_pos, sizeof(Real) * 3 * nv, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemset(d_lambda, 0, sizeof(Real) * (ne + nt)));

		//printf("num_tets:%d\n", nt);

		
		
		//printf("maxIter:%d\n", maxIte);
		
		itecount++;
		
		std::ofstream outfile("dis.txt", std::ios::app);
		for (int i = 0; i < maxIte; i++) {
			//solve_distance_constraint_kernel << <ndcblocks, nthreads >> > (
			//	ne, d_w, d_predict_pos, d_pos, d_edges, d_rest_length, 0, d_alpha,
			//	d_lambda);
			//solve_volume_constraint_kernel << <ntcblocks, nthreads >> > (
			//	nt, d_w, d_predict_pos, d_pos, d_tets, d_rest_volume, ne, d_alpha,
			//	d_lambda);

			//printf("Ite:%d\n", i);
			for (int j = 0; j < num_color; j++){
				//int ntcblocks_color = (color_tet_num[j] + nthreads - 1) / nthreads;
				//printf("j:%d\n", j);


					solve_neohookean_constraint_kernel << <ncblocks, nthreads >> > (
					max_clusterNum, d_w, /*d_predict_pos,*/ d_pos, d_tets, d_dm_inv, d_alpha_hydrostatic_,
					d_alpha_deviatoric_
					, j, d_clusterIndex, d_cluster_vertex, d_colorIndex, d_color_cluster,
					mode, d_delta_dis
					);
				//printf("j:%d over\n", j);
				//printf("first:%d\n", first);
				/*checkCudaErrors(cudaMemcpy(d_predict_pos, d_pos, sizeof(Real) * 3 * nv,
					cudaMemcpyDeviceToDevice));*/
			}
			
			if (itecount * maxIte == 6500) {
				Real* delta_dis = new Real[nt];
				checkCudaErrors(cudaMemcpy(delta_dis, d_delta_dis, sizeof(Real) * nt, cudaMemcpyDeviceToHost));
				Real avg_dis = 0.0;
				for (int j = 0; j < nt; j++){
					avg_dis += abs(delta_dis[j]);
				}
				
				outfile << avg_dis << std::endl;
				free(delta_dis);
				//printf("avg_dis:%f\n", avg_dis);
			}
			
			
		}
		checkCudaErrors(cudaMemcpy(d_boundary, boundary.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice));
		collision_response_kernel << <nblocks, nthreads >> > (nv, d_w, d_pos,d_boundary);
		update_vel_kernel << <nblocks, nthreads >> > (nv, h, d_pos, d_old_pos, d_vels, d_w);

		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void Softbody_cuda::init_opengl_cuda_interop(int VBO) {
		// init d_pos with OpenGL vertex buffer
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO,
			cudaGraphicsMapFlagsWriteDiscard));
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
			(void**)&d_pos, &num_bytes, cuda_vbo_resource));

		init_gpu_data();
		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void Softbody_cuda::init_gpu_data() {
		checkCudaErrors(cudaMalloc((void**)&d_old_pos, sizeof(Real) * 3 * num_vertices_));
		//checkCudaErrors(cudaMalloc((void**)&d_predict_pos, sizeof(Real) * 3 * num_vertices_));
		checkCudaErrors(cudaMalloc((void**)&d_vels, sizeof(Real) * 3 * num_vertices_));
		checkCudaErrors(cudaMalloc((void**)&d_w, sizeof(Real) * num_vertices_));
		checkCudaErrors(cudaMalloc((void**)&d_edges, sizeof(int) * 2 * num_edges_));
		checkCudaErrors(cudaMalloc((void**)&d_tets, sizeof(int) * 4 * num_tets_));
		checkCudaErrors(cudaMalloc((void**)&d_rest_length, sizeof(Real) * num_edges_));
		checkCudaErrors(cudaMalloc((void**)&d_rest_volume, sizeof(Real) * num_tets_));
		checkCudaErrors(cudaMalloc((void**)&d_lambda, sizeof(Real) * (num_edges_ + num_tets_)));
		checkCudaErrors(cudaMalloc((void**)&d_alpha, sizeof(Real) * (num_edges_ + num_tets_)));
		checkCudaErrors(cudaMalloc((void**)&d_alpha_hydrostatic_, sizeof(Real) * num_tets_));
		checkCudaErrors(cudaMalloc((void**)&d_alpha_deviatoric_, sizeof(Real) * num_tets_));
		checkCudaErrors(cudaMalloc((void**)&d_dm_inv, sizeof(Mat3f) * num_tets_));
		checkCudaErrors(cudaMalloc((void**)&d_boundary, sizeof(Real) * 6));
		//checkCudaErrors(cudaMalloc((void**)&d_colors, 20000 * sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_clusterIndex, 20000 * sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_cluster_vertex, 20000 * sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_colorIndex, 20000 * sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_color_cluster, 20000 * sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_delta_dis, sizeof(Real) * num_tets_));


		checkCudaErrors(cudaMemcpy(d_old_pos, old_pos_.data(), sizeof(Real) * 3 * num_vertices_, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(d_predict_pos, pos_.data(), sizeof(Real) * 3 * num_vertices_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_vels, vel_.data(), sizeof(Real) * 3 * num_vertices_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_w, w_.data(), sizeof(Real) * num_vertices_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_edges, edges_.data(), sizeof(int) * 2 * num_edges_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_tets, tets_.data(), sizeof(int) * 4 * num_tets_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_rest_length, rest_length_.data(), sizeof(Real) * num_edges_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_rest_volume, rest_volume_.data(), sizeof(Real) * num_tets_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_lambda, 0.0, sizeof(Real) * (num_edges_ + num_tets_)));
		checkCudaErrors(cudaMemcpy(d_alpha, alpha_.data(), sizeof(Real) * (num_edges_ + num_tets_), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_alpha_hydrostatic_, alpha_hydrostatic_.data(), sizeof(Real) * num_tets_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_alpha_deviatoric_, alpha_deviatoric_.data(), sizeof(Real) * num_tets_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_dm_inv, dm_inv_.data(), sizeof(Mat3f) * num_tets_, cudaMemcpyHostToDevice));


		std::ifstream infile("D:\\LearnOpenGL\\XPBD_cluster\\XPBD\\color_liver_cluster.txt");
		std::string line;
		int num = -1;


		size_t size = 20000 * sizeof(int);
		int colors[20000] = {0};
		int clusterIndex[20000] = {0};  //每个cluster第一个点的索引
		int cluster_vertex[20000] = {0}; //所有点，按聚类排序

		int colorIndex[20000] = {0}; //每个颜色第一个聚类的索引
		int color_cluster_withoutcolor[20000] = {0}; //只存储聚类序号

		struct color2cluster{
			int color;
			int cluster;
		};
		std::vector<color2cluster> color_cluster; //所有聚类，按颜色排序
		
		if (mode == 1) {
			std::string content;
			while (std::getline(infile, content)) {
				if (num == -1){
					num_color = std::stoi(content);
					std::cout << "num_color: " << num_color << std::endl;
				}
				else{
					std::istringstream iss(content);
					int color;
					iss >> color;
					colors[num] = color;
					color2cluster temp;
					temp.color = color;
					temp.cluster = num;
					color_cluster.push_back(temp);
				}
				num++;
			}
			sort(color_cluster.begin(), color_cluster.end(), [](color2cluster a, color2cluster b) { return a.color < b.color; }); //节点按颜色排序
			int clusterNum_temp = 0;
			for (int i = 0; i < color_cluster.size(); i++){
				if (colorIndex[color_cluster[i].color] == 0){
					colorIndex[color_cluster[i].color] = i;
					clusterNum_temp = 1;
				}
				else {
					clusterNum_temp++;
					if (clusterNum_temp > max_clusterNum){
						max_clusterNum = clusterNum_temp;
					}
				}
				color_cluster_withoutcolor[i] = color_cluster[i].cluster;
			}
			colorIndex[num_color] = cluster_num + 1;
			cluster_num++;
			printf("num_color %d\n", num_color);
			printf("max_clusterNum %d\n", max_clusterNum);

			cudaError_t status = cudaMemcpy(d_color_cluster, color_cluster_withoutcolor, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				printf("d_color_cluster copy error!\n");
				// 处理错误情况，例如退出程序
			}
			status = cudaMemcpy(d_colorIndex, colorIndex, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				printf("d_colorIndex copy error!\n");
				// 处理错误情况，例如退出程序
			}

		}
		else if (mode == 2) {
			std::string content;
			while (std::getline(infile, content)) {
				if (num == -1){
					num_color = std::stoi(content);
					std::cout << "num_color: " << num_color << std::endl;
				}
				else{
					std::istringstream iss(content);
					int index, cluster, color;
					iss >> index >> cluster >> color;
					//std::cout << index << " " << cluster << " " << color << std::endl;
					colors[cluster] = color;
					if (clusterIndex[cluster] == 0){
						clusterIndex[cluster] = num;
						color2cluster temp;
						temp.color = color;
						temp.cluster = cluster;
						color_cluster.push_back(temp);
					}
					cluster_vertex[num] = index;

					if (cluster > cluster_num){
						cluster_num = cluster;
					}
				}
				num++;
			}
			sort(color_cluster.begin(), color_cluster.end(), [](color2cluster a, color2cluster b) { return a.color < b.color; }); //聚类按颜色排序
			
			
			
			int clusterNum_temp = 0;
			for (int i = 0; i < color_cluster.size(); i++){
				if (colorIndex[color_cluster[i].color] == 0){
					colorIndex[color_cluster[i].color] = i;
					clusterNum_temp = 1;
				}
				else {
					clusterNum_temp++;
					if (clusterNum_temp > max_clusterNum){
						max_clusterNum = clusterNum_temp;
					}
				}
				color_cluster_withoutcolor[i] = color_cluster[i].cluster;
			}

			colorIndex[num_color] = cluster_num + 1;
			cluster_num++;
			clusterIndex[cluster_num] = num;
			printf("num_color %d\n", num_color);
			printf("max_clusterNum %d\n", max_clusterNum);
			
			cudaError_t status = status = cudaMemcpy(d_clusterIndex, clusterIndex, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				// 处理错误情况，例如退出程序
			}
			status = cudaMemcpy(d_cluster_vertex, cluster_vertex, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				// 处理错误情况，例如退出程序
			}
			/*
			cudaMemcpy(d_colors, colors, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				// 处理错误情况，例如退出程序
			}
			*/
		
			status = cudaMemcpy(d_color_cluster, color_cluster_withoutcolor, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				printf("d_color_cluster copy error!\n");
				// 处理错误情况，例如退出程序
			}
			status = cudaMemcpy(d_colorIndex, colorIndex, size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				printf("d_colorIndex copy error!\n");
				// 处理错误情况，例如退出程序
			}
			status = cudaMemset(d_delta_dis, 0, sizeof(Real) * num_tets_);
			if (status != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
				printf("d_delta_dis memset error!\n");
				// 处理错误情况，例如退出程序
			}
		}

		
	}
	Softbody_cuda::~Softbody_cuda() {
		//if (d_pos != nullptr)
		//	checkCudaErrors(cudaFree(d_pos));
		if (d_old_pos != nullptr)
			checkCudaErrors(cudaFree(d_old_pos));
		/*if (d_predict_pos != nullptr)
			checkCudaErrors(cudaFree(d_predict_pos));*/
		if (d_vels != nullptr)
			checkCudaErrors(cudaFree(d_vels));
		if (d_w != nullptr)
			checkCudaErrors(cudaFree(d_w));
		if (d_edges != nullptr)
			checkCudaErrors(cudaFree(d_edges));
		if (d_tets != nullptr)
			checkCudaErrors(cudaFree(d_tets));
		if (d_rest_length != nullptr)
			checkCudaErrors(cudaFree(d_rest_length));
		if (d_rest_volume != nullptr)
			checkCudaErrors(cudaFree(d_rest_volume));
		if (d_lambda != nullptr)
			checkCudaErrors(cudaFree(d_lambda));
		if (d_alpha != nullptr)
			checkCudaErrors(cudaFree(d_alpha));
		if (d_dm_inv != nullptr)
			checkCudaErrors(cudaFree(d_dm_inv));
		/*if (d_colors != nullptr)
			checkCudaErrors(cudaFree(d_colors));*/
		if (d_clusterIndex != nullptr)
			checkCudaErrors(cudaFree(d_clusterIndex));
		if (d_cluster_vertex != nullptr)
			checkCudaErrors(cudaFree(d_cluster_vertex));
		if (d_colorIndex != nullptr)
			checkCudaErrors(cudaFree(d_colorIndex));
		if (d_color_cluster != nullptr)
			checkCudaErrors(cudaFree(d_color_cluster));
		if (d_delta_dis != nullptr)
			checkCudaErrors(cudaFree(d_delta_dis));
		
	}
} // namespace XPBD
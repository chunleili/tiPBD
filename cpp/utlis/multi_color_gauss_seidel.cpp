/*
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


/*
NOTE:
The most important part of the code are the lines 105 to 121. Here the Gauss-Seidel method is implemented.
*/

#include <set>
#include <vector>

const int N = 100;
const float SHRINKING_FACTOR = 7.5f;
const int NO_PROGRESS_STREAK_THRESHOLD = 100;
const float EPS = 0.00001f;
typedef std::vector<int> Partition;

// vector of dimension Nx1
class Vec {
public:
	float v[N];

	Vec() {
		for (int i = 0; i < N; ++i) {
			v[i] = 0.0f;
		}
	}

	void print(char* s) {
		printf("%s ", s);
		for (int i = 0; i < N; ++i) {
			printf("%f, ", v[i]);
		}
		printf("\n");
	}
};

// matrix of dimension NxN.
class Mat {
public:
	float m[N][N];

	Mat() {
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				m[i][j] = 0.0f;
			}
		}
	}

	void print(char* s) {
		printf("%s ", s);
		for (int i = 0; i < N; ++i) {

			for (int j = 0; j < N; ++j) {
				printf("%f, ", m[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	Vec mult(const Vec& v) const {
		Vec r;

		for (int row = 0; row < N; ++row) {
			for (int col = 0; col < N; ++col) {
				r.v[row] += this->m[row][col] * v.v[col];
			}
		}
		return r;
	}

	static int gauss_seidel(Vec& x, const Vec& b, const Mat& m, float tol, int maxiter, const std::vector<Partition>& partitions) {

		int iter;
		for (iter = 0; iter < maxiter; ++iter) {
			for (Partition partition : partitions) {
				// we do a gauss-seidel step for this partition.
				// every partition stores a set of variables that will be solved for.
				// and these variables can be solved for independently of each other.
				// thus, the below loop can easily be parallelized.
				// note that this code is very similar to the Gauss-Seidel method implemented
				// in the previous article. It's just that the variables are solved for in a different order.
				for (int variable : partition) {
					float s = 0.0f;
					for (int j = 0; j < N; ++j) {
						if (j != variable) {
							s += m.m[variable][j] * x.v[j];
						}
					}
					x.v[variable] = (1.0f / m.m[variable][variable]) * (b.v[variable] - s);
				}
			}

			Vec mx = m.mult(x);

			float norm = 0.0f;
			for (int i = 0; i < N; ++i) {
				float a = mx.v[i] - b.v[i];
				norm += a*a;
			}
			norm = sqrt(norm);

			if (norm < tol) {
				break;
			}
		}

		return iter;
	}
};

std::vector<Partition> randomized_graph_coloring(const Mat& m) {
	std::set<int> neighbours[N];

	int node_colors[N]; // colors assigned to the nodes.
	int next_color[N]; // next color of every node, in case the palette runs out.				 
	std::set<int> node_palettes[N]; // palettes of the nodes.
	std::set<int> U;

	/*
	Every node needs to know about it's neighbours. so find that.
	There is an edge between two nodes i and j, if the matrix coefficient at row i, column j is non-zero.
	*/
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (i != j && fabs(m.m[i][j]) > EPS) {

				// if necessary, make j a neighbour of i.
				if (neighbours[i].find(j) == neighbours[i].end()) {
					neighbours[i].insert(j);
				}

				// if necessary, make i a neighbour of j.
				if (neighbours[j].find(i) == neighbours[j].end()) {
					neighbours[j].insert(i);
				}
			}
		}
	}

	// calculate max degree of a single node.
	int delta_v = 0;
	for (int i = 0; i < N; ++i) {
		if ((int)neighbours[i].size() > delta_v) {
			delta_v = neighbours[i].size();
		}
	}

	// initially, every node has a palette of size delta_v/shrinking_factor.
	// the maximum number of colors necessary for a graph coloring is delta_v, but many
	// graphs won't need that many colors. therefore, we choose to shrink delta_v by a shrinking factor.
	// if the shrinking factor is too big, so that the problem is unsolvable, then more colors will be added on the fly.
	int max_color = int((float)delta_v / SHRINKING_FACTOR);
	if (max_color <= 0) {
		max_color = 1;
	}
	max_color = 2;

	// initialize the palettes for all the node.
	// the colors in the palette will be chosen randomly from, for all the remaining nodes in U.
	for (int iv = 0; iv < N; ++iv) {
		for (int ic = 0; ic < max_color; ++ic) {
			node_palettes[iv].insert(ic);
		}
		next_color[iv] = max_color;
	}

	for (int iv = 0; iv < N; ++iv) {
		U.insert(iv);
	}

	// keep track of the number of iterations with no progress.
	int no_progress_streak = 0;

	/*
	If a node has found a color that solves the graph coloring for that node, then remove from U.
	Once U is empty, the graph coloring problem is done.
	*/
	while (U.size()) {

		// all remaining nodes in U are given a random color.
		for (int iv : U) {
			// get random color from palette, and assign it.
			int m = rand() % node_palettes[iv].size();
			auto setIt = node_palettes[iv].begin();
			advance(setIt, m);

			node_colors[iv] = *setIt;
		}

		std::set<int> temp;


		/*
		  Now let's find all the nodes whose colors are different from all their neighbours.
		  Those nodes will be removed from U, because they are done, with respect to the graph coloring problem.
		*/
		for (int iv : U) {

			int icolor = node_colors[iv];

			/*
			Check if graph coloring property is solved for node.
			*/
			bool different_from_neighbours = true;
			for (int neighbour : neighbours[iv]) {

				if (node_colors[neighbour] == icolor) {
					different_from_neighbours = false;
					break;
				}
			}

			if (different_from_neighbours) {
				// found the right color for this one.
				// so remove from U.

				// also, the neighbours of iv can't use this color anymore.
				// so remove it from their palettes.
				for (int neighbour : neighbours[iv]) {
					node_palettes[neighbour].erase(icolor);
				}

			}
			else {
				// not a correct color. don't remove from U.
				temp.insert(iv);
			}

			// feed the hungry!
			// if palette empty, we add more colors on the fly.
			// if we don't do this, the algorithm will get stuck in a loop.
			if (node_palettes[iv].empty()) {
				node_palettes[iv].insert(next_color[iv]++);
			}

		}

		if (U.size() == temp.size()) {
			no_progress_streak++;

			// if no progress for too many iterations, we have no choice but to feed a random node.
			if (no_progress_streak > NO_PROGRESS_STREAK_THRESHOLD) {
				int m = rand() % U.size();
				auto setIt = U.begin();
				advance(setIt, m);

				node_palettes[*setIt].insert(next_color[*setIt]++);

				no_progress_streak = 0;
			}
		}

		U = temp;
	}

	// find the number of colors used in our solution.
	// this is also the number of partitions.
	int num_colors = 0;
	for (int i = 0; i < N; ++i) {
		if (next_color[i] > num_colors) {
			num_colors = next_color[i];
		}
	}

	/*
	Finally, we collect all the partitions then.
	*/
	std::vector<Partition> partitions;
	for (int ic = 0; ic < num_colors; ++ic) {
		Partition partition;

		/*
		The first partition is all nodes that use color 0,
		the second partition use color 1, and so on.
		*/
		for (int inode = 0; inode < N; ++inode) {
			if (node_colors[inode] == ic) {
				partition.push_back(inode);
			}
		}

		partitions.push_back(partition);
	}

	return partitions;
}


Mat get_diagonally_dominant_matrix() {
	Mat m;

	/*
	generate random matrix entries
	*/
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			if (rand() % 9 == 0) {
				m.m[i][j] = float(rand() % 10);
				m.m[j][i] = m.m[i][j];
			}
		}
	}

	/*
	Now we iterate over every row, and modify the matrix to make sure it's diagonally dominant.
	*/
	for (int i = 0; i < N; ++i) {
		float diag = fabs(m.m[i][i]);
		float row_sum = 0.0f;

		for (int j = 0; j < N; ++j) {
			if (i != j) {
				row_sum += fabs(m.m[i][j]);
			}
		}

		/*
		Not diagonally dominant. So increase the diagonal value to fix that.
		*/
		if (!(diag >= row_sum)) {
			m.m[i][i] += (row_sum - diag);
		}

		if (fabs(m.m[i][i]) < EPS) {
			m.m[i][i] += 1.0f;
		}
	}

	return m;
}

int main() {
	srand(13000);

	Mat m = get_diagonally_dominant_matrix();

	int nonZeros = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			if (fabs(m.m[i][j]) > EPS) {
				nonZeros++;
			}
		}
	}
	printf("percent of non-zeros of M: %d%%\n", int(100.0f * float(nonZeros) / float(N*N)));

	Vec expected_solution;
	for (int i = 0; i < N; ++i) {
		expected_solution.v[i] = 8.0f * float(rand() % 100) / 100.0f - 4.0f;
	}

	Vec b = m.mult(expected_solution);

	/*
	With that, we have generated a linear system
	M*x = b.
	Now let's solve it!
	*/

	Vec x;
	for (int i = 0; i < N; ++i) {
		x.v[i] = 0.0f;
	}

	printf("solving linear system where N = %d\n\n\n", N);

	expected_solution.print("expected solution:\n");
	printf("\n");

	// graph coloring to partition the problem. 
	std::vector<Partition> partitions = randomized_graph_coloring(m);

	int iter = Mat::gauss_seidel(x, b, m, 0.001f, 10000, partitions);
	printf("number of partitions: %d\n", partitions.size());
	x.print("gauss-seidel method solution:\n");
	printf("number of iterations: %d\n", iter);

	system("pause");
}
#include <tuple>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/multi_array.hpp>
#include <omp.h>
#include "misc.hpp"
#include "mpienv.hpp"

using namespace std;

//explicit vectorization of 3D coords (1 slot wasted)
typedef float float4 [[gnu::vector_size(16)]];

/*
 * Work on a single tile of the NxN matrix of the points. Tuning is necessary to have this function work on caches.
 * The tile may be on the diagonal, hence forces and energies are double counted but halved before returning.
 * This case scales ~N and ~1/subdivisions^2, so it shouldn't be a concern.
 */

using pointview = boost::multi_array<float4, 1>::const_array_view<1>::type;
using forcesview = boost::multi_array<float4, 1>::array_view<1>::type;
[[gnu::target_clones("default","arch=ivybridge","arch=haswell","arch=broadwell")]]
float process_block(pointview p1s, forcesview f1s, pointview p2s, forcesview f2s, float cut2, bool diagonal) {
	auto n1 = p1s.size(), n2 = p2s.size();
	alignas(32) float rsqrts[n1][n2];
	//calculate squared euclidean distances, don't store them as it's supposedly a fast op and it would take another
	//NxN matrix of float4.
	loopi(n1) {
		auto p1 = p1s[i];
		auto rsqrts_base = rsqrts[i];
		loopj(n2) {
			auto d = p2s[j] - p1;
			d *= d;
			rsqrts_base[j] = d[0] + d[1] + d[2];
		}
	}
	float energy = 0, normalize = diagonal ? 0.5 : 1;
	//calculate square root, energy, then 1/d^3
	#pragma omp simd
	for(size_t i = 0; i < n1 * n2; i++) {
		auto& rsqrt = rsqrts[0][i];
		rsqrt = rsqrt == 0 || rsqrt > cut2 ? 0.f : 1.f / sqrtf(rsqrt);   //this will be vectorized with rsqrtps
		energy += rsqrt;
		rsqrt *= rsqrt * rsqrt * normalize;
	}
	/*
	 * Accumulate the force for each single particle to minimize atomic ops. The first loop
	 * has no strides while the other does, but everything should already be in the cache.
	 */
	loopi(n1) {
		float4 forcei{ 0, 0, 0, 0 };
		auto p1 = p1s[i];
		auto rsqrts_base = rsqrts[i];
		loopj(n2){
			auto forcej = (p1 - p2s[j]) * rsqrts_base[j];
			forcei -= forcej;
			f2s[j] += forcej;
		}
		f1s[i] += forcei;
	}
	if(diagonal) energy /= 2;
	return energy;
}

int nbody_main(int argc, char** argv){

	float cut2;
	uint32_t tile;
	{
		using namespace boost::program_options;
		options_description options("Options for nbody");
		options.add_options()
				("help", "show help")
				("cut", value(&cut2)->default_value(sqrt(1000.f)), "max distance")
				("tile", value(&tile)->required(), "tile side length");
		variables_map vm;
		store(parse_command_line(argc, argv, options), vm);
		notify(vm);
		if (vm.count("help")) {
			cerr << options << endl;
			return 0;
		}
		if (cut2 < 0 || !tile) throw invalid_argument("cut and sub must be > 0");
		cut2 *= cut2;
	}


	using namespace boost;
	multi_array<float4, 1> ps;
	{
		vector<float4> ps_tmp;  //this has push_back, resizing a multi_array is stupidly slow
		float4 r{ 0, 0, 0, 0 };
		while (cin.read((char*)&r, 3 * sizeof(float)) && cin.gcount() == 3 * sizeof(float))
			ps_tmp.push_back(r);
		ps.resize(extents[ps_tmp.size()]);
		ps.assign(ps_tmp.begin(), ps_tmp.end());
	}
	uint32_t n = ps.size();
	if(!n) throw runtime_error("no points found in input");
	ps.resize(extents[n]);
	float energy = 0;

	/*
	 * Domain is triangular, but to use collapse(2) indices must be independent:
	 * cut the triangle in half in one dimension, flip the smallest area in the x and y directions,
	 * and place it next to the bottom part to obtain a rectangle.
	 */
	auto subdivisions = n / tile + !!(n % tile), len_last = n % tile ?: tile;
	if(subdivisions % 2) subdivisions--, len_last += tile;
	if(!subdivisions) throw runtime_error("tile is too big for input size");
	auto slice_indices = [=](auto tileidx) {
		return multi_array_types::index_range(tileidx * tile, tileidx * tile + (tileidx == subdivisions - 1 ? len_last : tile));
	};
	vector<multi_array<float4, 1>> forces_threads(omp_get_max_threads(), [n]{
		multi_array<float4, 1> ret(extents[n]);
		memset(ret.data(), 0, sizeof(float4) * n);
		return ret;
	}());
	measure_runtime intv;
	#pragma omp parallel for reduction(+:energy) collapse(2)
	for(uint32_t i = 0; i <= subdivisions; i++)
		for(uint32_t j = 0; j < subdivisions / 2; j++) {
			auto x = i + j, y = j;
			if(x >= subdivisions) y = subdivisions - 1 - j, x = 2 * subdivisions - 1 - x;
			auto idx_x = slice_indices(x), idx_y = slice_indices(y);
			auto& forces = forces_threads[omp_get_thread_num()];
			energy += process_block(ps[indices[idx_x]], forces[indices[idx_x]],
					ps[indices[idx_y]], forces[indices[idx_y]], cut2, x == y);
		}
	multi_array<float, 2> forces(extents[n][3]);
	#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < n; i++){
		float4 forcesum{ 0, 0, 0, 0 };
		for(auto& forces: forces_threads) forcesum += forces[i];
		loopk(3) forces[i][k] = forcesum[k];
	}
	auto elapsed = intv.elapsed_msec();
	cerr<<"energy "<<energy<<" time "<<elapsed<<"ms"<<endl;
	cout.write((char*)forces.data(), forces.num_elements() * sizeof(float));
	return 0;
}

ginit = []{
	programs["nbody"] = nbody_main;
};

#include <iostream>
#include <tuple>
#include <chrono>
#include <boost/program_options.hpp>
#include "misc.hpp"
#include "mpienv.hpp"

using namespace std;

//archs: all x86_64, my laptop, Galileo (haswell/broadwell)
[[gnu::target_clones("default","arch=ivybridge","arch=haswell","arch=broadwell")]]
auto add_arrays(float* a, const float* __restrict__ b, size_t n, double sum_max){
	double sum = 0;
	for(size_t i_base = 0; i_base < n;) {
		//speculatively try to calculate everything vectorized in bunches of bunch_len
		constexpr const auto bunch_len = 4096;      //this is arbitrary
		double sum_bunch = 0;
		size_t max_i = min(n, i_base + bunch_len);
		//#pragma omp simd is not necessary for gcc >= 6
		#pragma omp simd
		for(size_t i = i_base; i < max_i; i++){
			sum_bunch += b[i];
			a[i] += expf(b[i]);     //generates call to libmvec with -Ofast
		}
		if(sum + sum_bunch <= sum_max){
			sum += sum_bunch;
			i_base = max_i;
			continue;
		}
		//calc sum without vectorization
		size_t span = 0;
		for(; i_base + span < n && span < bunch_len;){
			sum += b[i_base + span++];
			if(sum > sum_max) break;
		}
		//rollback a[i] += expf(b[i])
		#pragma omp simd
		for(size_t i = i_base + span; i < max_i; i++) a[i] -= expf(b[i]);
		return make_tuple(sum, i_base + span);
	}
	return make_tuple(sum, n);
}

int add_vector_main(int argc, char** argv){

	uint32_t len;
	double sum_max;
	{
		using namespace boost::program_options;
		options_description options("Options for mpibologna");
		options.add_options()
				("help", "show help")
				("len", value(&len)->default_value(100000000), "vectors length")
				("sum_max", value(&sum_max), "sum max (default half of len)");
		variables_map vm;
		store(parse_command_line(argc, argv, options), vm);
		notify(vm);
		if (vm.count("help")) {
			cerr << options << endl;
			return 0;
		}
		if (!len) throw invalid_argument("len and bunch_size must be > 0");
		if (!vm.count("sum_max")) sum_max = len / 2;
	}

	vector<float> a(len), b(len);
	if(!cin.read((char*)a.data(), len * sizeof(float)) || !cin.read((char*)b.data(), len * sizeof(float))) throw runtime_error("Cannot read vectors");
	auto start = chrono::steady_clock::now();
	auto [sum, added_elements] = add_arrays(a.data(), b.data(), len, sum_max);
	auto elapsed = chrono::steady_clock::now() - start;
	cout<<"sum(b) "<<sum<<" for N elements "<<added_elements<<" in "<<chrono::duration_cast<chrono::milliseconds>(elapsed).count()<<"ms"<<endl;

	return 0;

}

ginit = []{
	programs["add_vector"] = add_vector_main;
};

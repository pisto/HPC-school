#include <iostream>
#include <stdexcept>
#include <sstream>
#include <random>
#include <vector>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <boost/core/demangle.hpp>
#include <boost/program_options.hpp>
#include "mpienv.hpp"
#include "misc.hpp"

using namespace std;
namespace mpi = boost::mpi;

const mpi::environment mpienv [[gnu::init_priority(101)]] (mpi::threading::multiple, false);
const mpi::communicator mpiworld;
const int mpiworld_coord = mpiworld.rank();

//archs: all x86_64, modern CPUs, my laptop, Galileo
[[gnu::target_clones("default","avx","arch=ivybridge","arch=haswell")]]
auto add_arrays(float* a, const float* __restrict__ b, size_t n, double sum_max){
	double sum = 0.0;
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

int main(int argc, char** argv) try {

	uint32_t len;
	double sum_max;
	try {
		using namespace boost::program_options;
		options_description options("Options for mpibologna");
		options.add_options()
			("help", "show help")
			("len", value(&len)->default_value(100000000), "vectors length")
			("sum_max", value(&sum_max), "sum max (default half of len)");
		variables_map vm;
		store(parse_command_line(argc, argv, options), vm);
		notify(vm);
		if(vm.count("help")){
			cerr<<options<<endl;
			return 0;
		}
		if(!len) throw invalid_argument("len and bunch_size must be > 0");
		if(!vm.count("sum_max")) sum_max = len / 2;
	} catch(const invalid_argument& e){
		cerr<<"Invalid argument found in command line: "<<e.what()<<endl;
		return 1;
	} catch(const boost::program_options::error& e){
		cerr<<"Failed to parse command line: "<<e.what()<<endl;
		return 1;
	}

	vector<float> a(len), b(len);
	if(!cin.read((char*)a.data(), len * sizeof(float)) || !cin.read((char*)b.data(), len * sizeof(float))) throw runtime_error("Cannot read vectors");
	auto start = chrono::steady_clock::now();
	auto [sum, added_elements] = add_arrays(a.data(), b.data(), len, sum_max);
	auto elapsed = chrono::steady_clock::now() - start;
	cout<<"sum(b) "<<sum<<" for N elements "<<added_elements<<" in "<<chrono::duration_cast<chrono::milliseconds>(elapsed).count()<<"ms"<<endl;

} catch(const exception& e) {
	ostringstream err;
	err<<"Uncaught exception of type "<<boost::core::demangled_name(typeid(e))<<": "<<e.what()<<endl;
	cerr<<err.str();
	return 1;
}

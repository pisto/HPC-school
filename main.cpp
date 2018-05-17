#include <iostream>
#include <sstream>
#include <tuple>
#include <chrono>
#include <boost/core/demangle.hpp>
#include <boost/program_options.hpp>
#include "misc.hpp"
#include "mpienv.hpp"

using namespace std;

const boost::mpi::environment mpienv [[gnu::init_priority(101)]] (boost::mpi::threading::multiple, false);
const boost::mpi::communicator mpiworld;
const int mpiworld_coord = mpiworld.rank();

std::map<std::string, std::function<int(int argc, char** argv)>> programs [[gnu::init_priority(101)]];

int main(int argc, char** argv) try {
	auto program = programs.find(argc > 1 ? argv[1] : "");
	if(program == programs.end()){
		cerr<<"please specific a valid program among:";
		for(auto kv: programs) cerr<<' '<<kv.first;
		cerr<<endl;
		return 1;
	}
	return program->second(argc - 1, argv + 1);
} catch(const exception& e) {
	ostringstream err;
	err<<"Uncaught exception of type "<<boost::core::demangled_name(typeid(e))<<": "<<e.what()<<endl;
	cerr<<err.str();
	return 1;
}

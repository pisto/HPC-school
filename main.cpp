#include <iostream>
#include <stdexcept>
#include <sstream>
#include <boost/core/demangle.hpp>
#include <boost/mpi.hpp>
#include "misc.hpp"

using namespace std;
namespace mpi = boost::mpi;

int main(int argc, char** argv) try {
	mpi::environment env(argc, argv, false);
	mpi::communicator world;
	if(world.size() < 2) throw runtime_error("mpi world size < 2");
	int me = world.rank(), other;
	world.sendrecv(me ^ 1, 0, me, me ^ 1, 0, other);
	cout<<"me "<<me<<" other "<<other<<endl;
} catch(const exception& e) {
	ostringstream err;
	err<<"Uncaught exception of type "<<boost::core::demangled_name(typeid(e))<<": "<<e.what()<<endl;
	cerr<<err.str();
	return 1;
}

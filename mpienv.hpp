#pragma once

#include <string>
#include <functional>
#include <map>
#include <boost/mpi.hpp>

extern const boost::mpi::environment mpienv;
extern const boost::mpi::communicator mpiworld;
extern const int mpiworld_coord;

extern std::map<std::string, std::function<int(int argc, char** argv)>> programs;
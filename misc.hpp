#pragma once

#include <utility>
#include <cstdint>

/*
 * Compile time constant debugging tools. Caught type/value shown in compiler error.
 */

#define catchtypeof(t) {catchtype<decltype(t)> _;}
#define catchvalueof(t) {catchvalue<decltype(t), t> _;}

template<typename T> struct catchtype;
template<typename T, T> struct catchvalue;

//pseudo get RIP

template<int = 1> [[gnu::noinline]] void* getRIP(){ return __builtin_return_address(0); };

//looping
#define loop(v, m)		for(size_t v = 0, m__ = size_t(m); v < m__; v++)
#define loopi(m)		loop(i, m)
#define loopj(m)		loop(j, m)
#define loopk(m)		loop(k, m)

/*
 * Fake destructor for C-like resources, to cope with exceptions. E.g.:
 * void* array = malloc(123);
 * destructor([=]{ free(array); });
 */

#define destructor(f) destructor_helper_macro_1(f, __LINE__)

template<typename F> struct destructor_helper{
        F f;
        ~destructor_helper(){ f(); }
};
template<typename F> destructor_helper<F> make_destructor_helper(F&& f){
        return destructor_helper<F>{std::move(f)};
}
#define destructor_helper_macro_2(f, l) auto destructor_ ## l = make_destructor_helper(f)
#define destructor_helper_macro_1(f, l) destructor_helper_macro_2(f, l)

/*
 * Global init:
 * ginit = some_function;
 */

#define ginit ginit_helper_macro_1(__LINE__)
struct ginit_helper {
	template<typename F> ginit_helper(F&& f){ f(); }
};
#define ginit_helper_macro_2(l) static ginit_helper ginit_ ## l __attribute__((used))
#define ginit_helper_macro_1(l) ginit_helper_macro_2(l)

/*
 * MPI errors to exceptions
 */

#include <string>
#include <stdexcept>
#include <mpi.h>

#define assertmpi >>assertmpi_helper{__FILE__ ":" + std::to_string(__LINE__)}

struct MPI_error : std::runtime_error {
	MPI_error(const std::string& place, const std::string& err): std::runtime_error("MPI @ " + place + ": " + err) {}
};

struct assertmpi_helper { std::string place; };
inline int operator>>(int ret, assertmpi_helper&& p){
	if(!ret) return 0;
	char buffer[MPI_MAX_ERROR_STRING];
	int resultlen = 0;
	throw MPI_error(p.place, MPI_Error_string(ret, buffer, &resultlen) ? "unknown error" : std::string(buffer, resultlen));
}

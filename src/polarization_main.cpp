#include <iostream>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <eigen3/Eigen/Dense>

#include "definitions.h"
#include "functions.h"
#include "nelder_mead.h"

using namespace Eigen;
using namespace std;

// floating point type
using scalar = boost::multiprecision::cpp_bin_float_double;
// using scalar = boost::multiprecision::mpfr_float_50;
// using scalar = boost::multiprecision::cpp_bin_float_100;

namespace Eigen {
template <>
struct NumTraits<scalar> : GenericNumTraits<scalar> {};
}  // namespace Eigen

int main() {
    const auto basis_s_file = "basis_1s.dat";
    const auto basis_p_file = "basis_2p.dat";

    ios::sync_with_stdio(false);
    cout << " Eigen is using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    const auto [en_s, basis_s] = load_basis<scalar>(basis_s_file);
    const auto [en_p, basis_p] = load_basis<scalar>(basis_p_file);
    const auto [h_p, n_p] = generate_matrices_P<scalar>(basis_p);
    
}
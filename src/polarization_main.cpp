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

    const scalar en_drake_s("-2.903724377034119598311e+00");
    constexpr int max_iters = 150;

    ios::sync_with_stdio(false);
    cout << " Eigen is using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    const auto [en_s, basis_s] = load_basis<scalar>(basis_s_file);
    const auto [en_p, basis_p] = load_basis<scalar>(basis_p_file);
    const auto [h_s, n_s]      = generate_matrices_S<scalar>(basis_s);
    const auto [h_p, n_p]      = generate_matrices_P<scalar>(basis_p);
    const auto j_ps            = generate_current_PS<scalar>(basis_p, basis_s);
    const auto [en_s_solv, wf_s] =
        solve_for_state<scalar>(h_s, n_s, en_drake_s - 1.0e-5, max_iters, 1.0e-45);

    if (en_s != en_s_solv) {
        cout << " Energies from file and from solver for S state does not match!\n"
             << " solv: " << en_s_solv << '\n'
             << " file: " << en_s << '\n'
             << flush;
    }

    constexpr double eps = 1.0e-8;

    auto j_times_s = (j_ps * wf_s).eval();

    auto res_denom = (h_p - en_s * n_p).eval();
    res_denom += res_denom.diagonal().asDiagonal() * eps;
    res_denom.ldlt().solveInPlace(j_times_s);

    const auto pol = j_times_s.squaredNorm() * 2.0 / 3.0;

    cout << pol << '\n';
}
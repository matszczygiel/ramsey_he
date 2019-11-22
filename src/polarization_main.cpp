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
//using scalar = boost::multiprecision::cpp_bin_float_oct;
// using scalar = boost::multiprecision::mpfr_float_50;
// using scalar = boost::multiprecision::cpp_bin_float_100;

namespace Eigen {
template <>
struct NumTraits<scalar> : GenericNumTraits<scalar> {};
}  // namespace Eigen

int main() {
    const auto basis_s_file = "/home/Mateusz/workspace/he_ramsey/he_cpp/build/basis_1s.dat";
    const auto basis_p_file = "/home/Mateusz/workspace/he_ramsey/he_cpp/build/basis_2p.dat";

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

    constexpr double eps = 0.0;

    Hamiltonian<scalar> res_denom = h_p - en_s * n_p;
    res_denom += res_denom.diagonal().asDiagonal() * eps;
    auto ham_dec = res_denom.ldlt();

    const auto j_times_s = j_ps * wf_s;
    const auto res       = ham_dec.solve(j_times_s);
    check_and_report_eigen_info(cout, ham_dec.info());

    const auto pol = j_times_s.dot(res) * 2.0 / 3.0;
    cout << pol << '\n';
}
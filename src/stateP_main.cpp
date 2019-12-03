#include <fstream>
#include <iostream>
#include <vector>

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
// using scalar = boost::multiprecision::cpp_bin_float_double;
using scalar = boost::multiprecision::cpp_bin_float_oct;
// using scalar = boost::multiprecision::mpfr_float_50;
// using scalar = boost::multiprecision::cpp_bin_float_100;

namespace Eigen {
template <>
struct NumTraits<scalar> : GenericNumTraits<scalar> {};
}  // namespace Eigen

int main() {
    constexpr int max_iterations = 150;
    const scalar en_drake("-2.123843086498093e+00");

    Eigen::initParallel();
    ios::sync_with_stdio(false);
    cout << " Eigen is using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    Matrix<scalar, m, 1> xv;

    constexpr int n = 1500;
    xv << scalar(
        "4.29927590856425569918372847074146525158949408479075518120877130749351418623e-01"),
        scalar("4.08503632152804427418599617191377513164455646862714024538850581826011184905e+00"),
        scalar("1.33542124657070805671370394534137291098770490513482417875459888902573057806e+00"),
        scalar("7.14909361426813808367854749283677284420944782049424675800341748184190731409e+00"),
        scalar("-4.31828803543303887898062820122054074512662937052946524135511146995646380001e-01"),
        scalar("9.08003759672963793434425751957930593519446645274106235118932299196239712637e-01"),
        scalar("2.79676356446687837525813425284254020771006908788543193396388424379236011987e-01"),
        scalar("1.51805299391235210018890773607293141194611253889732639244359418442467501124e+01"),
        scalar("1.57748059805413841900062004662999021309681737007375389219462305680124882032e+01"),
        scalar("5.68964669367956032364279931350603710270479119345989251634001650614726264527e+00"),
        scalar("1.36096461522299160122206874947267587829467553237811515106164609409823030982e+01"),
        scalar("6.49411232809759250112799817647083544616231275924585465050983075531354802489e+00"),
        scalar("1.53515823398863698886530585288412912069111909843998453613938854122839896133e+01"),
        scalar("3.62327596642792609394244941321783406058680271107240536817120038471225083231e+00");

    /*
        constexpr int n              = 100;
        xv << scalar("6.228738402342648417e-01"),
              scalar("4.087443137613022692e+00"),
              scalar("1.345890703982451697e+00"),
              scalar("7.161746316096892606e+00"),
              scalar("-6.344058652428644640e-01"),
              scalar("8.620005081121213664e-01"),
              scalar("5.763470111165558407e-01"),
              scalar("1.518652144831781925e+01"),
              scalar("1.577322895368443767e+01"),
              scalar("5.616341187484819031e+00"),
              scalar("1.351168971233441241e+01"),
              scalar("6.479918657111007008e+00"),
              scalar("1.536855051453726517e+01"),
              scalar("3.542781755070953942e+00");
    */

    const auto target = [&en_drake](const Matrix<scalar, m, 1>& x) {
        const scalar epsilon = 1.0e-45;
        scalar eig           = en_drake - 1.0e-5;
        const scalar eold    = eig;

        const auto phi = generate_basis(x, n);

        const auto [dh, dn] = generate_matrices_P(phi);

        Matrix<scalar, Dynamic, 1> v = Matrix<scalar, Dynamic, 1>::Ones(n);
        Matrix<scalar, Dynamic, 1> w = v;

        const auto ham_dec = (dh - eig * dn).ldlt();

        scalar eprev;
        scalar sm(1.0);
        int it = 1;
        for (; it <= max_iterations; ++it) {
            v     = ham_dec.solve(w * sm);
            w     = dn * v;
            sm    = 1.0 / sqrt(v.dot(w));
            eprev = eig;
            eig   = eold + sm;
            if (abs(eprev - eig) < epsilon * abs(eig))
                break;
        }
        if (it == max_iterations) {
            cout << " itmax or eps too small\n"
                 << " lack of convergence in inverse iteration\n";
        }
        return eig;
    };

    const auto en = nelder_mead_minimize_parallel<scalar, m>(target, xv, scalar(5.0e-2), 1.0, 2.0,
                                                             0.5, 0.5, 500);

    cout << "FINAL RESULT\n"
         << "energy:\n"
         << en << '\n'
         << "x vec:\n";
    cout << xv << '\n';

    ofstream ofs("basis_2p_1500.dat");
    ofs << setprecision(numeric_limits<scalar>::max_digits10) << scientific;
    ofs << "# E = " << en << '\n' << "# n = " << n << '\n' << generate_basis(xv, n) << '\n';
    ofs.close();
}
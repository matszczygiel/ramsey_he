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
    constexpr int max_iters = 150;
    const scalar en_drake("-2.903724377034119598311e+00");

    Eigen::initParallel();
    ios::sync_with_stdio(false);
    cout << " Eigen is using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    Matrix<scalar, m, 1> xv;

    constexpr int n = 1500;
    xv << scalar(
        "1.42043682224255552192648452205512902116281893277140873470417409988482535649e+00"),
        scalar("4.29487843021043600160679449793341056023712587912543043000554047033579869402e+00"),
        scalar("1.13866620714069379522528685268623081338616222639483174067250225851119498767e+00"),
        scalar("7.68882883577075515470503329077154225316537440161819472031030416719041981010e+00"),
        scalar("-4.44109255368266047661241784844692855479937629936592199755788340158694427652e-01"),
        scalar("1.43743053206967610397953096858465158711126328508645408733935607610548786304e+00"),
        scalar("-3.04112241713840246245874007229344735212624742960997438448247421231144465714e-01"),
        scalar("1.48580715954296267856971998132880431462296738341547493756916248267019416285e+01"),
        scalar("1.56899595230114816627022050543686806894045039624439548254797472380217252445e+01"),
        scalar("5.74384759427408158254108209448887824308030195461533801361321714734843360212e+00"),
        scalar("1.42956125651972730278198360805550727492635306265080273024709246768721029815e+01"),
        scalar("6.14563575006347566612725085381997551768216766792845602582626109291277158488e+00"),
        scalar("1.59912180462863649860710050198698411927074650403481152100875974506267451652e+01"),
        scalar("2.61013786533665985890416282124499445511443075261265448194121167822877342277e+00");

    /*
        constexpr int n         = 100;
        xv << scalar("1.370659897531483873e+00"), scalar("2.567427642785374609e+00"),
            scalar("1.100897653897865425e+00"), scalar("5.287682379453643833e+00"),
            scalar("-9.158710023379174059e-02"), scalar("4.318208153953369544e-01"),
            scalar("1.206145093682640113e+00"), scalar("2.911287848993912686e+00"),
            scalar("5.342357541483666594e+00"), scalar("3.676084240850745921e+00"),
            scalar("6.616779402212323191e+00"), scalar("-1.547578409752147777e-01"),
            scalar("2.006913710260977535e+00"), scalar("2.868941166396368203e+00");
    */
    const auto target = [&en_drake](const Matrix<scalar, m, 1>& x) {
        const auto phi      = generate_basis(x, n);
        const auto [dh, dn] = generate_matrices_S(phi);
        return get<0>(solve_for_state<scalar>(dh, dn, en_drake - 1.0e-5, max_iters, 1.0e-45));
    };

    const auto en = nelder_mead_minimize_parallel<scalar, m>(target, xv, scalar(5.0e-2), 1.0, 2.0,
                                                             0.5, 0.5, 500);

    cout << "FINAL RESULT\n"
         << "energy:\n"
         << en << '\n'
         << "x vec:\n";
    cout << xv << '\n';

    ofstream ofs("basis_1s_1500.dat");
    ofs << setprecision(numeric_limits<scalar>::max_digits10) << scientific;
    ofs << "# E = " << en << '\n' << "# n = " << n << '\n' << generate_basis(xv, n) << '\n';
    ofs.close();
}
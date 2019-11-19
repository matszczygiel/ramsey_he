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
using scalar = boost::multiprecision::cpp_bin_float_double;
// using scalar = boost::multiprecision::mpfr_float_50;
// using scalar = boost::multiprecision::cpp_bin_float_100;

namespace Eigen {
template <>
struct NumTraits<scalar> : GenericNumTraits<scalar> {};
}  // namespace Eigen

int main() {
    constexpr int max_iters = 150;
    constexpr int n         = 100;
    const scalar en_drake("-2.903724377034119598311e+00");

    Eigen::initParallel();
    ios::sync_with_stdio(false);
    cout << " Eigen is using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    Matrix<scalar, m, 1> xv;

    // n=1500
    /*
    xv << scalar(
        "1.43921872077665042847945140931561500191340434857178218964234832855982751919e+00"),
        scalar("4.20967785893995519535898049002037097898748251899787980958331700462944274883e+00"),
        scalar("1.15495306070503856176098817233041284356125523939007790833616496808457824260e+00"),
        scalar("7.51536559248258133963461898549432326004114499414855063571581800671582780990e+00"),
        scalar("-4.53554088290049122815672692820883171416311995685436322702284571189987085031e-01"),
        scalar("1.47648587594454859863153040672789649135695800030210425960922123355423013045e+00"),
        scalar("1.24976629211336308493029687808821018498472678245668485810344672002619918224e-01"),
        scalar("1.49534265295240676211740466268831652257540306122560296283699593838881098927e+01"),
        scalar("1.56691736177971415108278077602872996569273116517439246616607997060693214785e+01"),
        scalar("5.55647563076354019403237368354598903367002244067680139083878781074535730462e+00"),
        scalar("1.35944908483329847081551546213356012204928116955042450970823137653855121232e+01"),
        scalar("6.18676295920281346016484188384452883146327317506620984050542226580477173154e+00"),
        scalar("1.55642346323027899472780820281776610860019453908679966551081501810211924323e+01"),
        scalar("3.05050221019702797806973773475699603076557990490160168781020241920703291909e+00");
*/
    // n=100

    xv << scalar("1.370659897531483873e+00"), scalar("2.567427642785374609e+00"),
        scalar("1.100897653897865425e+00"), scalar("5.287682379453643833e+00"),
        scalar("-9.158710023379174059e-02"), scalar("4.318208153953369544e-01"),
        scalar("1.206145093682640113e+00"), scalar("2.911287848993912686e+00"),
        scalar("5.342357541483666594e+00"), scalar("3.676084240850745921e+00"),
        scalar("6.616779402212323191e+00"), scalar("-1.547578409752147777e-01"),
        scalar("2.006913710260977535e+00"), scalar("2.868941166396368203e+00");

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

    ofstream ofs("basis_1s.dat");
    ofs << setprecision(numeric_limits<scalar>::max_digits10) << scientific;
    ofs << "# E = " << en << '\n' << "# n = " << n << '\n' << generate_basis(xv, n) << '\n';
    ofs.close();
}
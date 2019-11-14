#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "definitions.h"
#include "functions.h"
#include "nelder_mead.h"

using namespace Eigen;
using namespace std;

int main() {
    Eigen::initParallel();
    ios::sync_with_stdio(false);
    cout << " Eigen is using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    constexpr int n = 1500;
    const scalar en_drake("-2.903724377034119598311e+00");

    Matrix<scalar, m, 1> xv;

    //n=1500
    /*
    xv << scalar("1.5484825102880660e+00"),
        scalar("3.9450565843621390e+00"),
        scalar("1.2518569958847740e+00"),
        scalar("7.0881138516837940e+00"),
        scalar("-1.2209851229161200e-01"),
        scalar("1.0221724244682980e+00"),
        scalar("3.9570759030633790e-01"),
        scalar("1.5069229538180200e+01"),
        scalar("1.5638604252400470e+01"),
        scalar("5.4625324055039820e+00"),
        scalar("1.3306565704140900e+01"),
        scalar("6.2832879717888820e+00"),
        scalar("1.5187919971788840e+01"),
        scalar("3.4031638437888390e+00");
*/
    xv << scalar("1.54848251028806600000000000000000000000000000000000000000000000000000000045e+00"),
        scalar("3.94505658436213899999999999999999999999999989999999999999999999999999999108e+00"),
        scalar("1.25185699588477399999999999999999999999999989999999999999999999999999999586e+00"),
        scalar("7.08811385168379400000000000000000000000000000000000000000000000000000001293e+00"),
        scalar("-1.22098512291612000000000000000000000000000000000000000000000000000000000112e-01"),
        scalar("1.02217242446829799999999999999999999999999990000000000000000000000000000209e+00"),
        scalar("4.95707590306337905551115123125782702118158340227050781249999999999999999896e-01"),
        scalar("1.50692295381801999999999999999999999999999990000000000000000000000000000065e+01"),
        scalar("1.56386042524004699999999999999999999999999990000000000000000000000000000262e+01"),
        scalar("5.46253240550398200000000000000000000000000000000000000000000000000000001455e+00"),
        scalar("1.33065657041409000000000000000000000000000000000000000000000000000000000151e+01"),
        scalar("6.28328797178888200000000000000000000000000000000000000000000000000000001307e+00"),
        scalar("1.51879199717888400000000000000000000000000000000000000000000000000000000257e+01"),
        scalar("3.40316384378883899999999999999999999999999990000000000000000000000000000659e+00");

    //n=100
    /*
    xv << scalar("1.2845084222741440e+00"),
        scalar("2.5525827520389550e+00"),
        scalar("1.3022061582156230e+00"),
        scalar("5.2592228607912200e+00"),
        scalar("-5.3005320302503930e-02"),
        scalar("5.0118833268966480e-01"),
        scalar("1.1664947151404250e+00"),
        scalar("2.9309131965855850e+00"),
        scalar("5.3276662855341290e+00"),
        scalar("3.6503129028788250e+00"),
        scalar("6.5776225188830830e+00"),
        scalar("-2.1506973153746800e-01"),
        scalar("1.9548131347420990e+00"),
        scalar("2.8616266805520390e+00");
*/

    const auto target = [&en_drake](const Matrix<scalar, m, 1>& x) {
        const scalar epsilon = 1.0e-45;
        scalar eig           = en_drake - 1.0e-5;
        const scalar eold    = eig;

        const auto phi = generate_wf(x, n);

        const auto [dh, dn] = generate_matrices(phi);

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

    const auto en = nelder_mead_minimize_parallel<scalar, m>(
        target, xv, scalar(5.0e-2), 1.0, 2.0, 0.5, 0.5, 150);

    cout << "FINAL RESULT\n"
         << "energy:\n"
         << en << '\n'
         << "x vec:\n";
    cout << xv << '\n';
}
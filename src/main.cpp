#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "functions.h"
#include "definitions.h"

using namespace Eigen;
using namespace std;

int main() {
    std::cout << std::setprecision(std::numeric_limits<scalar>::max_digits10) << scientific;

    constexpr int max_iterations = 30;
    constexpr int m              = 14;
    constexpr int n              = 100;
    const scalar en_drake("-2.903724377034119598311e+00");

    Matrix<scalar, m, 1> x;

    // for n = 1500
    /*
    x << scalar("1.2845084222741440e+00"),
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

    // for n = 100

    x << scalar("1.2845084222741440e+00"),
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

    const scalar epsilon("1.0e-50");
    scalar eig  = en_drake - 1.0e-5;
    scalar eold = eig;
    scalar eprev;

    const auto phi = generate_wf(x, n);

    const auto [dh, dn] = generate_matrices(phi);

    Matrix<scalar, Dynamic, 1> v = Matrix<scalar, Dynamic, 1>::Ones(n);
    Matrix<scalar, Dynamic, 1> w = v;
    Matrix<scalar, Dynamic, 1> u = v;

    auto hh           = dh - eig * dn;
    // Matrix<scalar, Dynamic, Dynamic>::Identity(n, n);
    auto dh_ldlt = hh.ldlt();

    scalar sm;
    int it = 1;
    for (; it <= max_iterations; ++it) {
        u     = dh_ldlt.solve(v);
        v     = u;
        w     = dn * v;
        sm    = v.dot(w);
        sm    = 1.0 / sqrt(sm);
        eprev = eig;
        eig   = eold + sm;
        std::cout << "it = " << it << "  eig = " << eig << std::endl;
        v = w * sm;
        if (abs(eprev - eig) < epsilon * abs(eig))
            break;
    }
    if (it == max_iterations) {
        std::cout << " ITMAX OR EPS TOO SMALL\n"
                  << " LACK OF CONVERGENCE IN INVERSE ITERATION\n";
    }

    std::cout << "N   =     " << n << '\n'
              << "EIG =     " << eig << '\n';

    v = u * sm;

    std::cout << "NORM-1 =  " << v.dot(dn * v) - scalar(1.0) << '\n';
    auto est = v.dot(dh * v);
    std::cout << "EST =     " << est << '\n'
              << "EIG-EST = " << eig - est << '\n';

}
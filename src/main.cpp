#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "definitions.h"
#include "functions.h"

using namespace Eigen;
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cout << " Using " << nbThreads() << " threads" << endl;
    cout << setprecision(numeric_limits<scalar>::max_digits10) << scientific;

    constexpr int n = 1500;
    const scalar en_drake("-2.903724377034119598311e+00");

    const array<scalar, m> x = {
        scalar("1.5484825102880660e+00"),
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
        scalar("3.4031638437888390e+00")};

    const scalar epsilon = 1.0e-40;
    scalar eig           = en_drake - 1.0e-5;
    const scalar eold    = eig;

    const auto phi = generate_wf(x, n);

    const auto [dh, dn] = generate_matrices(phi);

    Matrix<scalar, Dynamic, 1> v = Matrix<scalar, Dynamic, 1>::Ones(n);
    Matrix<scalar, Dynamic, 1> w = v;

    const auto ham_dec = (dh - eig * dn).ldlt();

    cout << " Decomposition info:\n";
    check_and_report_eigen_info(cout, ham_dec.info());

    scalar eprev;
    scalar sm(1.0);
    int it = 1;
    for (; it <= max_iterations; ++it) {
        v     = ham_dec.solve(w * sm);
        w     = dn * v;
        sm    = 1.0 / sqrt(v.dot(w));
        eprev = eig;
        eig   = eold + sm;
        cout << "it = " << it << "  eig = " << eig << endl;
        if (abs(eprev - eig) < epsilon * abs(eig))
            break;
    }
    if (it == max_iterations) {
        cout << " itmax or eps too small\n"
             << " lack of convergence in inverse iteration\n";
    }

    cout << "n   =     " << n << '\n'
         << "eig =     " << eig << '\n';

    v *= sm;

    cout << "norm - 1 =  " << v.dot(dn * v) - scalar(1.0) << '\n';
    const auto est = v.dot(dh * v);
    cout << "est =       " << est << '\n'
         << "eig - est = " << eig - est << '\n';
}
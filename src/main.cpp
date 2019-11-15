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
    xv << scalar("1.43395945662889271988699338620840887501452496176582110691883986955752478733e+00"),
        scalar("4.12058186614899346914274512375693640093555515859509160213779636722720894491e+00"),
        scalar("1.20835377149540318570683907278961927798367735539672593355547883031861981397e+00"),
        scalar("7.10687188828050513081254839707766546333719394998614958844634059007784607205e+00"),
        scalar("-1.68240100759589278282623123631295306456732694537189860784033820495792161521e-01"),
        scalar("1.09003104476989197885612427447580807629843927603856365183279710612941725110e+00"),
        scalar("4.35026208432069764695486929116173720198205438032437366023166501794150366121e-01"),
        scalar("1.50410161094451321707137192318184097418759344641772052776958987950155159093e+01"),
        scalar("1.56408965470297371822559044863809190439807192442514408721503598895421425821e+01"),
        scalar("5.45075571925741883510471242096656101778203819611464482135200158558440566315e+00"),
        scalar("1.33676019892637294704056342764894874842803744424709575715491500605792511198e+01"),
        scalar("6.30860490116557755773022057541154067317591603388155747535619094602142307240e+00"),
        scalar("1.52082973970572096980061739832193732411505688112338302357266105054273560510e+01"),
        scalar("3.38336569078710771706427163829502551058592316759218886648127769856515974318e+00");

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

        const auto phi = generate_basis(x, n);

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
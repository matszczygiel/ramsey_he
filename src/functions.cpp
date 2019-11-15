#include "functions.h"

#include <eigen3/Eigen/Dense>

#include "definitions.h"
#include "random.h"

using namespace Eigen;

Matrix<scalar, Dynamic, 3> generate_basis(const Matrix<scalar, m, 1>& x, int rows) {
    Matrix<scalar, Dynamic, 3> phi(rows, 3);
//    std::uniform_real_distribution<> dis(0.0, 1.0);
//    static auto rd_seed = std::random_device{}();
//    std::mt19937 rd{rd_seed};
    Ranmar rd(ranmar_i, ranmar_j);
    {
        const auto& a1 = x(0);
        const auto& a2 = x(1);
        const auto& b1 = x(2);
        const auto& b2 = x(3);
        const auto& c1 = x(4);
        const auto& c2 = x(5);
        const auto& de = x(6);

        for (int i = 0; i < phi.rows() / 2; ++i) {
            do {
                phi(i, 0) = rd() * (a2 - a1) + a1;
                phi(i, 1) = rd() * (b2 - b1) + b1;
                phi(i, 2) = rd() * (c2 - c1) + c1;
            } while (phi(i, 0) + phi(i, 1) < de || phi(i, 1) + phi(i, 2) < de || phi(i, 0) + phi(i, 2) < de);
        }
    }
    {
        const auto& a1 = x(7);
        const auto& a2 = x(8);
        const auto& b1 = x(9);
        const auto& b2 = x(10);
        const auto& c1 = x(11);
        const auto& c2 = x(12);
        const auto& de = x(13);

        for (int i = phi.rows() / 2; i < phi.rows(); ++i) {
            do {
                phi(i, 0) = rd() * (a2 - a1) + a1;
                phi(i, 1) = rd() * (b2 - b1) + b1;
                phi(i, 2) = rd() * (c2 - c1) + c1;
            } while (phi(i, 0) + phi(i, 1) < de || phi(i, 1) + phi(i, 2) < de || phi(i, 0) + phi(i, 2) < de);
        }
    }

    return phi;
}

std::tuple<Matrix<scalar, Dynamic, Dynamic>, Matrix<scalar, Dynamic, Dynamic>>
generate_matrices(
    const Matrix<scalar, Dynamic, 3>& phi) {
    const auto size = phi.rows();

    Matrix<scalar, Dynamic, Dynamic> h(size, size);
    Matrix<scalar, Dynamic, Dynamic> s(size, size);

    for (int i = 0; i < size; ++i) {
        const auto& a1 = phi(i, 0);
        const auto& b1 = phi(i, 1);
        const auto& c1 = phi(i, 2);

        for (int j = i; j < size; ++j) {
            const auto& a2 = phi(j, 0);
            const auto& b2 = phi(j, 1);
            const auto& c2 = phi(j, 2);

            const auto a = 2.0 / (b1 + b2 + c1 + c2);
            const auto b = 2.0 / (a1 + a2 + c1 + c2);
            const auto c = 2.0 / (a1 + a2 + b1 + b2);
            const auto d = 2.0 / (a1 + b2 + c1 + c2);
            const auto f = 2.0 / (b1 + a2 + c1 + c2);

            const auto X1  = a * b * c;
            const auto X2  = a * b;
            const auto X3  = a * c;
            const auto X4  = b * c;
            const auto X5  = X2 + X3 + X4;
            const auto X6  = a * a;
            const auto X7  = b * b;
            const auto X8  = c * c;
            const auto X9  = a + c;
            const auto X10 = b + c;
            const auto X11 = a + b;

            const auto Y1  = c * d * f;
            const auto Y2  = d * f;
            const auto Y3  = d * c;
            const auto Y4  = f * c;
            const auto Y5  = Y2 + Y3 + Y4;
            const auto Y6  = d * d;
            const auto Y7  = f * f;
            const auto Y8  = c * c;
            const auto Y9  = d + c;
            const auto Y10 = f + c;
            const auto Y11 = d + f;

            const scalar Z(2.0);

            h(i, j) = (X1 * (4.0 * X8 + 2.0 * X5 -
                             2.0 * X3 * X9 * (a2 * c1 + a1 * c2) - 2.0 * X4 * X10 * (b2 * c1 + b1 * c2) +
                             (X1 + X11 * X8 + X7 * X9 + X6 * X10) *
                                 (a1 * a2 + b1 * b2 + a2 * c1 + b2 * c1 + a1 * c2 + b1 * c2 + 2.0 * c1 * c2) -
                             4.0 * (X6 + X7 + X5) * Z)) /
                          128.0 +
                      (Y1 * (4.0 * Y8 - 2.0 * Y3 * (a2 * c1 + b1 * c2) * Y9 -
                             2.0 * Y4 * (b2 * c1 + a1 * c2) * Y10 + 2.0 * Y5 +
                             (a2 * b1 + a1 * b2 + a2 * c1 + b2 * c1 + a1 * c2 + b1 * c2 + 2.0 * c1 * c2) *
                                 (Y1 + Y9 * Y7 + Y6 * Y10 + Y8 * Y11) -
                             4.0 * (Y5 + Y6 + Y7) * Z)) /
                          128.0;

            s(i, j) = X1 * (X6 * X10 + X7 * X9 + X8 * X11 + X1) / 64.0 +
                      Y1 * (Y8 * Y11 + Y6 * Y10 + Y7 * Y9 + Y1) / 64.0;


            h(j, i) = h(i, j);
            s(j, i) = s(i, j);
        }
    }

    return std::make_tuple(h, s);
}
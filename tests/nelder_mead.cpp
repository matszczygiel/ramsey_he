#include <array>
#include <iostream>

#include "nelder_mead.h"

int main() {
    const auto rosenbrock = [](const Eigen::Matrix<double, 2, 1>& x) {
        return (1.0 - x(0)) * (1.0 - x(0)) + 100.0 * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
    };

    Eigen::Matrix<double, 2, 1> target_x;
    target_x << 1.0, 1.0;
    constexpr double target_f = 0.0;

    Eigen::Matrix<double, 2, 1> x;
    x << 2.0, -1.0;
    const auto f_min        = nelder_mead_minimize<double, 2>(rosenbrock, x, 0.2, 1.0, 2.0, 0.5, 0.5, 1.0e-5, 100);

    std::cout << "FINAL RESULT\n"
              << " from nelder_mead:\n"
              << " f = " << f_min << '\n'
              << " x = " << x(0) << '\n'
              << " y = " << x(1) << '\n'
              << " true minimum:\n"
              << " f = " << target_f << '\n'
              << " x = " << target_x(0) << '\n'
              << " y = " << target_x(1) << '\n';
};
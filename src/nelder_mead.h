#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>

#include <eigen3/Eigen/Dense>

//The Nelder–Mead method of minimazing function
//https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
template <class T, int N>
T nelder_mead_minimize(const std::function<T(const Eigen::Matrix<T, N, 1>&)>& func, Eigen::Matrix<T, N, 1>& x,
                       T initial_var, const double& alpha, const double& gamma, const double& rho, const double& sigma,
                       const T& epsilon, int max_iters) {
    assert(alpha > 0.0);
    assert(gamma > 1.0);
    assert(0.5 >= rho && rho > 0.0);
    assert(1.0 > sigma && sigma > 0.0);
    assert(max_iters > 1);

    using Vec_t = std::decay_t<decltype(x)>;

    std::array<std::pair<Vec_t, T>, N + 1 > xfn;
    const auto sort_xfns = [&xfn]() { std::sort(xfn.begin(), xfn.end(), [](const auto& xf1, const auto& xf2) {
                                          return xf1.second < xf2.second;
                                      }); };

    T current_minimum;
    const auto print_info = [&x, &current_minimum]() {
        std::cout << " minimum : "
                  << current_minimum << std::endl;
        //        std::cout << " x vector:\n";
        //        for (const auto& xi : x)
        //            std::cout << xi << '\n';
        //        std::cout << std::endl;
    };

    xfn[0] = std::make_pair(x, func(x));
    for (int i = 1; i <= N; ++i) {
        xfn[i].first = x;
        xfn[i].first[i - 1] += initial_var;
        xfn[i].second = func(xfn[i].first);
    }

    sort_xfns();

    std::tie(x, current_minimum) = xfn.back();

    std::cout << " inital parameters\n";
    print_info();

    Vec_t x_o, x_r, x_e, x_c;
    T f_r, f_e, f_c;

    int iteration = 1;
    for (; iteration <= max_iters; ++iteration) {
        std::cout << " iteration: ";
        std::cout << std::setw(3) << iteration << "  ";

        x_o = Vec_t::Zero();
        for (int j = 0; j < N; ++j) {
            x_o += xfn[j].first;
        }
        x_o /= N;
        x_r = (1.0 + alpha) * x_o - alpha * xfn.back().first;

        f_r = func(x_r);

        if (xfn[0].second <= f_r && f_r < xfn[N - 1].second) {
            xfn.back() = {x_r, f_r};
        } else if (f_r < xfn[0].second) {
            x_e = x_o * (1.0 - rho) + gamma * x_r;
            f_e = func(x_e);

            if (f_e < f_r) {
                xfn.back() = {x_e, f_e};
            } else {
                xfn.back() = {x_r, f_r};
            }
        } else {
            x_c = (1.0 - rho) * x_o + rho * xfn.back().first;
            f_c = func(x_c);
            if (f_c < xfn.back().second) {
                xfn.back() = {x_c, f_c};
            } else {
                for (int i = 1; i <= N; ++i) {
                    xfn[i].first  = (1.0 - sigma) * xfn[0].first + sigma * xfn[i].first;
                    xfn[i].second = func(xfn[i].first);
                }
            }
        }
        sort_xfns();
        const auto last_min          = current_minimum;
        std::tie(x, current_minimum) = xfn.back();
        print_info();

        //     if (abs(last_min - current_minimum) < epsilon)
        //         break;
    }

    return current_minimum;
}
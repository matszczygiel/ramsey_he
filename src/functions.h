#pragma once

#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

#include <eigen3/Eigen/Dense>

inline bool check_and_report_eigen_info(std::ostream& os, const Eigen::ComputationInfo& info) {
    switch (info) {
        case Eigen::ComputationInfo::NumericalIssue:
            os << " The provided data did not satisfy the prerequisites.\n";
            return true;
        case Eigen::ComputationInfo::NoConvergence:
            os << " Iterative procedure did not converge.\n";
            return true;
        case Eigen::ComputationInfo::InvalidInput:
            os << " The inputs are invalid, or the algorithm has been improperly called."
               << "When assertions are enabled, such errors trigger an assert.\n";
            return true;
        case Eigen::ComputationInfo::Success:
            os << " Success\n";
            return false;
    }
    return false;
}

constexpr int m = 14;

template <class T>
using Basis = Eigen::Matrix<T, Eigen::Dynamic, 3>;

template <class T>
Basis<T> generate_basis(const Eigen::Matrix<T, m, 1>& x, int rows);

template <class T>
using Hamiltonian = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T>
using Overlap = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <class T>
std::tuple<Hamiltonian<T>, Overlap<T>> generate_matrices_S(const Basis<T>& phi);
template <class T>
std::tuple<Hamiltonian<T>, Overlap<T>> generate_matrices_P(const Basis<T>& phi);

template <class T>
using Current = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <class T>
Current<T> generate_current_PS(const Basis<T>& phi_P, const Basis<T>& phi_S);

template <class T>
using Energy = T;

template <class T>
std::tuple<Energy<T>, Basis<T>> load_basis(const std::string_view filename);

template <class T>
using Wavefunction = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
std::tuple<Energy<T>, Wavefunction<T>> solve_for_state(const Hamiltonian<T>& h, const Overlap<T>& n,
                                                       const Energy<T> approx_eigenvalue,
                                                       const int max_iterations,
                                                       const double& epsilon);

template <class T>
Basis<T> generate_basis(const Eigen::Matrix<T, m, 1>& x, int rows) {
    Basis<T> phi(rows, 3);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::mt19937 rd{0};
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
                phi(i, 0) = dis(rd) * (a2 - a1) + a1;
                phi(i, 1) = dis(rd) * (b2 - b1) + b1;
                phi(i, 2) = dis(rd) * (c2 - c1) + c1;
            } while (phi(i, 0) + phi(i, 1) < de || phi(i, 1) + phi(i, 2) < de ||
                     phi(i, 0) + phi(i, 2) < de);
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
                phi(i, 0) = dis(rd) * (a2 - a1) + a1;
                phi(i, 1) = dis(rd) * (b2 - b1) + b1;
                phi(i, 2) = dis(rd) * (c2 - c1) + c1;
            } while (phi(i, 0) + phi(i, 1) < de || phi(i, 1) + phi(i, 2) < de ||
                     phi(i, 0) + phi(i, 2) < de);
        }
    }

    return phi;
}

template <class T>
std::tuple<Hamiltonian<T>, Overlap<T>> generate_matrices_S(const Basis<T>& phi) {
    const auto size = phi.rows();

    Hamiltonian<T> h(size, size);
    Overlap<T> s(size, size);

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
            const auto e = 2.0 / (b1 + a2 + c1 + c2);

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

            const auto Y1  = c * d * e;
            const auto Y2  = d * e;
            const auto Y3  = d * c;
            const auto Y4  = e * c;
            const auto Y5  = Y2 + Y3 + Y4;
            const auto Y6  = d * d;
            const auto Y7  = e * e;
            const auto Y8  = c * c;
            const auto Y9  = d + c;
            const auto Y10 = e + c;
            const auto Y11 = d + e;

            const T Z(2.0);

            h(i, j) =
                (X1 *
                 (4.0 * X8 + 2.0 * X5 - 2.0 * X3 * X9 * (a2 * c1 + a1 * c2) -
                  2.0 * X4 * X10 * (b2 * c1 + b1 * c2) +
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

    return {h, s};
}

template <class T>
std::tuple<Hamiltonian<T>, Overlap<T>> generate_matrices_P(const Basis<T>& phi) {
    const auto size = phi.rows();

    Hamiltonian<T> h(size, size);
    Overlap<T> s(size, size);

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
            const auto e = 2.0 / (b1 + a2 + c1 + c2);

            const T Z(2.0);

            h(i, j) =
                (a * (-a1 - a2) * b * c *
                     (2 * a * a * b * b + 3 * a * b * b * b + 2 * a * a * b * c +
                      3 * a * b * b * c + 3 * b * b * b * c + 2 * a * a * c * c +
                      3 * a * b * c * c + 4 * b * b * c * c + 3 * a * c * c * c +
                      3 * b * c * c * c) +
                 6 * a * b * c *
                     (a * a * b + a * b * b + a * b * b * b + a * a * c + a * b * c + b * b * c +
                      a * b * b * c + b * b * b * c + a * c * c + b * c * c + a * b * c * c +
                      2 * b * b * c * c + a * c * c * c + 3 * b * c * c * c + 4 * c * c * c * c) +
                 a * b * c * (a * b - a * c + b * c) *
                     (2 * a * b + 3 * b * b + 2 * a * c + 4 * b * c + 3 * c * c) * (-c1 - c2) +
                 a * b * c * (a * b - a * c + b * c) *
                     (3 * a * b * b + 6 * b * b * b + 4 * a * b * c + 9 * b * b * c +
                      3 * a * c * c + 9 * b * c * c + 6 * c * c * c) *
                     (a2 * c1 + a1 * c2) -
                 3 * a * b * c *
                     (a * a * b * b * b + 2 * a * b * b * b * b + a * a * b * b * c +
                      2 * a * b * b * b * c - 2 * b * b * b * b * c + a * a * b * c * c +
                      2 * a * b * b * c * c - 3 * b * b * b * c * c + a * a * c * c * c +
                      2 * a * b * c * c * c - 3 * b * b * c * c * c + 2 * a * c * c * c * c -
                      2 * b * c * c * c * c) *
                     (-(b2 * c1) - b1 * c2) +
                 3 * a * b * c *
                     (a * a * b * b * b + 2 * a * b * b * b * b + a * a * b * b * c +
                      2 * a * b * b * b * c + 2 * b * b * b * b * c + a * a * b * c * c +
                      2 * a * b * b * c * c + 3 * b * b * b * c * c + a * a * c * c * c +
                      2 * a * b * c * c * c + 3 * b * b * c * c * c + 2 * a * c * c * c * c +
                      2 * b * c * c * c * c) *
                     (a1 * a2 + b1 * b2 + 2 * c1 * c2) -
                 4 * a * b * c *
                     (a * a * b * b + 3 * a * b * b * b + 6 * b * b * b * b + a * a * b * c +
                      3 * a * b * b * c + 6 * b * b * b * c + a * a * c * c + 3 * a * b * c * c +
                      5 * b * b * c * c + 3 * a * c * c * c + 3 * b * c * c * c) *
                     Z) /
                    256.0 +
                (-(b1 * c * d * e * (c * d + c * e - d * e) *
                   (3 * c * c + 4 * c * d + 3 * d * d + 2 * c * e + 2 * d * e)) +
                 c * c1 * d * e * (c * d - c * e + d * e) *
                     (3 * c * c + 4 * c * d + 3 * d * d + 2 * c * e + 2 * d * e) -
                 c * c2 * d * e * (c * d - c * e - d * e) *
                     (3 * c * c + 2 * c * d + 4 * c * e + 2 * d * e + 3 * e * e) -
                 b2 * c * d * e * (c * d + c * e - d * e) *
                     (3 * c * c + 2 * c * d + 4 * c * e + 2 * d * e + 3 * e * e) +
                 3 * c * (a2 * b1 + a1 * b2 + 2 * c1 * c2) * d * e * (c * d + c * e - d * e) *
                     (2 * c * c * c + 2 * c * c * d + c * d * d + 2 * c * c * e + 2 * c * d * e +
                      d * d * e + c * e * e + d * e * e) +
                 4 * c * d * e *
                     (6 * c * c * c * c + 3 * c * c * c * d + c * c * d * d + 3 * c * c * c * e +
                      c * c * d * e + c * c * e * e - d * d * e * e) +
                 3 * c * (b2 * c1 + a1 * c2) * d * e *
                     (2 * c * c * c * c * d + 2 * c * c * c * d * d + c * c * d * d * d -
                      2 * c * c * c * c * e + c * c * c * d * e + c * c * d * d * e -
                      2 * c * c * c * e * e + c * c * d * e * e - d * d * d * e * e -
                      c * c * e * e * e + c * d * e * e * e - d * d * e * e * e) +
                 3 * c * (-(a2 * c1) - b1 * c2) * d * e *
                     (2 * c * c * c * c * d + 2 * c * c * c * d * d + c * c * d * d * d -
                      2 * c * c * c * c * e - c * c * c * d * e - c * c * d * d * e -
                      c * d * d * d * e - 2 * c * c * c * e * e - c * c * d * e * e +
                      d * d * d * e * e - c * c * e * e * e + d * d * e * e * e) -
                 2 * c * d * e * (c * d + c * e - d * e) *
                     (6 * c * c + 6 * c * d + 3 * d * d + 6 * c * e + 4 * d * e + 3 * e * e) * Z) /
                    256.0;

            s(i, j) = (3 * a * b * c *
                       (a * a * b * b * b + 2 * a * b * b * b * b + a * a * b * b * c +
                        2 * a * b * b * b * c + 2 * b * b * b * b * c + a * a * b * c * c +
                        2 * a * b * b * c * c + 3 * b * b * b * c * c + a * a * c * c * c +
                        2 * a * b * c * c * c + 3 * b * b * c * c * c + 2 * a * c * c * c * c +
                        2 * b * c * c * c * c)) /
                          128.0 +
                      (3 * c * d * e * (c * d + c * e - d * e) *
                       (2 * c * c * c + 2 * c * c * d + c * d * d + 2 * c * c * e + 2 * c * d * e +
                        d * d * e + c * e * e + d * e * e)) /
                          128.0;

            s(j, i) = s(i, j);
            h(j, i) = h(i, j);
        }
    }

    return {h, s};
}

template <class T>
Current<T> generate_current_PS(const Basis<T>& phi_P, const Basis<T>& phi_S) {
    const auto size_p = phi_P.rows();
    const auto size_s = phi_S.rows();

    Current<T> hj(size_p, size_s);

    for (int i = 0; i < size_p; ++i) {
        const auto& a1 = phi_P(i, 0);
        const auto& b1 = phi_P(i, 1);
        const auto& c1 = phi_P(i, 2);

        for (int j = 0; j < size_s; ++j) {
            const auto& a2 = phi_S(j, 0);
            const auto& b2 = phi_S(j, 1);
            const auto& c2 = phi_S(j, 2);

            const auto a = 2.0 / (b1 + b2 + c1 + c2);
            const auto b = 2.0 / (a1 + a2 + c1 + c2);
            const auto c = 2.0 / (a1 + a2 + b1 + b2);
            const auto d = 2.0 / (a1 + b2 + c1 + c2);
            const auto e = 2.0 / (b1 + a2 + c1 + c2);

            hj(i, j) = (-3 * a * b * c *
                        (a * a * a * b * b - 2 * a * b * b * b * b - a * a * b * b * c -
                         2 * a * b * b * b * c - 2 * b * b * b * b * c - a * a * a * c * c -
                         2 * a * a * b * c * c - 3 * a * b * b * c * c - 4 * b * b * b * c * c -
                         3 * a * a * c * c * c - 4 * a * b * c * c * c - 5 * b * b * c * c * c -
                         4 * a * c * c * c * c - 4 * b * c * c * c * c)) /
                           128.0 -
                       (3 * c * d * e *
                        (-4 * c * c * c * c * d - 5 * c * c * c * d * d - 4 * c * c * d * d * d -
                         2 * c * d * d * d * d - 4 * c * c * c * c * e - 4 * c * c * c * d * e -
                         3 * c * c * d * d * e - 2 * c * d * d * d * e - 2 * d * d * d * d * e -
                         3 * c * c * c * e * e - 2 * c * c * d * e * e - c * d * d * e * e -
                         c * c * e * e * e + d * d * e * e * e)) /
                           128.0;
        }
    }

    return hj;
}

template <class T>
std::tuple<Energy<T>, Basis<T>> load_basis(const std::string_view filename) {
    std::ifstream file(filename.data());

    std::string line;
    std::getline(file, line);
    if (line[0] != '#')
        throw std::runtime_error("Invalid input file!");

    line.erase(0, 6);
    const Energy<T> en(line);

    std::getline(file, line);
    if (line[0] != '#')
        throw std::runtime_error("Invalid input file!");

    line.erase(0, 6);
    const auto n = std::stoi(line);

    Basis<T> basis(n, 3);

    for (int i = 0; i < n; ++i) {
        std::string num0, num1, num2;
        file >> num0 >> num1 >> num2;
        basis(i, 0) = T(num0);
        basis(i, 1) = T(num1);
        basis(i, 2) = T(num2);
    }

    return {en, basis};
}

template <class T>
std::tuple<Energy<T>, Wavefunction<T>> solve_for_state(const Hamiltonian<T>& h, const Overlap<T>& n,
                                                       const Energy<T> approx_eigenvalue,
                                                       const int max_iterations,
                                                       const double& epsilon) {
    const T eps = epsilon;
    T eig       = approx_eigenvalue;

    Wavefunction<T> v = Wavefunction<T>::Ones(h.cols());
    Wavefunction<T> w = v;

    const auto ham_dec = (h - eig * n).ldlt();

    T eprev;
    T sm(1.0);
    int it = 1;
    for (; it <= max_iterations; ++it) {
        v     = ham_dec.solve(w * sm);
        w     = n * v;
        sm    = 1.0 / sqrt(v.dot(w));
        eprev = eig;
        eig   = approx_eigenvalue + sm;
        if (abs(eprev - eig) < eps * abs(eig))
            break;
    }
    if (it == max_iterations) {
        std::cout << " itmax or eps too small\n"
                  << " lack of convergence in inverse iteration\n";
    }
    return {eig, v * sm};
}
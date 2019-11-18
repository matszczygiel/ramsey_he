#pragma once

#include <array>

#include <eigen3/Eigen/Dense>

#include "definitions.h"

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

using Basis = Eigen::Matrix<scalar, Eigen::Dynamic, 3>;

Basis generate_basis(const Eigen::Matrix<scalar, m, 1>& x, int rows);

using Hamiltonian = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Overlap     = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;

std::tuple<Hamiltonian, Overlap> generate_matrices_S(const Basis& phi);
std::tuple<Hamiltonian, Overlap> generate_matrices_P(const Basis& phi);

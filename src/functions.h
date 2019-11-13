#pragma once

#include <array>

#include <eigen3/Eigen/Dense>

#include "definitions.h"

Eigen::Matrix<scalar, Eigen::Dynamic, 3> generate_wf(const Eigen::Matrix<scalar, m, 1>& x, int rows);

std::tuple<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>>
generate_matrices(
    const Eigen::Matrix<scalar, Eigen::Dynamic, 3>& phi);

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
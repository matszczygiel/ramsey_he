#pragma once

#include <eigen3/Eigen/Dense>

#include "definitions.h"

Eigen::Matrix<scalar, Eigen::Dynamic, 3> generate_wf(const Eigen::Matrix<scalar, 14, 1>& x, int rows);

std::tuple<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>>
generate_matrices(
    const Eigen::Matrix<scalar, Eigen::Dynamic, 3>& phi);
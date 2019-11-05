#pragma once 

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <eigen3/Eigen/Dense>

namespace boost {
namespace multiprecision {
using cpp_bin_float_oct = number<backends::cpp_bin_float<237, backends::digit_base_2, void, boost::int32_t, -262142, 262143>, et_off>;
}
}  // namespace boost

// floating point type
using scalar = boost::multiprecision::cpp_bin_float_oct;
//using scalar = boost::multiprecision::mpfr_float_50;

namespace Eigen {
template <>
struct NumTraits<scalar>
    : GenericNumTraits<scalar> {
};
}  // namespace Eigen
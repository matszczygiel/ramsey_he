#pragma once

#include <boost/multiprecision/cpp_bin_float.hpp>

namespace boost {
namespace multiprecision {
using cpp_bin_float_oct = number<
    backends::cpp_bin_float<237, backends::digit_base_2, void, boost::int32_t, -262142, 262143>,
    et_off>;
}
}  // namespace boost
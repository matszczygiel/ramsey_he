#include "random.h"

std::mt19937& random_engine() noexcept {
    static std::mt19937 engine{std::random_device{}()};
    return engine;
}

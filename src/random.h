#pragma once

#include <cassert>

class Ranmar {
   public:
    constexpr Ranmar(int ij, int kl) {
        assert(ij >= 0 && ij <= 31328);
        assert(kl >= 0 && kl <= 30081);

        auto i = (ij / 177) % 177 + 2;
        auto j = ij % 177 + 2;
        auto k = (kl / 169) % 178 + 1;
        auto l = kl % 169;

        for (int ii = 0; ii < 97; ++ii) {
            auto s = 0.0;
            auto t = 0.5;
            for (int jj = 0; jj < 24; ++jj) {
                const auto m = (((i * j) % 179) * k) % 179;
                i      = j;
                j      = k;
                k      = m;
                l      = (53 * l + 1) % 169;
                if ((l * m) % 64 >= 32)
                    s = s + t;
                t = 0.5 * t;
            }
            u[ii] = s;
        }
    }

    constexpr double operator()() {
        auto uni = u[i97] - u[j97];
        if (uni < 0.0)
            uni += 1.0;
        u[i97] = uni;
        if (--i97 < 0)
            i97 = 96;
        if (--j97 < 0)
            j97 = 96;
        c -= cd;
        if (c < 0.0)
            c += cm;
        uni -= c;
        if (uni < 0.0)
            uni += 1.0;
        return uni;
    }

   private:
    double u[97]{};
    double c{362436.0 / 16777216.0};
    int i97{96};
    int j97{32};

    static constexpr double cd{7654321.0 / 16777216.0};
    static constexpr double cm{16777213.0 / 16777216.0};
};
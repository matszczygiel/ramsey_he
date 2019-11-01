#include <iostream>
#include <vector>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <eigen3/Eigen/Dense>

namespace boost {
namespace multiprecision {
using cpp_bin_float_oct = number<backends::cpp_bin_float<237, backends::digit_base_2, void, boost::int32_t, -262142, 262143>, et_off>;
}
}  // namespace boost

using scalar = boost::multiprecision::cpp_bin_float_oct;

namespace Eigen {
template <>
struct NumTraits<scalar>
    : GenericNumTraits<scalar> {
};
}  // namespace Eigen

using namespace Eigen;
using namespace std;

/*
This is the initialization routine for the random_engine number generator RANMAR()
NOTE: The seed variables can have values between:    0 <= IJ <= 31328
                                                     0 <= KL <= 30081
The random_engine number sequences created by these two seeds are of sufficient 
length to complete an entire calculation with. For example, if sveral 
different groups are working on different parts of the same calculation,
each group could be assigned its own IJ seed. This would leave each group
with 30000 choices for the second seed. That is to say, this random_engine 
number generator can create 900 million different subsequences -- with 
each subsequence having a length of approximately 10^30.

Use IJ = 1802 & KL = 9373 to test the random_engine number generator. The
subroutine RANMAR should be used to generate 20000 random_engine numbers.
Then display the next six random_engine numbers generated multiplied by 4096*4096
If the random_engine number generator is working properly, the random_engine numbers
should be:
          6533892.0  14220222.0  7275067.0
          6172232.0  8354498.0   10633180.0
*/

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
                auto m = (((i * j) % 179) * k) % 179;
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

    double operator()() {
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
        uni = uni - c;
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

static Ranmar random_engine(6, 17);

void para2(Matrix<scalar, Dynamic, 3>& phi, Matrix<scalar, 14, 1>& x) {
    auto A1 = x(0);
    auto A2 = x(1);
    auto B1 = x(2);
    auto B2 = x(3);
    auto C1 = x(4);
    auto C2 = x(5);
    auto DE = x(6);

    for (int i = 0; i < phi.rows() / 2; ++i) {
        do {
            phi(i, 0) = random_engine() * (A2 - A1) + A1;
            phi(i, 1) = random_engine() * (B2 - B1) + B1;
            phi(i, 2) = random_engine() * (C2 - C1) + C1;
        } while (phi(i, 0) + phi(i, 1) < DE || phi(i, 1) + phi(i, 2) < DE || phi(i, 0) + phi(i, 2) < DE);
    }

    A1 = x(7);
    A2 = x(8);
    B1 = x(9);
    B2 = x(10);
    C1 = x(11);
    C2 = x(12);
    DE = x(13);

    for (int i = phi.rows() / 2; i < phi.rows(); ++i) {
        do {
            phi(i, 0) = random_engine() * (A2 - A1) + A1;
            phi(i, 1) = random_engine() * (B2 - B1) + B1;
            phi(i, 2) = random_engine() * (C2 - C1) + C1;
        } while (phi(i, 0) + phi(i, 1) < DE || phi(i, 1) + phi(i, 2) < DE || phi(i, 0) + phi(i, 2) < DE);
    }
}

void hamlnorm(Matrix<scalar, Dynamic, Dynamic>& hh,
              Matrix<scalar, Dynamic, Dynamic>& nn,
              Matrix<scalar, Dynamic, 3>& phi) {
    for (int i = 0; i < phi.rows(); ++i) {
        auto a1 = phi(i, 0);
        auto b1 = phi(i, 1);
        auto c1 = phi(i, 2);
        for (int j = i; j < phi.rows(); ++j) {
            auto a2 = phi(j, 0);
            auto b2 = phi(j, 1);
            auto c2 = phi(j, 2);

            auto a = 2.0 / (b1 + b2 + c1 + c2);
            auto b = 2.0 / (a1 + a2 + c1 + c2);
            auto c = 2.0 / (a1 + a2 + b1 + b2);
            auto d = 2.0 / (a1 + b2 + c1 + c2);
            auto f = 2.0 / (b1 + a2 + c1 + c2);

            auto X1  = a * b * c;
            auto X2  = a * b;
            auto X3  = a * c;
            auto X4  = b * c;
            auto X5  = X2 + X3 + X4;
            auto X6  = a * a;
            auto X7  = b * b;
            auto X8  = c * c;
            auto X9  = a + c;
            auto X10 = b + c;
            auto X11 = a + b;

            auto Y1  = c * d * f;
            auto Y2  = d * f;
            auto Y3  = d * c;
            auto Y4  = f * c;
            auto Y5  = Y2 + Y3 + Y4;
            auto Y6  = d * d;
            auto Y7  = f * f;
            auto Y8  = c * c;
            auto Y9  = d + c;
            auto Y10 = f + c;
            auto Y11 = d + f;

            const scalar Z(2.0);

            hh(i, j) = (X1 * (4.0 * X8 + 2.0 * X5 -
                              2.0 * X3 * X9 * (a2 * c1 + a1 * c2) - 2.0 * X4 * X10 * (b2 * c1 + b1 * c2) +
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

            nn(i, j) = X1 * (X6 * X10 + X7 * X9 + X8 * X11 + X1) / 64.0 +
                       Y1 * (Y8 * Y11 + Y6 * Y10 + Y7 * Y9 + Y1) / 64.0;

            hh(j, i) = hh(i, j);
            nn(j, i) = nn(i, j);
        }
    }
}

int main() {
    std::cout << std::setprecision(std::numeric_limits<scalar>::max_digits10) << scientific;

    constexpr int max_iterations = 30;
    constexpr int m              = 14;
    constexpr int n              = 100;
    const scalar en_drake("-2.903724377034119598311e+00");

    Matrix<scalar, m, 1> x;

    // for n = 1500
    /*
    x << scalar("1.2845084222741440e+00"),
        scalar("2.5525827520389550e+00"),
        scalar("1.3022061582156230e+00"),
        scalar("5.2592228607912200e+00"),
        scalar("-5.3005320302503930e-02"),
        scalar("5.0118833268966480e-01"),
        scalar("1.1664947151404250e+00"),
        scalar("2.9309131965855850e+00"),
        scalar("5.3276662855341290e+00"),
        scalar("3.6503129028788250e+00"),
        scalar("6.5776225188830830e+00"),
        scalar("-2.1506973153746800e-01"),
        scalar("1.9548131347420990e+00"),
        scalar("2.8616266805520390e+00");
*/

    // for n = 100

    x << scalar("1.2845084222741440e+00"),
        scalar("2.5525827520389550e+00"),
        scalar("1.3022061582156230e+00"),
        scalar("5.2592228607912200e+00"),
        scalar("-5.3005320302503930e-02"),
        scalar("5.0118833268966480e-01"),
        scalar("1.1664947151404250e+00"),
        scalar("2.9309131965855850e+00"),
        scalar("5.3276662855341290e+00"),
        scalar("3.6503129028788250e+00"),
        scalar("6.5776225188830830e+00"),
        scalar("-2.1506973153746800e-01"),
        scalar("1.9548131347420990e+00"),
        scalar("2.8616266805520390e+00");

    const scalar epsilon("1.0e-50");
    scalar eig  = en_drake - 1.0e-5;
    scalar eold = eig;
    scalar eprev;

    Matrix<scalar, Dynamic, 3> phi(n, 3);
    para2(phi, x);

    Matrix<scalar, Dynamic, Dynamic> dh(n, n);
    Matrix<scalar, Dynamic, Dynamic> dn(n, n);
    hamlnorm(dh, dn, phi);
    
    Matrix<scalar, Dynamic, 1> v = Matrix<scalar, Dynamic, 1>::Ones(n);
    Matrix<scalar, Dynamic, 1> w = v;
    Matrix<scalar, Dynamic, 1> u = v;

    auto dh_ldlt = dh.ldlt();

    scalar sm;
    int it = 1;
    for (; it <= max_iterations; ++it) {
        v = dh_ldlt.solve(v);
        u     = v;
        w     = dn * v;
        sm    = v.dot(w);
        sm    = 1.0 / sqrt(sm);
        eprev = eig;
        eig   = eold + sm;
        std::cout << "it = " << it << "  eig = " << eig << std::endl;
        v = w * sm;
        if (abs(eprev - eig) > epsilon * abs(eig))
            break;
    }
    if (it == max_iterations) {
        std::cout << " ITMAX OR EPS TOO SMALL\n"
                  << " LACK OF CONVERGENCE IN INVERSE ITERATION\n";
    }

    std::cout << "N   =     " << n << '\n'
              << "EIG =     " << eig << '\n';

    v = u * sm;

    std::cout << "NORM-1 =  " << v.dot(dn * v) - scalar(1.0) << '\n';
    auto est = v.dot(dh * v);
    std::cout << "EST =     " << est << '\n'
              << "EIG-EST = " << est - eig << '\n';

    /*
      nb = 32
      nbi = 32
      la = (n*(n+nb+1))/2
      ln = (n*(n+1))/2

      ALLOCATE(HH(la),DH(ln),DN(ln),U(N),V(N),W(N),PHI(N,3),d(2*n),buf(n+nb*n),perm(n))

      CALL PARA2(PHI,X,N)
      CALL HAMLNORM(DH,DN,PHI,N)
      DO I=1,N
         V(I) = one
      ENDDO
      HH(la+1-ln:la) = DH(1:ln)-eig*DN(1:ln)
!
      call ma64_factor_qd(n,n,nb,nbi,HH,la,cntl,q,ll,perm,d,buf,info)
      if(info%num_zero > 0) WRITE(*,*) "WARNING, num_zero = ", info%num_zero
!
      do it=1,ITMAX
         call hsl_leq1s(n,n,nb,v,flag,HH,ll,d,perm)
         u(1:n) = v(1:n)
         call qdspmv('l',n,one,DN,v,1,zero,w,1)
         call qddot(n,v,1,w,1,sm)
         sm = one/sqrt(sm)
         eprev = eig
         eig = eole+sm
!         WRITE(*,*) 'it=',it,'eig='
!         call QDWRITE(6,eig)
         v(1:n) = w(1:n)*sm
         if (abs(eprev-eig).lt.eps*abs(eig)) goto 100
      end do
      WRITE(*,*) 'ITMAX OR EPS TOO SMALL'
      WRITE(*,*) 'LACK OF CONVERGENCE IN INVERSE ITERATION'
  100 CONTINUE
    
      WRITE(*,*) 'N=',N
      WRITE(*,*) 'EIG='
      CALL QDWRITE(6,EIG)
!
!     EIGENVECTOR
!
      v(1:n) = sm*u(1:n)
!
!      WRITE( output_file, '( "wf_1S0_", i0,"_qu.dat" )' ) N
!      OPEN(UNIT=9,FILE=output_file,FORM='UNFORMATTED')
!      DO I=1,N
!         WRITE(9) PHI(I,1),PHI(I,2),PHI(I,3),V(I)
!      ENDDO
!      CLOSE(9)
!
!     TESTING NORM
!
      call qdspmv('l',n,one,DN,v,1,zero,w,1)
      call qddot(n,v,1,w,1,sm)
      WRITE(*,*) "NORM-1="
      sm = sm-qd_one
      call QDWRITE(6,sm)      
!
!     TESTING ENERGY
!
      call qdspmv('l',n,one,DH,v,1,zero,w,1)
      call qddot(n,v,1,w,1,est)
      WRITE(*,*) "EST="
      call QDWRITE(6,est)
      WRITE(*,*) "EIG-EST="
      call QDWRITE(6,eig-est)  
*/
}
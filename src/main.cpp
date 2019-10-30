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

/*
This is the initialization routine for the random number generator RANMAR()
NOTE: The seed variables can have values between:    0 <= IJ <= 31328
                                                     0 <= KL <= 30081
The random number sequences created by these two seeds are of sufficient 
length to complete an entire calculation with. For example, if sveral 
different groups are working on different parts of the same calculation,
each group could be assigned its own IJ seed. This would leave each group
with 30000 choices for the second seed. That is to say, this random 
number generator can create 900 million different subsequences -- with 
each subsequence having a length of approximately 10^30.

Use IJ = 1802 & KL = 9373 to test the random number generator. The
subroutine RANMAR should be used to generate 20000 random numbers.
Then display the next six random numbers generated multiplied by 4096*4096
If the random number generator is working properly, the random numbers
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

void para2(Matrix<scalar, Dynamic, 3>& phi, Matrix<scalar, 14, 1>& x) {
}
/*
     SUBROUTINE PARA2(PHI,X,N)
C
C     GENERATES INITIAL 3 N NONLINEAR PARAMETERS
C
      USE qdmodule
      INTEGER I,N    
      type(qd_real) PHI(N,3),QY(3),X(14),DE,A1,A2,B1,B2,C1,C2,T1,T2,T3
      REAL Y(3)
 
C     THE ASYMPTOTIC WAVE FUNCTION DECAYS ACCORDING TO DE=SQRT(2*E_DISSOCIATION)
C     THEREFORE NONLINEAR PARAMETERS SHOULD BE NOT MUCH SMALLER
C     DE SHOULD BE SET BY HAND FOR A CONSIDERED SYSTEM DE=SQRT(2(2-E))

      CALL RMARIN(6,17)
      A1 = X(1)
      A2 = X(2)
      B1 = X(3)
      B2 = X(4)
      C1 = X(5)
      C2 = X(6)
      DE = X(7) 
 
      DO I=1,N/2
 
 10   CONTINUE
      CALL RANMAR(Y,3)

      QY(1) = DBLE(Y(1))      
      QY(2) = DBLE(Y(2))
      QY(3) = DBLE(Y(3))      

      T1 = QY(1)*(A2-A1)+A1
      T2 = QY(2)*(B2-B1)+B1
      T3 = QY(3)*(C2-C1)+C1
 
      IF((T1+T2).LT.DE .OR. (T2+T3).LT.DE .OR. (T1+T3).LT.DE) GOTO 10

       PHI(I,1) = T1
       PHI(I,2) = T2
       PHI(I,3) = T3
      ENDDO

      A1 = X(8)
      A2 = X(9)
      B1 = X(10)
      B2 = X(11)
      C1 = X(12)
      C2 = X(13) 
      DE = X(14)
 
      DO I=N/2+1,N
 
 11      CONTINUE
      CALL RANMAR(Y,3)

      QY(1) = DBLE(Y(1))      
      QY(2) = DBLE(Y(2))
      QY(3) = DBLE(Y(3))      
    
      T1 = QY(1)*(A2-A1)+A1
      T2 = QY(2)*(B2-B1)+B1
      T3 = QY(3)*(C2-C1)+C1  
 
      IF((T1+T2).LT.DE .OR. (T2+T3).LT.DE .OR. (T1+T3).LT.DE) GOTO 11

       PHI(I,1) = T1
       PHI(I,2) = T2
       PHI(I,3) = T3
      ENDDO

       RETURN
      END
*/

int main() {
    std::cout << "Hello !\n";

    constexpr int max_iterations = 30;
    constexpr int m              = 14;
    const scalar en_drake("-2.903724377034119598311e+00");

    Matrix<scalar, m, 1> x;

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

    /*
Use IJ = 1802 & KL = 9373 to test the random number generator. The
subroutine RANMAR should be used to generate 20000 random numbers.
Then display the next six random numbers generated multiplied by 4096*4096
If the random number generator is working properly, the random numbers
should be:
          6533892.0  14220222.0  7275067.0
          6172232.0  8354498.0   10633180.0
*/
    Ranmar r(1802, 9373);
    for (int i = 0; i < 20000; ++i)
        r();

    std::cout << std::fixed;
    for (int i = 0; i < 6; ++i)
        std::cout << r() * 4096 * 4096 << "    ";
    std::cout << '\n';

    /*      nb = 32
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
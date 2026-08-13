//
// Created by lara on 10/9/25.
//

#ifndef SUBWAVELENGTHRESONATORS_TOOLS_H
#define SUBWAVELENGTHRESONATORS_TOOLS_H


#include <cassert>
#include <iostream>
#include <fstream>
#include <complex>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "Eigen/Dense"
#include "unsupported/Eigen/FFT"
#if __has_include(<omp.h>)
#include <omp.h>
#endif
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif
#include "math_constants.h"
//#include "fftw3.h"
//#include <limits>

using namespace Eigen;
using namespace std;
using cpxd = complex<double>;

struct Result {
    complex<double> root;
    complex<double> f_at_root;
    int iterations;
    bool converged;
};

/*inline void do_fft_fftw(const VectorXcd& fft_f, VectorXcd& temp) {
    const int N = static_cast<int>(fft_f.size()); // should be 2*M

    // Make sure output has correct size
    temp.resize(N);

    // FFTW works with fftw_complex = double[2]
    // We can safely reinterpret Eigen's std::complex<double> data
    fftw_complex* in  = reinterpret_cast<fftw_complex*>(
            const_cast<std::complex<double>*>(fft_f.data())
    );
    fftw_complex* out = reinterpret_cast<fftw_complex*>(temp.data());

    // Create plan (you probably want to cache/reuse this in real code)
    fftw_plan plan = fftw_plan_dft_1d(
            N,
            in,
            out,
            FFTW_FORWARD,
            FFTW_ESTIMATE   // or FFTW_MEASURE if you don't mind a slower setup
    );

    fftw_execute(plan);
    fftw_destroy_plan(plan);
}*/

namespace Tools {

    /** @brief True when stdout is a TTY so in-place progress rendering is safe. */
    inline bool stdout_is_tty() {
#if defined(_WIN32)
        static const bool tty = (_isatty(_fileno(stdout)) != 0);
#else
        static const bool tty = (::isatty(STDOUT_FILENO) != 0);
#endif
        return tty;
    }

    /** @brief Ends an in-place progress line with a newline in TTY mode. */
    inline void finish_progress_line() {
        if (stdout_is_tty()) {
            std::cout << '\n';
        }
    }

    /** @brief Compact stage progress indicator with TTY and non-TTY fallbacks. */
    inline void print_stage_progress(const std::string& label, int done, int total) {
        if (total <= 0) {
            return;
        }
        done = std::clamp(done, 0, total);

        if (!stdout_is_tty()) {
            const int step = std::max(1, total / 20);
            if (done % step == 0 || done == total) {
                std::cout << label << ": " << done << "/" << total << std::endl;
            }
            return;
        }

        const int width = 30;
        const double progress = static_cast<double>(done) / static_cast<double>(total);
        const int pos = static_cast<int>(progress * width);

        std::cout << "\r\033[2K" << label << " [";
        for (int i = 0; i < width; ++i) {
            std::cout << (i < pos ? '=' : ' ');
        }
        std::cout << "] " << int(progress * 100.0) << "%" << std::flush;
    }

    /**
     * @brief Find a complex root using Muller's method.
     * @return Root estimate, residual, iteration count, and convergence flag.
     */
    inline Result MullersMethod(const function<complex<double>(complex<double>)>& f, complex<double> z0, complex<double> z1,
                         complex<double> z2, int iter, double tolf, double tolz) {
        complex<double> f2 = f(z2);
        complex<double> f1 = f(z1);
        complex<double> f0 = f(z0);
        complex<double> f3, z3;

        for (int i = 0; i < iter; i++) {
            complex<double> q = (z2 - z1) / (z1 - z0);
            complex<double> a = q * f2 - q * (1. + q) * f1 + q * q * f0;
            complex<double> b = (2. * q + 1.) * f2 - (1. + q) * (1. + q) * f1 + q * q * f0;
            complex<double> c = (1. + q) * f2;

            complex<double> D = sqrt(b * b - 4. * a * c);
            z3 = z2 - (z2 - z1) * (2. * c) / (b + (abs(b + D) > abs(b - D) ? D : -D));

            f3 = f(z3);
            if (abs(f3) < tolf || abs(z3 - z2) < tolz) {
                return Result{z3, f3, i + 1, true};
            }
            z0 = z1;
            z1 = z2;
            z2 = z3;

            f0 = f1;
            f1 = f2;
            f2 = f3;
        }

        // cout << "Muller's method did not converge within the maximum number of iterations." << endl;
        return Result{z3, f3, iter, false};
    }

    /**
     * @brief Find a real root using the secant method.
     * @return Pair of final iterate and function value at that iterate.
     */
    inline pair<double,double> secant(function<double(double)> f, double x0, double x1, int iter, double tol) {\
        double x2, fof1;
        for (int i = 0; i < iter; i++) {
            fof1 = f(x1);
            if (abs(fof1) < tol) {
                cout << "Secant found the zero with " << i << " iterations." << endl;
                return make_pair(x1, fof1);
            }
            x2 = x1 - fof1 * (x1 - x0) / (fof1 - f(x0));
            x0 = x1;
            x1 = x2;
        }
        cout << "Maxiter reached!" << endl;
        return make_pair(x1, fof1);
    }

    // /* Exponential integral E1 for real argument x > 0
    //  * from Abramowitz and Stegun, Handbook of Mathematical Functions
    //  * 5.1.53 and 5.1.56
    //  * error less than 2e-7
    //  */

    /** @brief Exponential integral E1(x) for x > 0. */
    inline double E1(double x) {
        return -std::expint(-x);
        assert(("E1 is only defined for x > 0!", x > 0));

        // if (0 <= x && x <= 1) {
        //     std::array<double, 6> a = {-0.57721566,
        //                           0.99999193,
        //                           -0.24991055,
        //                           0.05519968,
        //                           -0.00976004,
        //                           0.00107857};

        //     double sum = a[5];

        //     // Horner's rule
        //     for (int i = 4; i >= 0; i--) {
        //         sum = sum * x + a[i];
        //     }

        //     return -std::log(x) + sum;

        // } else {
        //     std::array<double, 4> a = {8.5733287401,
        //                           18.0590169730,
        //                           8.6347608925,
        //                           0.2677737343};

        //     std::array<double, 4> b = {9.5733223454,
        //                           25.6329561486,
        //                           21.0996530827,
        //                           3.9584969228};
        //     double numerator = 1.0;
        //     double denominator = 1.0;

        //     // Horner's rule
        //     for (int i = 0; i < 4; i++) {
        //         numerator = numerator * x + a[i];
        //         denominator = denominator * x + b[i];
        //     }

        //     return (exp(-x) * numerator) / (denominator * x);
        // }
    }

    /** @brief Exponential integral E0(x) = exp(-x)/x for x != 0. */
    inline double E0(double x) {
        assert(x != 0 && "E0 is only defined for x != 0!");
        return exp(-x) / x;
    }

    /** @brief Recurrence-based generalized exponential integral helper Eq(x,q). */
    inline double Eq(double x, int q) {
        if (q == 0) {
            return exp(-x) / x;
        } else if (q == 1) {
            return E1(x);
        } else {
            return (exp(-x) - x * Eq(x, q - 1)) / q;
        }
    }

    /* @brief Evaluate the complex Faddeeva function w(z).
    * Implementation of the Faddeeva function based on the paper:
    * J. A. C. Weideman, SIAM J. Numer. Anal. 31, (1994).
    * Optimized and thread-safe */
    inline std::complex<double> faddeeva(const std::complex<double> &z) {
        using namespace std;

        // 1. Asymptotic expansion for large |z| to prevent cancellation
        if (abs(z) > 8.0) {
            complex<double> I(0, 1.0);
            complex<double> z2 = z * z;
            return (I / (sqrt(M_PI) * z)) * (1.0 + 0.5/z2 + 0.75/(z2*z2) + 1.875/(z2*z2*z2));
        }

        // Symmetry mapping for lower half-plane
        if (z.imag() < 0) {
            return 2.0 * exp(-z * z) - faddeeva(-z);
        }

        // 2. Thread-Safe STATIC Initialization using a hidden struct
        struct FaddeevaCoeffs {
            Eigen::VectorXd a;
            double l;

            FaddeevaCoeffs() {
                const int N = 32;
                const int M = 2 * N;
                l = sqrt(N * sqrt(0.5));

                Eigen::VectorXd fft_f = Eigen::VectorXd::Zero(2 * M);
                for (int n = 0; n < M; n++) {
                    double t = l * tan(n * M_PI / M * 0.5);
                    fft_f(n) = exp(-t * t) * (l * l + t * t);
                    if (n > 0) {
                        fft_f(2 * M - n) = fft_f(n); // Even symmetry
                    }
                }

                Eigen::FFT<double> fft;
                Eigen::VectorXcd temp(2 * M);
                fft.fwd(temp, fft_f);

                Eigen::VectorXcd F = temp / static_cast<double>(2 * M);
                a = F.segment(1, N).real();
            }
        };

        // The C++ compiler guarantees this constructor runs exactly ONCE, safely.
        // All other threads will wait here until the first thread finishes.
        static const FaddeevaCoeffs coeffs;

        // 3. Weideman's Evaluation (Safe for |z| <= 8)
        const int N = 32;
        complex<double> I(0, 1.0);
        complex<double> sum = 0.0;

        // Use the safely precomputed coeffs.l and coeffs.a
        complex<double> Z = (coeffs.l + I * z) / (coeffs.l - I * z);

        for (int n = N - 1; n >= 0; n--) {
            sum = Z * sum + coeffs.a(n); // Horner's rule
        }

        complex<double> w = 2.0 * sum / (coeffs.l - I * z) + sqrt(M_1_PI);
        return w / (coeffs.l - I * z);
    }

    /** @brief Complex error function erf(z). */
    inline complex<double> erf_complex(const complex<double> &z) {
        return 1.0 - exp(-z * z) * faddeeva(complex<double>(0, 1.0) * z);
    }

    /** @brief Complex complementary error function erfc(z).
     * The Faddeeva identity erfc(z) = exp(-z^2) w(iz) is evaluated only for Re(z) >= 0, where
     * w's large-|z| asymptotics are valid (Im(iz) >= 0) and exp(-z^2) cannot overflow; the left
     * half-plane is reached through the reflection erfc(z) = 2 - erfc(-z). Without the
     * reflection, arguments with Re(z) < 0 and |z| > 8 lost the dominant term (erfc(-large) ~ 2
     * came out as ~0), which zeroed the propagating far-field of the periodic kernels for any
     * Im(k) != 0. */
    inline complex<double> erfc_complex(const complex<double> &z) {
        if (z.real() < 0.0) {
            return 2.0 - erfc_complex(-z);
        }
        return exp(-z * z) * faddeeva(complex<double>(0, 1.0) * z);
    }

    /** @brief Derivative of complex complementary error function approximation used by kernels. */
    inline complex<double> derfc_complex(const complex<double> &z) {
        return -exp(-z * z) * 2.0 * sqrt(M_1_PI);
    }

    /**
     * @brief Linear operator wrapper that applies solve(A, x) for eigen-solvers.
     */
    class ShiftInvertOp
    {
        using Complex = std::complex<double>;
        using MatrixXc = Eigen::MatrixXcd;
        using VectorXc = Eigen::VectorXcd;
    public:
        using Scalar = Complex;

        /** @brief Factorize matrix for repeated shift-invert solves. */
        ShiftInvertOp(const MatrixXc& A)
        {
            lu.compute(A);
        }

        /** @brief Number of rows in the linear operator. */
        int rows() const { return lu.matrixLU().rows(); }
        /** @brief Number of columns in the linear operator. */
        int cols() const { return lu.matrixLU().cols(); }

        // Spectra interface: y_out = A^{-1} * x_in
        /** @brief Apply y = A^{-1}x using cached LU factorization. */
        void perform_op(const Complex* x_in, Complex* y_out) const
        {
            Eigen::Map<const VectorXc> x(x_in, rows());
            Eigen::Map<VectorXc>       y(y_out, rows());
            y = lu.solve(x);
        }

    private:
        Eigen::PartialPivLU<MatrixXc> lu;
    };

    /** @brief Write numeric table data to a CSV file. */
    inline static void dump_csv(const std::string& fname,
                         const std::vector<std::vector<double>>& A)
    {
        std::ofstream out(fname);
        out.setf(std::ios::scientific);
        out.precision(16);
        for (const auto& row : A) {
            for (size_t j=0;j<row.size();++j) {
                if (j) out << ",";
                out << row[j];
            }
            out << "\n";
        }
        std::cerr << "[wrote] " << fname << " (" << A.size() << "x" << (A.empty()?0: A[0].size()) << ")\n";
    }
}


#endif //SUBWAVELENGTHRESONATORS_TOOLS_H

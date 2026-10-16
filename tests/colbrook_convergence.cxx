#include <gtest/gtest.h>
#include "workflows/colbrook_convergence.h"
#include <limits>
#include <stdexcept>

TEST(ColbrookConvergence, RefinesKnownMinimumWithRectangularEmbedding) {
    // Orthogonal columns: near target, sigma(z)^2=(z-exact_z)^2+0.03^2+0.04^2.
    const double exact_z = 1.731234567;
    Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(4, 2);
    C(0, 0) = 0.03;
    C(1, 0) = std::complex<double>(exact_z, 0.04);
    C(2, 1) = 4.2;
    C(3, 1) = 0.2;
    const auto coarse = workflows::find_colbrook_minimum(C, 1, 1.75, 0.2, 20, 1);
    const auto fine = workflows::find_colbrook_minimum(C, 1, 1.75, 0.2, 20, 4);
    EXPECT_NEAR(fine.z, exact_z, fine.z_resolution);
    EXPECT_NEAR(fine.sigma_min, 0.05, 1e-8);
    EXPECT_LE(fine.sigma_min, coarse.sigma_min);
    EXPECT_LT(fine.z_resolution, coarse.z_resolution);
    EXPECT_FALSE(fine.at_boundary);
}

TEST(ColbrookConvergence, FlagsMinimumOutsideSearchInterval) {
    Eigen::MatrixXcd C = Eigen::MatrixXcd::Constant(1, 1, 2.0);
    const auto result = workflows::find_colbrook_minimum(C, 0, 1.0, 0.2, 20, 3);
    EXPECT_DOUBLE_EQ(result.z, 1.2);
    EXPECT_NEAR(result.sigma_min, 0.8, 1e-14);
    EXPECT_TRUE(result.at_boundary);
}

TEST(ColbrookConvergence, RejectsInvalidEmbeddingAndSearch) {
    Eigen::MatrixXcd C = Eigen::MatrixXcd::Identity(4, 2);
    EXPECT_THROW(workflows::find_colbrook_minimum(C, 3, 1.0, 0.2, 20, 3), std::invalid_argument);
    EXPECT_THROW(workflows::find_colbrook_minimum(C, 0, 1.0, 0.0, 20, 3), std::invalid_argument);
    EXPECT_THROW(workflows::find_colbrook_minimum(C, 0, 1.0, 0.2, 3, 3), std::invalid_argument);
    C(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(workflows::find_colbrook_minimum(C, 0, 1.0, 0.2, 20, 3), std::invalid_argument);
}

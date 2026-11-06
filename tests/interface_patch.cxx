#include <gtest/gtest.h>
#include "workflows/interface_patch.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

using namespace workflows;
using namespace Eigen;

namespace {
InterfacePatch diagonal_templates(int modes, int n_up, int n_down) {
    InterfacePatch ip;
    ip.modes = modes;
    ip.patch_radius = 2;
    ip.colbrook = 1;
    ip.window_radius = 1;
    ip.fringe = 0;
    ip.disks_up.resize(n_up);
    ip.disks_down.resize(n_down);
    const int R = 1;
    for (int t = 0; t < 2 * R + 2; ++t) {
        std::vector<RowTarget> targets;
        for (int x = -R; x <= R; ++x)
            for (int y = -R; y <= R; ++y)
                for (int a = 0; a < (y <= t - R - 1 ? n_down : n_up); ++a)
                    targets.push_back({x, y, a});
        const int count = 0 <= t - R - 1 ? n_down : n_up;
        MatrixXcd C = MatrixXcd::Zero(modes * targets.size(), modes * count);
        for (std::size_t q = 0; q < targets.size(); ++q) {
            const auto& target = targets[q];
            if (target.local_x == 0 && target.local_y == 0)
                C.block(modes * q, modes * target.disk_number, modes, modes) =
                    3.25 * MatrixXcd::Identity(modes, modes);
        }
        ip.patch_colC.push_back(C);
        ip.row_offsets.push_back(targets);
    }
    return ip;
}
}

TEST(InterfacePatch, RectangularShiftMatchesPhysicalDofsWithUnequalCells) {
    for (int modes : {1, 2}) {
        for (const auto counts : {std::pair{2, 1}, std::pair{1, 2}}) {
            auto ip = diagonal_templates(modes, counts.first, counts.second);
            const auto layout = make_interface_layout(ip);
            const auto C = assemble_interface_matrix(ip);
            // Independently enumerate physical disk/mode labels, then match labels across domains.
            using Label = std::tuple<int, int, int, int>;
            const auto labels = [&](int radius) {
                std::vector<Label> result;
                for (int x = -radius; x <= radius; ++x)
                    for (int y = -radius; y <= radius; ++y)
                        for (int a = 0; a < (y < 0 ? counts.second : counts.first); ++a)
                            for (int s = 0; s < modes; ++s) result.emplace_back(x, y, a, s);
                return result;
            };
            const auto rows = labels(3), cols = labels(2);
            ASSERT_EQ(C.rows(), rows.size());
            ASSERT_EQ(C.cols(), cols.size());
            bool has_gap = false;
            for (Eigen::Index j = 0; j < C.cols(); ++j) {
                const auto match = std::find(rows.begin(), rows.end(), cols[j]) - rows.begin();
                EXPECT_EQ(layout.source_rows[j], match);
                if (j && match != layout.source_rows[j - 1] + 1) has_gap = true;
                for (Eigen::Index i = 0; i < C.rows(); ++i)
                    EXPECT_EQ(C(i, j), std::complex<double>(i == match ? 3.25 : 0.0));
            }
            EXPECT_TRUE(has_gap);
            MatrixXcd shifted = C;
            for (Eigen::Index j = 0; j < C.cols(); ++j) shifted(layout.source_rows[j], j) -= 3.42;
            JacobiSVD<MatrixXcd> svd(shifted);
            ASSERT_EQ(svd.info(), Success);
            EXPECT_TRUE(svd.singularValues().isApprox(VectorXd::Constant(C.cols(), 0.17), 1e-12));
        }
    }
}

TEST(InterfacePatch, DifferentVerticalLatticesShareConsistentGlobalGeometry) {
    const LatticeVectors up{Vector2d(1, 0), Vector2d(0.2, 1)};
    const LatticeVectors down{Vector2d(1, 0), Vector2d(0.4, -1.3)};
    const std::vector<Disk> upper{{0.07, Vector2d(0.3, 0.3), 1.0}};
    const std::vector<Disk> lower{{0.06, Vector2d(0.2, -0.3), 0.8},
                                  {0.06, Vector2d(0.65, -0.5), 1.2}};
    const auto ip = build_interface_patch(1, 1, 1, up, down, 11, upper, lower,
                                           0, 16, 0.8, 1.0, 3, false);
    ASSERT_EQ(ip.meshes.size(), 4);
    ASSERT_EQ(ip.solve_diagnostics.size(), ip.meshes.size());
    for (const auto& d : ip.solve_diagnostics) {
        EXPECT_TRUE(std::isfinite(d.rcond));
        EXPECT_GT(d.rcond, 0);
        EXPECT_LE(d.rcond, 1);
        EXPECT_LT(d.relative_residual, 1e-10);
        EXPECT_LT(d.backward_error, 1e-12);
        EXPECT_TRUE(std::isfinite(d.density_norm));
        EXPECT_GT(d.density_norm, 0);
    }
    EXPECT_TRUE(ip.bigC.allFinite());
    for (int t = 0; t < 4; ++t) {
        EXPECT_TRUE(ip.patch_X[t].allFinite());
        const int h = 1 - t;
        const Vector2d source_origin = h >= 0 ? Vector2d(0.2 * h, h)
                                              : Vector2d(0.4 * (-h - 1), -1.3 * (-h - 1));
        for (std::size_t q = 0; q < ip.row_offsets[t].size(); ++q) {
            const auto target = ip.row_offsets[t][q];
            const int gy = h + target.local_y;
            const int gx = target.local_x;
            const auto& d = gy < 0 ? lower[target.disk_number] : upper[target.disk_number];
            const Vector2d global = gy >= 0 ? Vector2d(gx + 0.2 * gy, gy)
                                           : Vector2d(gx + 0.4 * (-gy - 1), -1.3 * (-gy - 1));
            EXPECT_TRUE((ip.disks[t][q].c + source_origin).isApprox(global + d.c, 1e-12));
            for (std::size_t j = 0; j < q; ++j)
                EXPECT_GT((ip.disks[t][q].c - ip.disks[t][j].c).norm(), ip.disks[t][q].r + ip.disks[t][j].r);
        }
    }
}

TEST(InterfacePatch, ReflectedCellProducesFiniteTemplatesAndNoCoincidentDisks) {
    const double h = std::sqrt(3.0);
    const LatticeVectors up{Vector2d(1, 0), Vector2d(0.5, h / 2)};
    const LatticeVectors down{Vector2d(1, 0), Vector2d(0.5, -h / 2)};
    std::vector<Disk> upper{{0.1, Vector2d(0.5, h / 6), 1}, {0.2, Vector2d(1, h / 3), 0.5}};
    auto lower = upper;
    for (auto& d : lower) d.c.y() = -d.c.y();
    const auto ip = build_interface_patch(1, 1, 1, up, down, 12, upper, lower, 0, 16, 0.35, 1, 3, false);
    EXPECT_TRUE(ip.bigC.allFinite());
    for (const auto& X : ip.patch_X) EXPECT_TRUE(X.allFinite());
    EXPECT_TRUE(interface_cell_origin(ip, 0, 0).isZero());
    EXPECT_TRUE(interface_cell_origin(ip, 0, -1).isZero());
    EXPECT_TRUE(interface_cell_origin(ip, 0, -2).isApprox(down.a2));
    // Reusing upper centers below the interface makes the two first rows coincide.
    EXPECT_THROW(build_interface_patch(1, 1, 1, up, down, 12, upper, upper, 0, 16, 0.35, 1, 3, false),
                 std::invalid_argument);
    upper[1].c = Vector2d(0.25, h / 12); // The original overlapping cell.
    EXPECT_THROW(build_interface_patch(1, 1, 1, up, down, 12, upper, lower, 0, 16, 0.35, 1, 3, false),
                 std::invalid_argument);
}

TEST(InterfacePatch, RejectsNonfiniteTemplateBeforeAssembly) {
    auto ip = diagonal_templates(2, 2, 1);
    ip.patch_colC[1](0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(assemble_interface_matrix(ip), std::invalid_argument);
}

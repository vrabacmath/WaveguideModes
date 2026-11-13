#pragma once

#include <string>
#include "workflows/interface_patch.h"

namespace workflows {
// Save unscaled G0 fields, one column per central disk/angular basis.
// Reconstruct a mode using source_speed * eigenvector coefficient.
void run_interface_field(const InterfacePatch& p, int grid_points,
                         const std::string& prefix = "tessellated_interface");
}

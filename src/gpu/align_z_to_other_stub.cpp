//
// Created by konrad on 10/23/17.
//

#include "../../include/librealsense/rs.h"
#include "../types.h"
#include <iostream>
#include <cmath>
#include <mutex>

#ifndef DEBUG
#define DEBUG(x) do {} while(0);
#else
#define DEBUG(x) do { std::cerr << x << std::endl; } while(0);
#endif

namespace gpu {
bool align_z_to_other(rsimpl::byte * z_aligned_to_other, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin) {
    DEBUG("GPU DISABLED");
    return false;
}
}



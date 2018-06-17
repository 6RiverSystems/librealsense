#pragma once
#ifndef LIBREALSENSE2_CUDA_CONVERSION_H
#define LIBREALSENSE2_CUDA_CONVERSION_H

#ifdef USE_CUDA

// Types
#include <stdint.h>
#include "../include/librealsense2/rs.h"
#include "../assert.h"
#include "../types.h"

// CUDA headers
#include <cuda_runtime.h>

#ifdef _MSC_VER 
// Add library dependencies if using VS
#pragma comment(lib, "cudart_static")
#endif

void unpack_yuy2_rgb8_cuda(const uint8_t* src, uint8_t* dst, int n);
void unpack_yuy2_rgb8a_cuda(const uint8_t* src, uint8_t* dst, int n);
void unpack_yuy2_bgr8_cuda(const uint8_t* src, uint8_t* dst, int n);
void unpack_yuy2_bgr8a_cuda(const uint8_t* src, uint8_t* dst, int n);

#endif // USE_CUDA

#endif // LIBREALSENSE2_CUDA_CONVERSION_H

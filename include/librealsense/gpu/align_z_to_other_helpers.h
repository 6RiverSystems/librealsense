//
// Created by konrad on 10/23/17.
//

#ifndef PROJECT_ALIGN_Z_TO_OTHER_H
#define PROJECT_ALIGN_Z_TO_OTHER_H

#include "rs.h"
#include <assert.h>

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

#else

namespace gpu {
    /* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
    __device__ inline void rs_project_point_to_pixel(float pixel[2], const struct rs_intrinsics * intrin, const float point[3])
    {
        assert(intrin->model != RS_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image
        assert(intrin->model != RS_DISTORTION_FTHETA); // Cannot project to an ftheta image

        float multiplier = 1 / point[2];
        float x = point[0] * multiplier, y = point[1] * multiplier;
        if(intrin->model == RS_DISTORTION_MODIFIED_BROWN_CONRADY)
        {
            float r2  = x*x + y*y;

            float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
            x *= f;
            y *= f;
            float dx = x + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
            float dy = y + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
            x = dx;
            y = dy;
        }
        pixel[0] = x * intrin->fx + intrin->ppx;
        pixel[1] = y * intrin->fy + intrin->ppy;
    }

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */

    __device__ inline void rs_deproject_pixel_to_point(float point[3], const struct rs_intrinsics * intrin, const float pixel[2], float depth, float invfx = 0.0, float invfy = 0.0)
    {
        assert(intrin->model != RS_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
        assert(intrin->model != RS_DISTORTION_FTHETA); // Cannot deproject to an ftheta image

        float x = 0;
        if (invfx == 0.0) {
            x = (pixel[0] - intrin->ppx) / intrin->fx;
        } else {
            x = (pixel[0] - intrin->ppx) * invfx;
        }
        float y = 0;
        if (invfy == 0.0) {
            y = (pixel[1] - intrin->ppy) / intrin->fy;
        } else {
            y = (pixel[1] - intrin->ppy) * invfy;
        }
        if(intrin->model == RS_DISTORTION_INVERSE_BROWN_CONRADY)
        {
            float r2  = x * x + y * y;
            float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
            float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
            float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
            x = ux;
            y = uy;
        }
        point[0] = depth * x;
        point[1] = depth * y;
        point[2] = depth;
    }

/* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */

    __device__ inline void rs_transform_point_to_point(float to_point[3], const struct rs_extrinsics * extrin, const float from_point[3])
    {
        to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] + extrin->rotation[6] * from_point[2] + extrin->translation[0];
        to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[7] * from_point[2] + extrin->translation[1];
        to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
    }
};

#endif

#endif //PROJECT_ALIGN_Z_TO_OTHER_H

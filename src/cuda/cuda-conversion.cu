#ifdef USE_CUDA


#include <stdint.h>
#include <assert.h>

#define RS_CUDA_THREADS_PER_BLOCK 16

// YUYV
// Each four bytes is two pixels. Each four bytes is two Y's, a Cb and a Cr. 
// Each Y goes to one of the pixels, and the Cb and Cr belong to both pixels.
// Also known in Windows as YUY2

__global__ void kernel_unpack_yuy2_rgb8_cuda(const uint8_t * src, uint8_t *dst, int superPixCount)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= superPixCount)
        return;

    int idx = i * 4;

    uint8_t y0 = src[idx];
    uint8_t u0 = src[idx + 1];
    uint8_t y1 = src[idx + 2];
    uint8_t v0 = src[idx + 3];

    int16_t luma = y0 - 16;
    int16_t chromaCb = u0 - 128;
    int16_t chromaCr = v0 - 128;

    int32_t t;
#define clamp(x)  ((t=(x)) > 255 ? 255 : t < 0 ? 0 : t)

    int odx = i * 6;

    dst[odx]     = clamp((298 * luma + 409 * chromaCr + 128) >> 8);
    dst[odx + 1] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 2] = clamp((298 * luma + 516 * chromaCb + 128) >> 8);

    luma = y1 - 16;

    dst[odx + 3] = clamp((298 * luma + 409 * chromaCr + 128) >> 8);
    dst[odx + 4] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 5] = clamp((298 * luma + 516 * chromaCb + 128) >> 8);

#undef clamp
}

__global__ void kernel_unpack_yuy2_rgb8a_cuda(const uint8_t * src, uint8_t *dst, int superPixCount)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= superPixCount)
        return;

    int idx = i * 4;

    uint8_t y0 = src[idx];
    uint8_t u0 = src[idx + 1];
    uint8_t y1 = src[idx + 2];
    uint8_t v0 = src[idx + 3];

    int16_t luma = y0 - 16;
    int16_t chromaCb = u0 - 128;
    int16_t chromaCr = v0 - 128;

    int32_t t;
#define clamp(x)  ((t=(x)) > 255 ? 255 : t < 0 ? 0 : t)

    int odx = i * 6;

    dst[odx]     = clamp((298 * luma + 409 * chromaCr + 128) >> 8);
    dst[odx + 1] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 2] = clamp((298 * luma + 516 * chromaCb + 128) >> 8);
    dst[odx + 3] = 255 ;

    luma = y1 - 16;

    dst[odx + 4] = clamp((298 * luma + 409 * chromaCr + 128) >> 8);
    dst[odx + 5] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 6] = clamp((298 * luma + 516 * chromaCb + 128) >> 8);
    dst[odx + 7] = 255 ;

#undef clamp
}


__global__ void kernel_unpack_yuy2_bgr8_cuda(const uint8_t * src, uint8_t *dst, int superPixCount)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= superPixCount)
        return;

    int idx = i * 4;

    uint8_t y0 = src[idx];
    uint8_t u0 = src[idx + 1];
    uint8_t y1 = src[idx + 2];
    uint8_t v0 = src[idx + 3];

    int16_t luma = y0 - 16;
    int16_t chromaCb = u0 - 128;
    int16_t chromaCr = v0 - 128;

    int32_t t;
#define clamp(x)  ((t=(x)) > 255 ? 255 : t < 0 ? 0 : t)

    int odx = i * 8;

    dst[odx]     = clamp((298 * luma + 516 * chromaCb + 128) >> 8);
    dst[odx + 1] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 2] = clamp((298 * luma + 409 * chromaCr + 128) >> 8);

    luma = y1 - 16;

    dst[odx + 3] = clamp((298 * luma + 516 * chromaCb + 128) >> 8);
    dst[odx + 4] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 5] = clamp((298 * luma + 409 * chromaCr + 128) >> 8);

#undef clamp
}

__global__ void kernel_unpack_yuy2_bgr8a_cuda(const uint8_t * src, uint8_t *dst, int superPixCount)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= superPixCount)
        return;

    int idx = i * 4;

    uint8_t y0 = src[idx];
    uint8_t u0 = src[idx + 1];
    uint8_t y1 = src[idx + 2];
    uint8_t v0 = src[idx + 3];

    int16_t luma = y0 - 16;
    int16_t chromaCb = u0 - 128;
    int16_t chromaCr = v0 - 128;

    int32_t t;
#define clamp(x)  ((t=(x)) > 255 ? 255 : t < 0 ? 0 : t)

    int odx = i * 8;

    dst[odx]     = clamp((298 * luma + 516 * chromaCb + 128) >> 8);
    dst[odx + 1] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 2] = clamp((298 * luma + 409 * chromaCr + 128) >> 8);
    dst[odx + 3] = 255 ;

    luma = y1 - 16;

    dst[odx + 4] = clamp((298 * luma + 516 * chromaCb + 128) >> 8);
    dst[odx + 5] = clamp((298 * luma - 100 * chromaCb - 409 * chromaCr + 128) >> 8);
    dst[odx + 6] = clamp((298 * luma + 409 * chromaCr + 128) >> 8);
    dst[odx + 7] = 255 ;

#undef clamp
}


void unpack_yuy2_rgb8_cuda(const uint8_t* src, uint8_t* dst, int n)
{
    // How many super pixels do we have?
    int superPix = n / 2;
    uint8_t *devSrc = nullptr;
    uint8_t *devDst = nullptr;

    cudaError_t result = cudaMalloc(&devSrc, superPix * sizeof(uint8_t) * 4);
    assert(result == cudaSuccess);

    result = cudaMalloc(&devDst, n * sizeof(uint8_t) * 3);
    assert(result == cudaSuccess);

    result = cudaMemcpy(devSrc, src, superPix * sizeof(uint8_t) * 4, cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    int numBlocks = superPix / RS_CUDA_THREADS_PER_BLOCK;

    // Call the kernel
    kernel_unpack_yuy2_rgb8_cuda<<<numBlocks, RS_CUDA_THREADS_PER_BLOCK >>>(devSrc, devDst, superPix);
    result = cudaGetLastError();
    assert(result == cudaSuccess);

    // Copy back
    result = cudaMemcpy(dst, devDst, n * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    cudaFree(devSrc);
    cudaFree(devDst);
}

void unpack_yuy2_rgb8a_cuda(const uint8_t* src, uint8_t* dst, int n)
{
    // How many super pixels do we have?
    int superPix = n / 2;
    uint8_t *devSrc = nullptr;
    uint8_t *devDst = nullptr;

    cudaError_t result = cudaMalloc(&devSrc, superPix * sizeof(uint8_t) * 4);
    assert(result == cudaSuccess);

    result = cudaMalloc(&devDst, n * sizeof(uint8_t) * 4);
    assert(result == cudaSuccess);

    result = cudaMemcpy(devSrc, src, superPix * sizeof(uint8_t) * 4, cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    int numBlocks = superPix / RS_CUDA_THREADS_PER_BLOCK;

    // Call the kernel
    kernel_unpack_yuy2_rgb8a_cuda<<<numBlocks, RS_CUDA_THREADS_PER_BLOCK >>>(devSrc, devDst, superPix);
    result = cudaGetLastError();
    assert(result == cudaSuccess);

    // Copy back
    result = cudaMemcpy(dst, devDst, n * sizeof(uint8_t) * 4, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    cudaFree(devSrc);
    cudaFree(devDst);
}

void unpack_yuy2_bgr8_cuda(const uint8_t* src, uint8_t* dst, int n)
{
    // How many super pixels do we have?
    int superPix = n / 2;
    uint8_t *devSrc = nullptr;
    uint8_t *devDst = nullptr;

    cudaError_t result = cudaMalloc(&devSrc, superPix * sizeof(uint8_t) * 4);
    assert(result == cudaSuccess);

    result = cudaMalloc(&devDst, n * sizeof(uint8_t) * 3);
    assert(result == cudaSuccess);

    result = cudaMemcpy(devSrc, src, superPix * sizeof(uint8_t) * 4, cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    int numBlocks = superPix / RS_CUDA_THREADS_PER_BLOCK;

    // Call the kernel
    kernel_unpack_yuy2_bgr8_cuda<<<numBlocks, RS_CUDA_THREADS_PER_BLOCK >>>(devSrc, devDst, superPix);
    result = cudaGetLastError();
    assert(result == cudaSuccess);

    // Copy back
    result = cudaMemcpy(dst, devDst, n * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    cudaFree(devSrc);
    cudaFree(devDst);
}


void unpack_yuy2_bgr8a_cuda(const uint8_t* src, uint8_t* dst, int n)
{
    // How many super pixels do we have?
    int superPix = n / 2;
    uint8_t *devSrc = nullptr;
    uint8_t *devDst = nullptr;

    cudaError_t result = cudaMalloc(&devSrc, superPix * sizeof(uint8_t) * 4);
    assert(result == cudaSuccess);

    result = cudaMalloc(&devDst, n * sizeof(uint8_t) * 4);
    assert(result == cudaSuccess);

    result = cudaMemcpy(devSrc, src, superPix * sizeof(uint8_t) * 4, cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    int numBlocks = superPix / RS_CUDA_THREADS_PER_BLOCK;

    // Call the kernel
    kernel_unpack_yuy2_bgr8a_cuda<<<numBlocks, RS_CUDA_THREADS_PER_BLOCK >>>(devSrc, devDst, superPix);
    result = cudaGetLastError();
    assert(result == cudaSuccess);

    // Copy back
    result = cudaMemcpy(dst, devDst, n * sizeof(uint8_t) * 4, cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    cudaFree(devSrc);
    cudaFree(devDst);
}

#endif

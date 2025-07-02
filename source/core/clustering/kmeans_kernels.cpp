/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "aoclda_types.h"
#include "da_kernel_utils.hpp"
#include "immintrin.h"
#include "kmeans.hpp"
#include "macros.h"
namespace ARCH {

namespace da_kmeans {

/* These functions contain performance-critical loops which must vectorize for performance. */

/* Reduction part of the elkan iteration, on a pair of scattered vectors */
template <typename T>
T elkan_reduction_kernel_scalar(da_int m, const T *x, da_int incx, T *y, da_int incy) {
    T sum = (T)0.0;
#pragma omp simd reduction(+ : sum)
    for (da_int k = 0; k < m; k++) {
        T tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }
    return sum;
}

template <>
float elkan_reduction_kernel<float, kmeans_kernel::scalar>(da_int m, const float *x,
                                                           da_int incx, float *y,
                                                           da_int incy) {
    return elkan_reduction_kernel_scalar(m, x, incx, y, incy);
}

template <>
double elkan_reduction_kernel<double, kmeans_kernel::scalar>(da_int m, const double *x,
                                                             da_int incx, double *y,
                                                             da_int incy) {
    return elkan_reduction_kernel_scalar(m, x, incx, y, incy);
}

// LCOV_EXCL_START
template <>
float elkan_reduction_kernel<float, kmeans_kernel::avx>(da_int m, const float *x,
                                                        da_int incx, float *y,
                                                        da_int incy) {

    v4sf_t v_sum;

    v_sum.v = _mm_setzero_ps();

    da_int simd_length = 4;
    da_int simd_loop_size = m - m % simd_length;
    da_int prefetch_condition = simd_loop_size - simd_length;

    da_int incx2 = incx * 2;
    da_int incy2 = incy * 2;
    da_int incx3 = incx * 3;
    da_int incy3 = incy * 3;

    da_int simd_incx = incx * simd_length;
    da_int simd_incy = incy * simd_length;

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincy = k * incy;
        da_int kincx = k * incx;
        const float *x_ptr = &x[kincx];
        const float *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }
        __m128 v_x = _mm_set_ps(x_ptr[incx3], x_ptr[incx2], x_ptr[incx], x_ptr[0]);
        __m128 v_y = _mm_set_ps(y_ptr[incy3], y_ptr[incy2], y_ptr[incy], y_ptr[0]);

        __m128 v_diff = _mm_sub_ps(v_x, v_y);
        v_sum.v = _mm_fmadd_ps(v_diff, v_diff, v_sum.v);
    }
    // Handle the remainder
    float sum = 0.0f;
    for (da_int k = simd_loop_size; k < m; k++) {
        float tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum + v_sum.f[0] + v_sum.f[1] + v_sum.f[2] + v_sum.f[3];
}

template <>
double elkan_reduction_kernel<double, kmeans_kernel::avx>(da_int m, const double *x,
                                                          da_int incx, double *y,
                                                          da_int incy) {

    v2df_t v_sum;

    v_sum.v = _mm_setzero_pd();

    da_int simd_length = 2;
    da_int simd_loop_size = m - m % simd_length;
    da_int prefetch_condition = simd_loop_size - simd_length;

    da_int simd_incx = incx * simd_length;
    da_int simd_incy = incy * simd_length;

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincy = k * incy;
        da_int kincx = k * incx;
        const double *x_ptr = &x[kincx];
        const double *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }

        __m128d v_x = _mm_set_pd(x_ptr[incx], x_ptr[0]);
        __m128d v_y = _mm_set_pd(y_ptr[incy], y_ptr[0]);

        __m128d v_diff = _mm_sub_pd(v_x, v_y);
        v_sum.v = _mm_fmadd_pd(v_diff, v_diff, v_sum.v);
    }

    // Handle the remainder
    double sum = 0.0;
    for (da_int k = simd_loop_size; k < m; k++) {
        float tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum + v_sum.d[0] + v_sum.d[1];
}

template <>
float elkan_reduction_kernel<float, kmeans_kernel::avx2>(da_int m, const float *x,
                                                         da_int incx, float *y,
                                                         da_int incy) {

    v8sf_t v_sum;

    v_sum.v = _mm256_setzero_ps();
    da_int simd_length = 8;
    da_int simd_loop_size = m - m % simd_length;
    da_int prefetch_condition = simd_loop_size - simd_length;

    da_int incx2 = incx * 2, incx3 = incx * 3, incx4 = incx * 4, incx5 = incx * 5,
           incx6 = incx * 6, incx7 = incx * 7, incy2 = incy * 2, incy3 = incy * 3,
           incy4 = incy * 4, incy5 = incy * 5, incy6 = incy * 6, incy7 = incy * 7;

    da_int simd_incx = incx * simd_length;
    da_int simd_incy = incy * simd_length;

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincy = k * incy;
        da_int kincx = k * incx;
        const float *x_ptr = &x[kincx];
        const float *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }

        __m256 v_x = _mm256_set_ps(x_ptr[incx7], x_ptr[incx6], x_ptr[incx5], x_ptr[incx4],
                                   x_ptr[incx3], x_ptr[incx2], x_ptr[incx], x_ptr[0]);
        __m256 v_y = _mm256_set_ps(y_ptr[incy7], y_ptr[incy6], y_ptr[incy5], y_ptr[incy4],
                                   y_ptr[incy3], y_ptr[incy2], y_ptr[incy], y_ptr[0]);

        __m256 v_diff = _mm256_sub_ps(v_x, v_y);
        v_sum.v = _mm256_fmadd_ps(v_diff, v_diff, v_sum.v);
    }

    // Handle the remainder
    float sum = 0.0f;
    for (da_int k = simd_loop_size; k < m; k++) {
        float tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum + v_sum.f[0] + v_sum.f[1] + v_sum.f[2] + v_sum.f[3] + v_sum.f[4] +
           v_sum.f[5] + v_sum.f[6] + v_sum.f[7];
}

template <>
double elkan_reduction_kernel<double, kmeans_kernel::avx2>(da_int m, const double *x,
                                                           da_int incx, double *y,
                                                           da_int incy) {

    v4df_t v_sum;

    v_sum.v = _mm256_setzero_pd();
    da_int simd_length = 4;
    da_int simd_loop_size = m - m % simd_length;
    da_int prefetch_condition = simd_loop_size - simd_length;

    da_int incx2 = incx * 2;
    da_int incy2 = incy * 2;
    da_int incx3 = incx * 3;
    da_int incy3 = incy * 3;

    da_int simd_incx = incx * simd_length;
    da_int simd_incy = incy * simd_length;

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincy = k * incy;
        da_int kincx = k * incx;
        const double *x_ptr = &x[kincx];
        const double *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }

        __m256d v_x = _mm256_set_pd(x_ptr[incx3], x_ptr[incx2], x_ptr[incx], x_ptr[0]);
        __m256d v_y = _mm256_set_pd(y_ptr[incy3], y_ptr[incy2], y_ptr[incy], y_ptr[0]);

        __m256d v_diff = _mm256_sub_pd(v_x, v_y);
        v_sum.v = _mm256_fmadd_pd(v_diff, v_diff, v_sum.v);
    }

    // Handle the remainder
    double sum = 0.0;
    for (da_int k = simd_loop_size; k < m; k++) {
        double tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum + v_sum.d[0] + v_sum.d[1] + v_sum.d[2] + v_sum.d[3];
}

#if defined(__AVX512F__)

template <>
double elkan_reduction_kernel<double, kmeans_kernel::avx512>(da_int m, const double *x,
                                                             da_int incx, double *y,
                                                             da_int incy) {

    v8df_t v_sum;

    v_sum.v = _mm512_setzero_pd();
    da_int simd_length = 8;
    da_int simd_loop_size = m - m % simd_length;
    da_int prefetch_condition = simd_loop_size - simd_length;

    da_int incx2 = incx * 2, incx3 = incx * 3, incx4 = incx * 4, incx5 = incx * 5,
           incx6 = incx * 6, incx7 = incx * 7, incy2 = incy * 2, incy3 = incy * 3,
           incy4 = incy * 4, incy5 = incy * 5, incy6 = incy * 6, incy7 = incy * 7;

    da_int simd_incx = incx * simd_length;
    da_int simd_incy = incy * simd_length;

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincy = k * incy;
        da_int kincx = k * incx;
        const double *x_ptr = &x[kincx];
        const double *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }

        __m512d v_x =
            _mm512_set_pd(x_ptr[incx7], x_ptr[incx6], x_ptr[incx5], x_ptr[incx4],
                          x_ptr[incx3], x_ptr[incx2], x_ptr[incx], x_ptr[0]);
        __m512d v_y =
            _mm512_set_pd(y_ptr[incy7], y_ptr[incy6], y_ptr[incy5], y_ptr[incy4],
                          y_ptr[incy3], y_ptr[incy2], y_ptr[incy], y_ptr[0]);

        __m512d v_diff = _mm512_sub_pd(v_x, v_y);
        v_sum.v = _mm512_fmadd_pd(v_diff, v_diff, v_sum.v);
    }

    // Handle the remainder
    double sum = 0.0f;
    for (da_int k = simd_loop_size; k < m; k++) {
        double tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum + v_sum.d[0] + v_sum.d[1] + v_sum.d[2] + v_sum.d[3] + v_sum.d[4] +
           v_sum.d[5] + v_sum.d[6] + v_sum.d[7];
}

template <>
float elkan_reduction_kernel<float, kmeans_kernel::avx512>(da_int m, const float *x,
                                                           da_int incx, float *y,
                                                           da_int incy) {

    v16sf_t v_sum;

    v_sum.v = _mm512_setzero_ps();
    da_int simd_length = 16;
    da_int simd_loop_size = m - m % simd_length;
    da_int prefetch_condition = simd_loop_size - simd_length;

    da_int incx2 = incx * 2, incy2 = incy * 2, incx3 = incx * 3, incy3 = incy * 3,
           incx4 = incx * 4, incy4 = incy * 4, incx5 = incx * 5, incy5 = incy * 5,
           incx6 = incx * 6, incy6 = incy * 6, incx7 = incx * 7, incy7 = incy * 7,
           incx8 = incx * 8, incy8 = incy * 8, incx9 = incx * 9, incy9 = incy * 9,
           incx10 = incx * 10, incy10 = incy * 10, incx11 = incx * 11, incy11 = incy * 11,
           incx12 = incx * 12, incy12 = incy * 12, incx13 = incx * 13, incy13 = incy * 13,
           incx14 = incx * 14, incy14 = incy * 14, incx15 = incx * 15, incy15 = incy * 15;

    da_int simd_incx = incx * simd_length;
    da_int simd_incy = incy * simd_length;

    for (da_int k = 0; k < simd_loop_size; k += simd_length) {

        da_int kincy = k * incy;
        da_int kincx = k * incx;
        const float *x_ptr = &x[kincx];
        const float *y_ptr = &y[kincy];

        if (k < prefetch_condition) {
            da_int x_ptr_index = simd_incx;
            da_int y_ptr_index = simd_incy;
            // Prefetch the elements for the next iteration to help with cache misses
            for (da_int j = 0; j < simd_length; j++) {
                _mm_prefetch((const char *)&x_ptr[x_ptr_index], _MM_HINT_T0);
                _mm_prefetch((const char *)&y_ptr[y_ptr_index], _MM_HINT_T0);
                x_ptr_index += incx;
                y_ptr_index += incy;
            }
        }

        __m512 v_x =
            _mm512_set_ps(x_ptr[incx15], x_ptr[incx14], x_ptr[incx13], x_ptr[incx12],
                          x_ptr[incx11], x_ptr[incx10], x_ptr[incx9], x_ptr[incx8],
                          x_ptr[incx7], x_ptr[incx6], x_ptr[incx5], x_ptr[incx4],
                          x_ptr[incx3], x_ptr[incx2], x_ptr[incx], x_ptr[0]);
        __m512 v_y =
            _mm512_set_ps(y_ptr[incy15], y_ptr[incy14], y_ptr[incy13], y_ptr[incy12],
                          y_ptr[incy11], y_ptr[incy10], y_ptr[incy9], y_ptr[incy8],
                          y_ptr[incy7], y_ptr[incy6], y_ptr[incy5], y_ptr[incy4],
                          y_ptr[incy3], y_ptr[incy2], y_ptr[incy], y_ptr[0]);

        __m512 v_diff = _mm512_sub_ps(v_x, v_y);
        v_sum.v = _mm512_fmadd_ps(v_diff, v_diff, v_sum.v);
    }

    // Handle the remainder
    double sum = 0.0f;
    for (da_int k = simd_loop_size; k < m; k++) {
        float tmp = x[k * incx] - y[k * incy];
        sum += tmp * tmp;
    }

    return sum + v_sum.f[0] + v_sum.f[1] + v_sum.f[2] + v_sum.f[3] + v_sum.f[4] +
           v_sum.f[5] + v_sum.f[6] + v_sum.f[7] + v_sum.f[8] + v_sum.f[9] + v_sum.f[10] +
           v_sum.f[11] + v_sum.f[12] + v_sum.f[13] + v_sum.f[14] + v_sum.f[15];
}

#endif

// LCOV_EXCL_STOP

/* Within Elkan iteration update a block of the lower and upper bound matrices*/
template <class T>
void elkan_iteration_kernel_scalar(da_int block_size, T *l_bound, da_int ldl_bound,
                                   T *u_bound, T *centre_shift, da_int *labels,
                                   da_int n_clusters) {

    da_int index = 0;
    for (da_int i = 0; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
#pragma omp simd
        for (da_int j = 0; j < n_clusters; j++) {
            l_bound[index + j] -= centre_shift[j];
            if (l_bound[index + j] < 0) {
                l_bound[index + j] = (T)0.0;
            }
        }
        index += ldl_bound;
    }
}

template <>
void elkan_iteration_kernel<double, kmeans_kernel::scalar>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {

    elkan_iteration_kernel_scalar(block_size, l_bound, ldl_bound, u_bound, centre_shift,
                                  labels, n_clusters);
}

template <>
void elkan_iteration_kernel<float, kmeans_kernel::scalar>(
    da_int block_size, float *l_bound, da_int ldl_bound, float *u_bound,
    float *centre_shift, da_int *labels, da_int n_clusters) {

    elkan_iteration_kernel_scalar(block_size, l_bound, ldl_bound, u_bound, centre_shift,
                                  labels, n_clusters);
}

template <>
void elkan_iteration_kernel<double, kmeans_kernel::avx>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {
    __m128d v_zero = _mm_setzero_pd();
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += 2) {
            da_int index = col_index + j;
            __m128d v_l_bound = _mm_loadu_pd(&l_bound[index]);
            __m128d v_centre_shift = _mm_loadu_pd(&centre_shift[j]);
            v_l_bound = _mm_sub_pd(v_l_bound, v_centre_shift);
            v_l_bound = _mm_max_pd(v_l_bound, v_zero);
            _mm_storeu_pd(&l_bound[index], v_l_bound);
        }
    }

    da_int simd_length = 2;
    da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        __m128d v_centre_shift =
            _mm_set_pd(centre_shift[labels[i + 1]], centre_shift[labels[i]]);
        __m128d v_u_bound = _mm_loadu_pd(&u_bound[i]);
        v_u_bound = _mm_add_pd(v_u_bound, v_centre_shift);
        _mm_storeu_pd(&u_bound[i], v_u_bound);
    }

    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

template <>
void elkan_iteration_kernel<float, kmeans_kernel::avx>(da_int block_size, float *l_bound,
                                                       da_int ldl_bound, float *u_bound,
                                                       float *centre_shift,
                                                       da_int *labels,
                                                       da_int n_clusters) {

    __m128 v_zero = _mm_setzero_ps();
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += 4) {
            da_int index = col_index + j;
            __m128 v_l_bound = _mm_loadu_ps(&l_bound[index]);
            __m128 v_centre_shift = _mm_loadu_ps(&centre_shift[j]);
            v_l_bound = _mm_sub_ps(v_l_bound, v_centre_shift);
            v_l_bound = _mm_max_ps(v_l_bound, v_zero);
            _mm_storeu_ps(&l_bound[index], v_l_bound);
        }
    }

    da_int simd_length = 4;
    da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        __m128 v_centre_shift =
            _mm_set_ps(centre_shift[labels[i + 3]], centre_shift[labels[i + 2]],
                       centre_shift[labels[i + 1]], centre_shift[labels[i]]);
        __m128 v_u_bound = _mm_loadu_ps(&u_bound[i]);
        v_u_bound = _mm_add_ps(v_u_bound, v_centre_shift);
        _mm_storeu_ps(&u_bound[i], v_u_bound);
    }

    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

template <>
void elkan_iteration_kernel<double, kmeans_kernel::avx2>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {
    __m256d v_zero = _mm256_setzero_pd();
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += 4) {
            da_int index = col_index + j;
            __m256d v_l_bound = _mm256_loadu_pd(&l_bound[index]);
            __m256d v_centre_shift = _mm256_loadu_pd(&centre_shift[j]);
            v_l_bound = _mm256_sub_pd(v_l_bound, v_centre_shift);
            v_l_bound = _mm256_max_pd(v_l_bound, v_zero);
            _mm256_storeu_pd(&l_bound[index], v_l_bound);
        }
    }

    da_int simd_length = 4;
    da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        __m256d v_centre_shift =
            _mm256_set_pd(centre_shift[labels[i + 3]], centre_shift[labels[i + 2]],
                          centre_shift[labels[i + 1]], centre_shift[labels[i]]);
        __m256d v_u_bound = _mm256_loadu_pd(&u_bound[i]);
        v_u_bound = _mm256_add_pd(v_u_bound, v_centre_shift);
        _mm256_storeu_pd(&u_bound[i], v_u_bound);
    }

    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

template <>
void elkan_iteration_kernel<float, kmeans_kernel::avx2>(da_int block_size, float *l_bound,
                                                        da_int ldl_bound, float *u_bound,
                                                        float *centre_shift,
                                                        da_int *labels,
                                                        da_int n_clusters) {
    __m256 v_zero = _mm256_setzero_ps();
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += 8) {
            da_int index = col_index + j;
            __m256 v_l_bound = _mm256_loadu_ps(&l_bound[index]);
            __m256 v_centre_shift = _mm256_loadu_ps(&centre_shift[j]);
            v_l_bound = _mm256_sub_ps(v_l_bound, v_centre_shift);
            v_l_bound = _mm256_max_ps(v_l_bound, v_zero);
            _mm256_storeu_ps(&l_bound[index], v_l_bound);
        }
    }

    da_int simd_length = 8;
    da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        __m256 v_centre_shift =
            _mm256_set_ps(centre_shift[labels[i + 7]], centre_shift[labels[i + 6]],
                          centre_shift[labels[i + 5]], centre_shift[labels[i + 4]],
                          centre_shift[labels[i + 3]], centre_shift[labels[i + 2]],
                          centre_shift[labels[i + 1]], centre_shift[labels[i]]);
        __m256 v_u_bound = _mm256_loadu_ps(&u_bound[i]);
        v_u_bound = _mm256_add_ps(v_u_bound, v_centre_shift);
        _mm256_storeu_ps(&u_bound[i], v_u_bound);
    }
    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

// LCOV_EXCL_START

#ifdef __AVX512F__
template <>
void elkan_iteration_kernel<double, kmeans_kernel::avx512>(
    da_int block_size, double *l_bound, da_int ldl_bound, double *u_bound,
    double *centre_shift, da_int *labels, da_int n_clusters) {

    __m512d v_zero = _mm512_setzero_pd();
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += 8) {
            da_int index = col_index + j;
            __m512d v_l_bound = _mm512_loadu_pd(&l_bound[index]);
            __m512d v_centre_shift = _mm512_loadu_pd(&centre_shift[j]);
            v_l_bound = _mm512_sub_pd(v_l_bound, v_centre_shift);
            v_l_bound = _mm512_max_pd(v_l_bound, v_zero);
            _mm512_storeu_pd(&l_bound[index], v_l_bound);
        }
    }

    da_int simd_length = 8;
    da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        __m512d v_centre_shift =
            _mm512_set_pd(centre_shift[labels[i + 7]], centre_shift[labels[i + 6]],
                          centre_shift[labels[i + 5]], centre_shift[labels[i + 4]],
                          centre_shift[labels[i + 3]], centre_shift[labels[i + 2]],
                          centre_shift[labels[i + 1]], centre_shift[labels[i]]);
        __m512d v_u_bound = _mm512_loadu_pd(&u_bound[i]);
        v_u_bound = _mm512_add_pd(v_u_bound, v_centre_shift);
        _mm512_storeu_pd(&u_bound[i], v_u_bound);
    }

    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}

template <>
void elkan_iteration_kernel<float, kmeans_kernel::avx512>(
    da_int block_size, float *l_bound, da_int ldl_bound, float *u_bound,
    float *centre_shift, da_int *labels, da_int n_clusters) {
    __m512 v_zero = _mm512_setzero_ps();
    for (da_int i = 0; i < block_size; i++) {
        da_int col_index = i * ldl_bound;

        for (da_int j = 0; j < n_clusters; j += 16) {
            da_int index = col_index + j;
            __m512 v_l_bound = _mm512_loadu_ps(&l_bound[index]);
            __m512 v_centre_shift = _mm512_loadu_ps(&centre_shift[j]);
            v_l_bound = _mm512_sub_ps(v_l_bound, v_centre_shift);
            v_l_bound = _mm512_max_ps(v_l_bound, v_zero);
            _mm512_storeu_ps(&l_bound[index], v_l_bound);
        }
    }

    da_int simd_length = 16;
    da_int simd_loop_size = block_size - block_size % simd_length;

    for (da_int i = 0; i < simd_loop_size; i += simd_length) {
        __m512 v_centre_shift =
            _mm512_set_ps(centre_shift[labels[i + 15]], centre_shift[labels[i + 14]],
                          centre_shift[labels[i + 13]], centre_shift[labels[i + 12]],
                          centre_shift[labels[i + 11]], centre_shift[labels[i + 10]],
                          centre_shift[labels[i + 9]], centre_shift[labels[i + 8]],
                          centre_shift[labels[i + 7]], centre_shift[labels[i + 6]],
                          centre_shift[labels[i + 5]], centre_shift[labels[i + 4]],
                          centre_shift[labels[i + 3]], centre_shift[labels[i + 2]],
                          centre_shift[labels[i + 1]], centre_shift[labels[i]]);
        __m512 v_u_bound = _mm512_loadu_ps(&u_bound[i]);
        v_u_bound = _mm512_add_ps(v_u_bound, v_centre_shift);
        _mm512_storeu_ps(&u_bound[i], v_u_bound);
    }
    // Handle the remainder
    for (da_int i = simd_loop_size; i < block_size; i++) {
        u_bound[i] += centre_shift[labels[i]];
    }
}
#endif

// LCOV_EXCL_STOP

template <class T>
void lloyd_iteration_kernel_scalar(bool update_centres, da_int block_size,
                                   T *centre_norms, da_int *cluster_count, da_int *labels,
                                   T *work, da_int ldwork, da_int n_clusters) {

    T tmp2 = centre_norms[0];

    // Go through each sample in work and find argmin

#pragma omp simd
    for (da_int i = 0; i < block_size; i++) {
        da_int ind = i * ldwork;
        T smallest_dist = work[ind] + tmp2;
        da_int label = 0;
        for (da_int j = 1; j < n_clusters; j++) {
            da_int index = ind + j;
            T tmp = work[index] + centre_norms[j];
            if (tmp < smallest_dist) {
                label = j;
                smallest_dist = tmp;
            }
        }
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<double, kmeans_kernel::scalar>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    lloyd_iteration_kernel_scalar(update_centres, block_size, centre_norms, cluster_count,
                                  labels, work, ldwork, n_clusters);
}

template <>
void lloyd_iteration_kernel<float, kmeans_kernel::scalar>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    lloyd_iteration_kernel_scalar(update_centres, block_size, centre_norms, cluster_count,
                                  labels, work, ldwork, n_clusters);
}

template <>
void lloyd_iteration_kernel<double, kmeans_kernel::avx>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    // Declare as unions so we can access individual elements later
    v2df_t v_smallest_dists;
    v2i64_t v_labels;

    __m128d v_centre_norms = _mm_loadu_pd(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v = _mm_add_pd(_mm_loadu_pd(work + ind_outer), v_centre_norms);

        v_labels.v = _mm_set_epi64x(1, 0);

        // No need to worry about n_clusters not being a multpile of 2 as we have already padded the relevant arrays
        for (da_int j = 2; j < n_clusters; j += 2) {
            da_int ind_inner = ind_outer + j;
            __m128d v_tmp = _mm_add_pd(_mm_loadu_pd(work + ind_inner),
                                       _mm_loadu_pd(centre_norms + j));
            __m128d v_mask = _mm_cmp_pd(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            __m128i v_mask_int = _mm_castpd_si128(v_mask);
            // Use 64 bit integers in v_labels and v_indices
            __m128i v_indices = _mm_set_epi64x(j + 1, j);
            v_labels.v = _mm_blendv_epi8(v_labels.v, v_indices, v_mask_int);
            v_smallest_dists.v = _mm_min_pd(v_smallest_dists.v, v_tmp);
        }

        labels[i] = (da_int)v_labels.i[0];

        if (v_smallest_dists.d[1] < v_smallest_dists.d[0]) {
            labels[i] = (da_int)v_labels.i[1];
        }

        if (update_centres)
            cluster_count[labels[i]] += 1;
    }
}

template <>
void lloyd_iteration_kernel<float, kmeans_kernel::avx>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    v4sf_t v_smallest_dists;
#if defined(AOCLDA_ILP64)
    v2i64_t v_labels1, v_labels2;
#else
    v4i32_t v_labels;
#endif

    __m128 v_centre_norms = _mm_loadu_ps(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v = _mm_add_ps(_mm_loadu_ps(work + ind_outer), v_centre_norms);
#if defined(AOCLDA_ILP64)
        // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
        v_labels1.v = _mm_set_epi64x(1, 0);
        v_labels2.v = _mm_set_epi64x(3, 2);
#else
        v_labels.v = _mm_set_epi32(3, 2, 1, 0);
#endif
        // No need to worry about n_clusters not being a multpile of 4 as we have already padded the relevant arrays
        for (da_int j = 4; j < n_clusters; j += 4) {
            da_int ind_inner = ind_outer + j;
            __m128 v_tmp = _mm_add_ps(_mm_loadu_ps(work + ind_inner),
                                      _mm_loadu_ps(centre_norms + j));
            __m128 v_mask = _mm_cmp_ps(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);

#if defined(AOCLDA_ILP64)
            // v_mask will currently only work for 32 bit integers, so we need to create two
            // 64 bit integer masks from it

            //  Extract the lower bits of v_mask and create a new mask which duplicates them
            __m128 v_mask_perm_lower = _mm_permute_ps(v_mask, 0b01010000);
            __m128i v_mask_lower = _mm_castps_si128(v_mask_perm_lower);

            //  Extract the upper bits of v_mask and create a new mask which duplicates them
            __m128 v_mask_perm_upper = _mm_permute_ps(v_mask, 0b11111010);
            __m128i v_mask_upper = _mm_castps_si128(v_mask_perm_upper);

            __m128i v_indices1 = _mm_set_epi64x(j + 1, j);
            __m128i v_indices2 = _mm_set_epi64x(j + 3, j + 2);

            // Use our new masks to blend the indices with v_labels1 and 2, all of which are 64 bits
            v_labels1.v = _mm_blendv_epi8(v_labels1.v, v_indices1, v_mask_lower);
            v_labels2.v = _mm_blendv_epi8(v_labels2.v, v_indices2, v_mask_upper);
#else
            __m128i v_mask_int = _mm_castps_si128(v_mask);
            __m128i v_indices = _mm_set_epi32(j + 3, j + 2, j + 1, j);
            v_labels.v = _mm_blendv_epi8(v_labels.v, v_indices, v_mask_int);
#endif
            v_smallest_dists.v = _mm_min_ps(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
#if defined(AOCLDA_ILP64)
        da_int label = v_labels1.i[0];

        if (v_smallest_dists.f[1] < v_smallest_dists.f[0]) {
            v_smallest_dists.f[0] = v_smallest_dists.f[1];
            label = (da_int)v_labels1.i[1];
        }

        if (v_smallest_dists.f[2] < v_smallest_dists.f[0]) {
            v_smallest_dists.f[0] = v_smallest_dists.f[2];
            label = (da_int)v_labels2.i[0];
        }

        if (v_smallest_dists.f[3] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels2.i[1];
        }

#else

        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 3; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.f[3] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels.i[3];
        }

#endif
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<double, kmeans_kernel::avx2>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    v4df_t v_smallest_dists;
    v4i64_t v_labels;

    __m256d v_centre_norms = _mm256_loadu_pd(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm256_add_pd(_mm256_loadu_pd(work + ind_outer), v_centre_norms);

        v_labels.v = _mm256_set_epi64x(3, 2, 1, 0);

        // No need to worry about n_clusters not being a multpile of 4 as we have already padded the relevant arrays
        for (da_int j = 4; j < n_clusters; j += 4) {
            da_int ind_inner = ind_outer + j;
            __m256d v_tmp = _mm256_add_pd(_mm256_loadu_pd(work + ind_inner),
                                          _mm256_loadu_pd(centre_norms + j));
            __m256d v_mask = _mm256_cmp_pd(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            __m256i v_mask_int = _mm256_castpd_si256(v_mask);
            // Note, we are working with 64 bit integers in v_labels and v_indices
            __m256i v_indices = _mm256_set_epi64x(j + 3, j + 2, j + 1, j);
            v_labels.v = _mm256_blendv_epi8(v_labels.v, v_indices, v_mask_int);
            v_smallest_dists.v = _mm256_min_pd(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 3; j++) {
            if (v_smallest_dists.d[j] < v_smallest_dists.d[0]) {
                v_smallest_dists.d[0] = v_smallest_dists.d[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.d[3] < v_smallest_dists.d[0]) {
            label = (da_int)v_labels.i[3];
        }

        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<float, kmeans_kernel::avx2>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    v8sf_t v_smallest_dists;
#if defined(AOCLDA_ILP64)
    v4i64_t v_labels1, v_labels2;
#else
    v8i32_t v_labels;
#endif

    __m256 v_centre_norms = _mm256_loadu_ps(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm256_add_ps(_mm256_loadu_ps(work + ind_outer), v_centre_norms);
#if defined(AOCLDA_ILP64)
        // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
        v_labels1.v = _mm256_set_epi64x(3, 2, 1, 0);
        v_labels2.v = _mm256_set_epi64x(7, 6, 5, 4);
#else
        v_labels.v = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
#endif
        // No need to worry about n_clusters not being a multpile of 8 as we have already padded the relevant arrays
        for (da_int j = 8; j < n_clusters; j += 8) {
            da_int ind_inner = ind_outer + j;
            __m256 v_tmp = _mm256_add_ps(_mm256_loadu_ps(work + ind_inner),
                                         _mm256_loadu_ps(centre_norms + j));
            __m256 v_mask = _mm256_cmp_ps(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            __m256i v_mask_int = _mm256_castps_si256(v_mask);

#if defined(AOCLDA_ILP64)
            // v_mask_int will currently only work for 32 bit integers, so we need to create two
            // 64 bit integer masks from it

            //  Extract the lower bits of v_mask_int and create a new mask which duplicates them
            __m256i control_mask_lower = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
            __m256i v_mask_lower =
                _mm256_permutevar8x32_epi32(v_mask_int, control_mask_lower);

            //  Extract the upper bits of v_mask_int and create a new mask which duplicates them
            __m256i control_mask_upper = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);
            __m256i v_mask_upper =
                _mm256_permutevar8x32_epi32(v_mask_int, control_mask_upper);

            // Use our new masks to blend the indices with v_labels1 and 2, all of which are 64 bits
            __m256i v_indices1 = _mm256_set_epi64x(j + 3, j + 2, j + 1, j);
            __m256i v_indices2 = _mm256_set_epi64x(j + 7, j + 6, j + 5, j + 4);
            v_labels1.v = _mm256_blendv_epi8(v_labels1.v, v_indices1, v_mask_lower);
            v_labels2.v = _mm256_blendv_epi8(v_labels2.v, v_indices2, v_mask_upper);
#else

            __m256i v_indices =
                _mm256_set_epi32(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
            v_labels.v = _mm256_blendv_epi8(v_labels.v, v_indices, v_mask_int);
#endif
            v_smallest_dists.v = _mm256_min_ps(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
#if defined(AOCLDA_ILP64)
        da_int label = v_labels1.i[0];
        for (da_int j = 1; j < 4; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels1.i[j];
            }
        }
        for (da_int j = 4; j < 7; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels2.i[j - 4];
            }
        }
        if (v_smallest_dists.f[7] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels2.i[3];
        }

#else
        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 7; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.f[7] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels.i[7];
        }
#endif
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

// LCOV_EXCL_START

#ifdef __AVX512F__
template <>
void lloyd_iteration_kernel<float, kmeans_kernel::avx512>(
    bool update_centres, da_int block_size, float *centre_norms, da_int *cluster_count,
    da_int *labels, float *work, da_int ldwork, da_int n_clusters) {

    v16sf_t v_smallest_dists;
#if defined(AOCLDA_ILP64)
    v8i64_t v_labels1, v_labels2;
#else
    v16i32_t v_labels;
#endif

    __m512 v_centre_norms = _mm512_loadu_ps(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm512_add_ps(_mm512_loadu_ps(work + ind_outer), v_centre_norms);
#if defined(AOCLDA_ILP64)
        // Because we are dealing with 64 bit integers, we need to use two integer vectors for the labels
        v_labels1.v = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        v_labels2.v = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
#else
        v_labels.v =
            _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#endif
        // No need to worry about n_clusters not being a multpile of 16 as we have already padded the relevant arrays
        for (da_int j = 16; j < n_clusters; j += 16) {
            da_int ind_inner = ind_outer + j;
            __m512 v_tmp = _mm512_add_ps(_mm512_loadu_ps(work + ind_inner),
                                         _mm512_loadu_ps(centre_norms + j));
            __mmask16 v_mask = _mm512_cmp_ps_mask(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);

#if defined(AOCLDA_ILP64)
            // v_mask will currently only work for 32 bit integers, so we need to create two
            // masks from it so we can work with 64 bit integers

            // Split v_mask into two _mmask8 variables
            __mmask8 v_mask_lower = v_mask & 0xFF;        // Keep the lower 8 bits
            __mmask8 v_mask_upper = (v_mask >> 8) & 0xFF; // Get the upper 8 bits

            __m512i v_indices1 =
                _mm512_set_epi64(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
            __m512i v_indices2 = _mm512_set_epi64(j + 15, j + 14, j + 13, j + 12, j + 11,
                                                  j + 10, j + 9, j + 8);

            // Use our new masks to blend the indices with v_labels1 and 2, all of which are 64 bits
            v_labels1.v = _mm512_mask_blend_epi64(v_mask_lower, v_labels1.v, v_indices1);
            v_labels2.v = _mm512_mask_blend_epi64(v_mask_upper, v_labels2.v, v_indices2);
#else

            __m512i v_indices = _mm512_set_epi32(j + 15, j + 14, j + 13, j + 12, j + 11,
                                                 j + 10, j + 9, j + 8, j + 7, j + 6,
                                                 j + 5, j + 4, j + 3, j + 2, j + 1, j);
            v_labels.v = _mm512_mask_blend_epi32(v_mask, v_labels.v, v_indices);
#endif
            v_smallest_dists.v = _mm512_min_ps(v_smallest_dists.v, v_tmp);
        }

        // Extract the label corresponding to the smallest distance computed (little to be gained from using permute operations here)
#if defined(AOCLDA_ILP64)
        da_int label = v_labels1.i[0];
        for (da_int j = 1; j < 8; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels1.i[j];
            }
        }
        for (da_int j = 8; j < 15; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels2.i[j - 8];
            }
        }
        if (v_smallest_dists.f[15] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels2.i[7];
        }

#else

        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 15; j++) {
            if (v_smallest_dists.f[j] < v_smallest_dists.f[0]) {
                v_smallest_dists.f[0] = v_smallest_dists.f[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.f[15] < v_smallest_dists.f[0]) {
            label = (da_int)v_labels.i[15];
        }

#endif
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}

template <>
void lloyd_iteration_kernel<double, kmeans_kernel::avx512>(
    bool update_centres, da_int block_size, double *centre_norms, da_int *cluster_count,
    da_int *labels, double *work, da_int ldwork, da_int n_clusters) {

    v8df_t v_smallest_dists;
    v8i64_t v_labels;

    __m512d v_centre_norms = _mm512_loadu_pd(centre_norms);

    for (da_int i = 0; i < block_size; i++) {
        da_int ind_outer = i * ldwork;
        v_smallest_dists.v =
            _mm512_add_pd(_mm512_loadu_pd(work + ind_outer), v_centre_norms);

        v_labels.v = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

        // No need to worry about n_clusters not being a multpile of 8 as we have already padded the relevant arrays
        for (da_int j = 8; j < n_clusters; j += 8) {
            da_int ind_inner = ind_outer + j;
            __m512d v_tmp = _mm512_add_pd(_mm512_loadu_pd(work + ind_inner),
                                          _mm512_loadu_pd(centre_norms + j));
            __mmask8 v_mask = _mm512_cmp_pd_mask(v_tmp, v_smallest_dists.v, _CMP_LT_OQ);
            // Note, we are working with 64 bit integers in v_labels and v_indices
            __m512i v_indices =
                _mm512_set_epi64(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
            v_labels.v = _mm512_mask_blend_epi64(v_mask, v_labels.v, v_indices);
            v_smallest_dists.v = _mm512_min_pd(v_smallest_dists.v, v_tmp);
        }

        da_int label = (da_int)v_labels.i[0];
        for (da_int j = 1; j < 7; j++) {
            if (v_smallest_dists.d[j] < v_smallest_dists.d[0]) {
                v_smallest_dists.d[0] = v_smallest_dists.d[j];
                label = (da_int)v_labels.i[j];
            }
        }
        if (v_smallest_dists.d[7] < v_smallest_dists.d[0]) {
            label = (da_int)v_labels.i[7];
        }
        labels[i] = label;
        if (update_centres)
            cluster_count[label] += 1;
    }
}
#endif

// LCOV_EXCL_STOP

} // namespace da_kmeans

} // namespace ARCH
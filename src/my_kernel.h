#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <pthread.h>
#include <immintrin.h>

#define ALIGN 64
#define NTHREADS 64
#define BATCH_SIZE 4
#define BATCH_GAP 6 // 读6行算4行
#define SIMD_LANE 8

#include "kernel_r6c4_avx2.h"

// 按照行划分任务
// factor是划分最小粒度
inline void distribute_row(const int nthreads, const int tid, const int width, const int height, const int factor, int &size, int &start)
{
    size = (int)height / nthreads;
    int nloops = (int)size / factor;
    size = nloops * factor;
    int remainder = height - size * nthreads;

    if (tid < remainder / factor)
    {
        size = size + factor;
        start = tid * size;
    }
    else if ((remainder % factor != 0) && tid == remainder / factor)
    { // 不能整除的情况
        start = tid * (size + factor);
        size = size + remainder % factor;
    }
    else
    {
        start = remainder + tid * size;
    }
}

inline void ApplyStencil_simd_avx2_float(float *in, float *out, int width, int height)
{
    // omp_set_num_threads(NTHREADS);

    int tid, m_omp, ms_omp;
#pragma omp parallel num_threads(NTHREADS) default(shared) private(tid, m_omp, ms_omp)
    {
        tid = omp_get_thread_num();
        // thread affinity
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(tid, &mask);
        if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0)
        {
            fprintf(stderr, "set thread affinity failed\n");
        }
        // 按照行划分任务
        distribute_row(NTHREADS, tid, width, height, BATCH_SIZE, m_omp, ms_omp);
        // 调用计算核心
        kernel_read6_compute4_avx256_float(in, out,  width,  height,  tid,  m_omp,  ms_omp);
    }
}
inline void ApplyStencil_avx256_float(float *in, float *out, int width, int height)
{
    // omp_set_num_threads(NTHREADS);

    int tid, m_omp, ms_omp;
#pragma omp parallel num_threads(NTHREADS) default(shared) private(tid, m_omp, ms_omp)
    {
        tid = omp_get_thread_num();
        // thread affinity
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(tid, &mask);
        if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0)
        {
            fprintf(stderr, "set thread affinity failed\n");
        }
        // 按照行划分任务
        distribute_row(NTHREADS, tid, width, height, BATCH_SIZE, m_omp, ms_omp);
        // m_omp 任务长度， os_omp，任务开始位置
        // 开始计算
        float *dat_1_1, *dat_1_2, *dat_1_3;
        float *dat_2_1, *dat_2_2, *dat_2_3;
        float *dat_3_1, *dat_3_2, *dat_3_3;
        float *dat_4_1, *dat_4_2, *dat_4_3;
        float *dat_5_1, *dat_5_2, *dat_5_3;
        float *dat_6_1, *dat_6_2, *dat_6_3;
        int i = 0;
        if (ms_omp < 1)
            i = 1;

        // 特判最后一行，防止越界
        int i_barrier = ms_omp + m_omp == height ? m_omp - BATCH_SIZE - 1 : m_omp - BATCH_SIZE;
        int actual_m_omp = ms_omp + m_omp == height ? m_omp - 1 : m_omp;

        for (; i <= i_barrier; i += BATCH_SIZE)
        {
            int j = 0;
            int index_start = (ms_omp + i - 1) * width;
            float *in_0 = in + (ms_omp + i - 1) * width;
            float *in_1 = in + (ms_omp + i) * width;
            float *in_2 = in + (ms_omp + i + 1) * width;
            float *in_3 = in + (ms_omp + i + 2) * width;
            float *in_4 = in + (ms_omp + i + 3) * width;
            float *in_5 = in + (ms_omp + i + 4) * width;
            float *out_1 = out + (ms_omp + i) * width;
            float *out_2 = out + (ms_omp + i + 1) * width;
            float *out_3 = out + (ms_omp + i + 2) * width;
            float *out_4 = out + (ms_omp + i + 3) * width;
            // 事先处理好第0 1个，直接从第2个开始算(3x3写回的第0个)
            for (j = 1; j <= width - SIMD_LANE - 1; j += SIMD_LANE)
            { //填充流水线需要展开

                // load data_1_1 and data_1_2
                dat_1_1 = in_0 + j - 1;
                dat_1_2 = in_0 + j;
                dat_1_3 = in_0 + j + 1;
                dat_2_1 = in_1 + j - 1;
                dat_2_2 = in_1 + j;
                dat_2_3 = in_1 + j + 1;
                dat_3_1 = in_2 + j - 1;
                dat_3_2 = in_2 + j;
                dat_3_3 = in_2 + j + 1;
                dat_4_1 = in_3 + j - 1;
                dat_4_2 = in_3 + j;
                dat_4_3 = in_3 + j + 1;
                dat_5_1 = in_4 + j - 1;
                dat_5_2 = in_4 + j;
                dat_5_3 = in_4 + j + 1;
                dat_6_1 = in_5 + j - 1;
                dat_6_2 = in_5 + j;
                dat_6_3 = in_5 + j + 1;
                float sum_23_1[SIMD_LANE], sum_23_2[SIMD_LANE], sum_23_3[SIMD_LANE];
                float sum_45_1[SIMD_LANE], sum_45_2[SIMD_LANE], sum_45_3[SIMD_LANE];
                float sum_123_1[SIMD_LANE], sum_234_1[SIMD_LANE], sum_345_1[SIMD_LANE], sum_456_1[SIMD_LANE];
                float sum_123_2[SIMD_LANE], sum_234_2[SIMD_LANE], sum_345_2[SIMD_LANE], sum_456_2[SIMD_LANE];
                float sum_123_3[SIMD_LANE], sum_234_3[SIMD_LANE], sum_345_3[SIMD_LANE], sum_456_3[SIMD_LANE];
                // float sum_1_12[SIMD_LANE], sum_2_12[SIMD_LANE], sum_3_12[SIMD_LANE], sum_4_12[SIMD_LANE], sum_5_12[SIMD_LANE], sum_6_12[SIMD_LANE];
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_23_1[k] = dat_2_1[k] + dat_3_1[k];
                    sum_23_2[k] = dat_2_2[k] + dat_3_2[k];
                    sum_45_1[k] = dat_4_1[k] + dat_5_1[k];
                    sum_45_2[k] = dat_4_2[k] + dat_5_2[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_123_1[k] = dat_1_1[k] + sum_23_1[k];
                    sum_123_2[k] = dat_1_2[k] + sum_23_2[k];
                    sum_234_1[k] = dat_4_1[k] + sum_23_1[k];
                    sum_234_2[k] = dat_4_2[k] + sum_23_2[k];
                    sum_345_1[k] = dat_3_1[k] + sum_45_1[k];
                    sum_345_2[k] = dat_3_2[k] + sum_45_2[k];
                    sum_456_1[k] = dat_6_1[k] + sum_45_1[k];
                    sum_456_2[k] = dat_6_2[k] + sum_45_2[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_23_3[k] = dat_2_3[k] + dat_3_3[k];
                    sum_45_3[k] = dat_4_3[k] + dat_5_3[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_123_3[k] = dat_1_3[k] + sum_23_3[k];
                    sum_234_3[k] = dat_4_3[k] + sum_23_3[k];
                    sum_345_3[k] = dat_3_3[k] + sum_45_3[k];
                    sum_456_3[k] = dat_6_3[k] + sum_45_3[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_123_1[k] += sum_123_2[k] + sum_123_3[k];
                    sum_234_1[k] += sum_234_2[k] + sum_234_3[k];
                    sum_345_1[k] += sum_345_2[k] + sum_345_3[k];
                    sum_456_1[k] += sum_456_2[k] + sum_456_3[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    out_1[j + k] = 9 * in_1[j + k] - sum_123_1[k];
                    out_2[j + k] = 9 * in_2[j + k] - sum_234_1[k];
                    out_3[j + k] = 9 * in_3[j + k] - sum_345_1[k];
                    out_4[j + k] = 9 * in_4[j + k] - sum_456_1[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    out_1[j + k] = out_1[j + k] < 0 ? 0 : out_1[j + k];
                    out_1[j + k] = out_1[j + k] > 255 ? 255 : out_1[j + k];
                    out_2[j + k] = out_2[j + k] < 0 ? 0 : out_2[j + k];
                    out_2[j + k] = out_2[j + k] > 255 ? 255 : out_2[j + k];
                    out_3[j + k] = out_3[j + k] < 0 ? 0 : out_3[j + k];
                    out_3[j + k] = out_3[j + k] > 255 ? 255 : out_3[j + k];
                    out_4[j + k] = out_4[j + k] < 0 ? 0 : out_4[j + k];
                    out_4[j + k] = out_4[j + k] > 255 ? 255 : out_4[j + k];
                }
            }
            
            // for (; j < width - 1; j++)
            { // 处理最后的
              int last_size = width - j;
                // load data_1_1 and data_1_2
                dat_1_1 = in_0 + j - 1;
                dat_1_2 = in_0 + j;
                dat_1_3 = in_0 + j + 1;
                dat_2_1 = in_1 + j - 1;
                dat_2_2 = in_1 + j;
                dat_2_3 = in_1 + j + 1;
                dat_3_1 = in_2 + j - 1;
                dat_3_2 = in_2 + j;
                dat_3_3 = in_2 + j + 1;
                dat_4_1 = in_3 + j - 1;
                dat_4_2 = in_3 + j;
                dat_4_3 = in_3 + j + 1;
                dat_5_1 = in_4 + j - 1;
                dat_5_2 = in_4 + j;
                dat_5_3 = in_4 + j + 1;
                dat_6_1 = in_5 + j - 1;
                dat_6_2 = in_5 + j;
                dat_6_3 = in_5 + j + 1;
                float sum_23_1[SIMD_LANE], sum_23_2[SIMD_LANE], sum_23_3[SIMD_LANE];
                float sum_45_1[SIMD_LANE], sum_45_2[SIMD_LANE], sum_45_3[SIMD_LANE];
                float sum_123_1[SIMD_LANE], sum_234_1[SIMD_LANE], sum_345_1[SIMD_LANE], sum_456_1[SIMD_LANE];
                float sum_123_2[SIMD_LANE], sum_234_2[SIMD_LANE], sum_345_2[SIMD_LANE], sum_456_2[SIMD_LANE];
                float sum_123_3[SIMD_LANE], sum_234_3[SIMD_LANE], sum_345_3[SIMD_LANE], sum_456_3[SIMD_LANE];
                // float sum_1_12[SIMD_LANE], sum_2_12[SIMD_LANE], sum_3_12[SIMD_LANE], sum_4_12[SIMD_LANE], sum_5_12[SIMD_LANE], sum_6_12[SIMD_LANE];
                
                for (int k = 0; k < last_size; k++)
                {
                    sum_23_1[k] = dat_2_1[k] + dat_3_1[k];
                    sum_23_2[k] = dat_2_2[k] + dat_3_2[k];
                    sum_45_1[k] = dat_4_1[k] + dat_5_1[k];
                    sum_45_2[k] = dat_4_2[k] + dat_5_2[k];
                }
                
                for (int k = 0; k < last_size; k++)
                {
                    sum_123_1[k] = dat_1_1[k] + sum_23_1[k];
                    sum_123_2[k] = dat_1_2[k] + sum_23_2[k];
                    sum_234_1[k] = dat_4_1[k] + sum_23_1[k];
                    sum_234_2[k] = dat_4_2[k] + sum_23_2[k];
                    sum_345_1[k] = dat_3_1[k] + sum_45_1[k];
                    sum_345_2[k] = dat_3_2[k] + sum_45_2[k];
                    sum_456_1[k] = dat_6_1[k] + sum_45_1[k];
                    sum_456_2[k] = dat_6_2[k] + sum_45_2[k];
                }
                
                for (int k = 0; k < last_size; k++)
                {
                    sum_23_3[k] = dat_2_3[k] + dat_3_3[k];
                    sum_45_3[k] = dat_4_3[k] + dat_5_3[k];
                }
                
                for (int k = 0; k < last_size; k++)
                {
                    sum_123_3[k] = dat_1_3[k] + sum_23_3[k];
                    sum_234_3[k] = dat_4_3[k] + sum_23_3[k];
                    sum_345_3[k] = dat_3_3[k] + sum_45_3[k];
                    sum_456_3[k] = dat_6_3[k] + sum_45_3[k];
                }
                
                for (int k = 0; k < last_size; k++)
                {
                    sum_123_1[k] += sum_123_2[k] + sum_123_3[k];
                    sum_234_1[k] += sum_234_2[k] + sum_234_3[k];
                    sum_345_1[k] += sum_345_2[k] + sum_345_3[k];
                    sum_456_1[k] += sum_456_2[k] + sum_456_3[k];
                }
                
                for (int k = 0; k < last_size; k++)
                {
                    out_1[j + k] = 9 * in_1[j + k] - sum_123_1[k];
                    out_2[j + k] = 9 * in_2[j + k] - sum_234_1[k];
                    out_3[j + k] = 9 * in_3[j + k] - sum_345_1[k];
                    out_4[j + k] = 9 * in_4[j + k] - sum_456_1[k];
                }
                
                for (int k = 0; k < last_size; k++)
                {
                    out_1[j + k] = out_1[j + k] < 0 ? 0 : out_1[j + k];
                    out_1[j + k] = out_1[j + k] > 255 ? 255 : out_1[j + k];
                    out_2[j + k] = out_2[j + k] < 0 ? 0 : out_2[j + k];
                    out_2[j + k] = out_2[j + k] > 255 ? 255 : out_2[j + k];
                    out_3[j + k] = out_3[j + k] < 0 ? 0 : out_3[j + k];
                    out_3[j + k] = out_3[j + k] > 255 ? 255 : out_3[j + k];
                    out_4[j + k] = out_4[j + k] < 0 ? 0 : out_4[j + k];
                    out_4[j + k] = out_4[j + k] > 255 ? 255 : out_4[j + k];
                }
            }
        }
        // 注意不要越界了
        // 处理剩下的
        for (; i < actual_m_omp; i++)
        {
            int j = 0;
            int index_start = (ms_omp + i - 1) * width;
            float *in_0 = in + (ms_omp + i - 1) * width;
            float *in_1 = in + (ms_omp + i) * width;
            float *in_2 = in + (ms_omp + i + 1) * width;
            float *out_1 = out + (ms_omp + i) * width;
            for (j = 1; j <= width - SIMD_LANE - 1; j += SIMD_LANE)
            { //填充流水线需要展开

                // load data_1_1 and data_1_2
                dat_1_1 = in_0 + j - 1;
                dat_1_2 = in_0 + j;
                dat_1_3 = in_0 + j + 1;
                dat_2_1 = in_1 + j - 1;
                dat_2_2 = in_1 + j;
                dat_2_3 = in_1 + j + 1;
                dat_3_1 = in_2 + j - 1;
                dat_3_2 = in_2 + j;
                dat_3_3 = in_2 + j + 1;
                float sum_23_1[SIMD_LANE], sum_23_2[SIMD_LANE], sum_23_3[SIMD_LANE];
                float sum_123_1[SIMD_LANE];
                float sum_123_2[SIMD_LANE];
                float sum_123_3[SIMD_LANE];
                // float sum_1_12[SIMD_LANE], sum_2_12[SIMD_LANE], sum_3_12[SIMD_LANE], sum_4_12[SIMD_LANE], sum_5_12[SIMD_LANE], sum_6_12[SIMD_LANE];
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_23_1[k] = dat_2_1[k] + dat_3_1[k];
                    sum_23_2[k] = dat_2_2[k] + dat_3_2[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_123_1[k] = dat_1_1[k] + sum_23_1[k];
                    sum_123_2[k] = dat_1_2[k] + sum_23_2[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_23_3[k] = dat_2_3[k] + dat_3_3[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_123_3[k] = dat_1_3[k] + sum_23_3[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    sum_123_1[k] += sum_123_2[k] + sum_123_3[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    out_1[j + k] = 9 * in_1[j + k] - sum_123_1[k];
                }
                #pragma unroll(SIMD_LANE)
                for (int k = 0; k < SIMD_LANE; k++)
                {
                    out_1[j + k] = out_1[j + k] < 0 ? 0 : out_1[j + k];
                    out_1[j + k] = out_1[j + k] > 255 ? 255 : out_1[j + k];
                }
            }
            
            for (; j < width - 1; j++) { // 处理最后的
                float val =
                    -in_0[j-1] - in_0[j] - in_0[j+1] - in_1[j-1] + 8 * in_1[j] - in_1[j+1] - in_2[j-1] - in_2[j] - in_2[j+1];

                val = (val < 0 ? 0 : val);
                val = (val > 255 ? 255 : val);

                out_1[j] = val;
            }
        }
    }
}
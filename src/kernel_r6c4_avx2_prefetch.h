#include <immintrin.h>

#define ALIGN 64
#define BATCH_SIZE 4
#define BATCH_GAP 6 // 读6行算4行
#define SIMD_LANE 8
#define PREFETCH_DISTANCE 16

inline void kernel_read6_compute4_avx256_float(float *in, float *out, int width, int height, int tid, int m_omp, int ms_omp)
{
    // 开始计算
    float *dat_1_1, *dat_1_2, *dat_1_3;
    float *dat_2_1, *dat_2_2, *dat_2_3;
    float *dat_3_1, *dat_3_2, *dat_3_3;
    float *dat_4_1, *dat_4_2, *dat_4_3;
    float *dat_5_1, *dat_5_2, *dat_5_3;
    float *dat_6_1, *dat_6_2, *dat_6_3;
    // 寄存器复用
    register __m256 reg_1_1, reg_1_2, reg_1_3;
    register __m256 reg_2_1, reg_2_2, reg_2_3;
    register __m256 reg_3_1, reg_3_2, reg_3_3;
    register __m256 reg_4_1, reg_4_2, reg_4_3;
    // register __m256 reg_5_1, reg_5_2, reg_5_3;
    // register __m256 reg_6_1, reg_6_2, reg_6_3;
    register __m256 reg_o_1, reg_o_2;//, reg_o_3, reg_o_4;
    int i = 0;
    if (ms_omp < 1)
        i = 1;

    // 特判最后一行，防止越界
    int i_barrier = ms_omp + m_omp == height ? m_omp - BATCH_SIZE - 1 : m_omp - BATCH_SIZE;
    int actual_m_omp = ms_omp + m_omp == height ? m_omp - 1 : m_omp;

    for (; i <= i_barrier; i += BATCH_SIZE)
    {
        int j = 0;
        // int index_start = (ms_omp + i - 1) * width;
        float *in_0 = in + (ms_omp + i - 1) * width;
        float *in_1 = in + (ms_omp + i) * width;
        float *in_2 = in + (ms_omp + i + 1) * width;
        float *in_3 = in + (ms_omp + i + 2) * width;
        float *in_4 = in + (ms_omp + i + 3) * width;
        float *in_5 = in + (ms_omp + i + 4) * width;
        _mm_prefetch((const char *)in_0, _MM_HINT_T0);
        _mm_prefetch((const char *)in_1, _MM_HINT_T0);
        _mm_prefetch((const char *)in_2, _MM_HINT_T0);
        _mm_prefetch((const char *)in_3, _MM_HINT_T0);
        _mm_prefetch((const char *)in_4, _MM_HINT_T0);
        _mm_prefetch((const char *)in_5, _MM_HINT_T0);
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
        _mm_prefetch((const char *)dat_1_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_2_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_3_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_4_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_5_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_6_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
// load data
            reg_1_1 = _mm256_loadu_ps(dat_1_1);
            reg_1_2 = _mm256_loadu_ps(dat_1_2);
            reg_1_3 = _mm256_loadu_ps(dat_1_3);
            reg_2_1 = _mm256_loadu_ps(dat_2_1);
            reg_2_2 = _mm256_loadu_ps(dat_2_2);
            reg_2_3 = _mm256_loadu_ps(dat_2_3);
            reg_3_1 = _mm256_loadu_ps(dat_3_1);
            reg_3_2 = _mm256_loadu_ps(dat_3_2);
            reg_3_3 = _mm256_loadu_ps(dat_3_3);
            reg_4_1 = _mm256_loadu_ps(dat_4_1);
            reg_4_2 = _mm256_loadu_ps(dat_4_2);
            reg_4_3 = _mm256_loadu_ps(dat_4_3);
            // reg_5_1 = _mm256_loadu_ps(dat_5_1);
            // reg_5_2 = _mm256_loadu_ps(dat_5_2);
            // reg_5_3 = _mm256_loadu_ps(dat_5_3);
            // reg_6_1 = _mm256_loadu_ps(dat_6_1);
            // reg_6_2 = _mm256_loadu_ps(dat_6_2);
            // reg_6_3 = _mm256_loadu_ps(dat_6_3);
            reg_o_1 = reg_2_2;
            reg_o_2 = reg_3_2;
            // reg_o_3 = reg_4_2;
            // reg_o_4 = reg_5_2;

            // float sum_23_1[SIMD_LANE], sum_23_2[SIMD_LANE], sum_23_3[SIMD_LANE];
            // float sum_45_1[SIMD_LANE], sum_45_2[SIMD_LANE], sum_45_3[SIMD_LANE];
            // float sum_123_1[SIMD_LANE], sum_234_1[SIMD_LANE], sum_345_1[SIMD_LANE], sum_456_1[SIMD_LANE];
            // float sum_123_2[SIMD_LANE], sum_234_2[SIMD_LANE], sum_345_2[SIMD_LANE], sum_456_2[SIMD_LANE];
            // float sum_123_3[SIMD_LANE], sum_234_3[SIMD_LANE], sum_345_3[SIMD_LANE], sum_456_3[SIMD_LANE];
// float sum_1_12[SIMD_LANE], sum_2_12[SIMD_LANE], sum_3_12[SIMD_LANE], sum_4_12[SIMD_LANE], sum_5_12[SIMD_LANE], sum_6_12[SIMD_LANE];

// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_23_1[k] = dat_2_1[k] + dat_3_1[k];
//                 sum_23_2[k] = dat_2_2[k] + dat_3_2[k];
//                 sum_45_1[k] = dat_4_1[k] + dat_5_1[k];
//                 sum_45_2[k] = dat_4_2[k] + dat_5_2[k];
//             }
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_3_1); //sum_23_1
            reg_2_2 = _mm256_add_ps(reg_2_2, reg_3_2); //sum_23_2
            // reg_5_1 = _mm256_add_ps(reg_4_1, reg_5_1); //sum_45_1
            // reg_5_2 = _mm256_add_ps(reg_4_2, reg_5_2); //sum_45_2
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_123_1[k] = dat_1_1[k] + sum_23_1[k];
//                 sum_123_2[k] = dat_1_2[k] + sum_23_2[k];
//                 sum_234_1[k] = dat_4_1[k] + sum_23_1[k];
//                 sum_234_2[k] = dat_4_2[k] + sum_23_2[k];
//                 sum_345_1[k] = dat_3_1[k] + sum_45_1[k];
//                 sum_345_2[k] = dat_3_2[k] + sum_45_2[k];
//                 sum_456_1[k] = dat_6_1[k] + sum_45_1[k];
//                 sum_456_2[k] = dat_6_2[k] + sum_45_2[k];
//             }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_2_1); // sum_123_1
            reg_1_2 = _mm256_add_ps(reg_1_2, reg_2_2); // sum_123_2
            reg_2_1 = _mm256_add_ps(reg_4_1, reg_2_1); // sum_234_1
            reg_2_2 = _mm256_add_ps(reg_4_2, reg_2_2); // sum_234_2
            // reg_3_1 = _mm256_add_ps(reg_3_1, reg_5_1); // sum_345_1
            // reg_3_2 = _mm256_add_ps(reg_3_2, reg_5_2); // sum_345_2
            // reg_4_1 = _mm256_add_ps(reg_6_1, reg_5_1); // sum_456_1
            // reg_4_2 = _mm256_add_ps(reg_6_2, reg_5_2); // sum_456_2
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_23_3[k] = dat_2_3[k] + dat_3_3[k];
//                 sum_45_3[k] = dat_4_3[k] + dat_5_3[k];
//             }
            reg_2_3 = _mm256_add_ps(reg_2_3, reg_3_3); //sum_23_3
            // reg_5_3 = _mm256_add_ps(reg_4_3, reg_5_3); //sum_45_3
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_123_3[k] = dat_1_3[k] + sum_23_3[k];
//                 sum_234_3[k] = dat_4_3[k] + sum_23_3[k];
//                 sum_345_3[k] = dat_3_3[k] + sum_45_3[k];
//                 sum_456_3[k] = dat_6_3[k] + sum_45_3[k];
//             }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_2);
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_2_2);
            // reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_2);
            // reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_2);

            reg_1_3 = _mm256_add_ps(reg_1_3, reg_2_3); // sum_123_3
            reg_2_3 = _mm256_add_ps(reg_4_3, reg_2_3); // sum_234_3
            // reg_3_3 = _mm256_add_ps(reg_3_3, reg_5_3); // sum_345_3
            // reg_4_3 = _mm256_add_ps(reg_6_3, reg_5_3); // sum_456_3
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_123_1[k] += sum_123_2[k] + sum_123_3[k];
//                 sum_234_1[k] += sum_234_2[k] + sum_234_3[k];
//                 sum_345_1[k] += sum_345_2[k] + sum_345_3[k];
//                 sum_456_1[k] += sum_456_2[k] + sum_456_3[k];
//             }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_3); //sum_123_1
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_2_3); //sum_234_1
            // reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_3); //sum_345_1
            // reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_3); //sum_456_1
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 out_1[j + k] = 9 * in_1[j + k] - sum_123_1[k];
//                 out_2[j + k] = 9 * in_2[j + k] - sum_234_1[k];
//                 out_3[j + k] = 9 * in_3[j + k] - sum_345_1[k];
//                 out_4[j + k] = 9 * in_4[j + k] - sum_456_1[k];
//             }
            reg_o_1 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_1,reg_1_1);
            reg_o_2 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_2,reg_2_1);
            // reg_o_3 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_3,reg_3_1);
            // reg_o_4 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_4,reg_4_1);
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
                    
//                 out_1[j + k] = out_1[j + k] < 0.0F ? 0.0F : out_1[j + k];
//                 out_1[j + k] = out_1[j + k] > 255.0F ? 255.0F : out_1[j + k];
//                 out_2[j + k] = out_2[j + k] < 0.0F ? 0.0F : out_2[j + k];
//                 out_2[j + k] = out_2[j + k] > 255.0F ? 255.0F : out_2[j + k];
//                 out_3[j + k] = out_3[j + k] < 0.0F ? 0.0F : out_3[j + k];
//                 out_3[j + k] = out_3[j + k] > 255.0F ? 255.0F : out_3[j + k];
//                 out_4[j + k] = out_4[j + k] < 0.0F ? 0.0F : out_4[j + k];
//                 out_4[j + k] = out_4[j + k] > 255.0F ? 255.0F : out_4[j + k];
//             }
            reg_o_1 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_1);
            reg_o_2 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_2);
            // reg_o_3 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_3);
            // reg_o_4 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_4);
            reg_o_1 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_1);
            reg_o_2 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_2);
            // reg_o_3 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_3);
            // reg_o_4 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_4);
            _mm256_storeu_ps(out_1 + j, reg_o_1);
            _mm256_storeu_ps(out_2 + j, reg_o_2);
// 复用寄存器
            reg_1_1 = _mm256_loadu_ps(dat_5_1);
            reg_1_2 = _mm256_loadu_ps(dat_5_2);
            reg_1_3 = _mm256_loadu_ps(dat_5_3);
            reg_2_1 = _mm256_loadu_ps(dat_6_1);
            reg_2_2 = _mm256_loadu_ps(dat_6_2);
            reg_2_3 = _mm256_loadu_ps(dat_6_3);
            reg_o_1 = reg_4_2;
            reg_o_2 = reg_1_2;
            reg_1_1 = _mm256_add_ps(reg_4_1, reg_1_1); //sum_45_1
            reg_1_2 = _mm256_add_ps(reg_4_2, reg_1_2); //sum_45_2
            reg_3_1 = _mm256_add_ps(reg_3_1, reg_1_1); // sum_345_1
            reg_3_2 = _mm256_add_ps(reg_1_2, reg_3_2); // sum_345_2
            reg_4_1 = _mm256_add_ps(reg_1_1, reg_2_1); // sum_456_1
            reg_4_2 = _mm256_add_ps(reg_1_2, reg_2_2); // sum_456_2
            reg_1_3 = _mm256_add_ps(reg_4_3, reg_1_3); //sum_45_3
            reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_2);
            reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_2);
            reg_3_3 = _mm256_add_ps(reg_3_3, reg_1_3); // sum_345_3
            reg_4_3 = _mm256_add_ps(reg_2_3, reg_1_3); // sum_456_3
            reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_3); //sum_345_1
            reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_3); //sum_456_1
            reg_o_1 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_1,reg_3_1);
            reg_o_2 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_2,reg_4_1);
            reg_o_1 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_1);
            reg_o_2 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_2);
            reg_o_1 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_1);
            reg_o_2 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_2);
            _mm256_storeu_ps(out_3 + j, reg_o_1);
            _mm256_storeu_ps(out_4 + j, reg_o_2);
        }

        // for (; j < width - 1; j++)
        { // 处理最后的
            int last_size = width - j;
            unsigned int mask0[8] = {0};
            for (int k = 0; k < last_size; k++) {
                mask0[k] = 0xffffffff;
            }
            register __m256i r_mask = _mm256_loadu_si256((__m256i const *)mask0);
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

            reg_1_1 = _mm256_maskload_ps(dat_1_1, r_mask);
            reg_1_2 = _mm256_maskload_ps(dat_1_2, r_mask);
            reg_1_3 = _mm256_maskload_ps(dat_1_3, r_mask);
            reg_2_1 = _mm256_maskload_ps(dat_2_1, r_mask);
            reg_2_2 = _mm256_maskload_ps(dat_2_2, r_mask);
            reg_2_3 = _mm256_maskload_ps(dat_2_3, r_mask);
            reg_3_1 = _mm256_maskload_ps(dat_3_1, r_mask);
            reg_3_2 = _mm256_maskload_ps(dat_3_2, r_mask);
            reg_3_3 = _mm256_maskload_ps(dat_3_3, r_mask);
            reg_4_1 = _mm256_maskload_ps(dat_4_1, r_mask);
            reg_4_2 = _mm256_maskload_ps(dat_4_2, r_mask);
            reg_4_3 = _mm256_maskload_ps(dat_4_3, r_mask);
            // reg_5_1 = _mm256_maskload_ps(dat_5_1, r_mask);
            // reg_5_2 = _mm256_maskload_ps(dat_5_2, r_mask);
            // reg_5_3 = _mm256_maskload_ps(dat_5_3, r_mask);
            // reg_6_1 = _mm256_maskload_ps(dat_6_1, r_mask);
            // reg_6_2 = _mm256_maskload_ps(dat_6_2, r_mask);
            // reg_6_3 = _mm256_maskload_ps(dat_6_3, r_mask);
            reg_o_1 = reg_2_2;
            reg_o_2 = reg_3_2;
            // reg_o_3 = reg_4_2;
            // reg_o_4 = reg_5_2;
            // float sum_23_1[SIMD_LANE], sum_23_2[SIMD_LANE], sum_23_3[SIMD_LANE];
            // float sum_45_1[SIMD_LANE], sum_45_2[SIMD_LANE], sum_45_3[SIMD_LANE];
            // float sum_123_1[SIMD_LANE], sum_234_1[SIMD_LANE], sum_345_1[SIMD_LANE], sum_456_1[SIMD_LANE];
            // float sum_123_2[SIMD_LANE], sum_234_2[SIMD_LANE], sum_345_2[SIMD_LANE], sum_456_2[SIMD_LANE];
            // float sum_123_3[SIMD_LANE], sum_234_3[SIMD_LANE], sum_345_3[SIMD_LANE], sum_456_3[SIMD_LANE];
            // float sum_1_12[SIMD_LANE], sum_2_12[SIMD_LANE], sum_3_12[SIMD_LANE], sum_4_12[SIMD_LANE], sum_5_12[SIMD_LANE], sum_6_12[SIMD_LANE];

            // for (int k = 0; k < last_size; k++)
            // {
            //     sum_23_1[k] = dat_2_1[k] + dat_3_1[k];
            //     sum_23_2[k] = dat_2_2[k] + dat_3_2[k];
            //     sum_45_1[k] = dat_4_1[k] + dat_5_1[k];
            //     sum_45_2[k] = dat_4_2[k] + dat_5_2[k];
            // }
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_3_1); //sum_23_1
            reg_2_2 = _mm256_add_ps(reg_2_2, reg_3_2); //sum_23_2
            // reg_5_1 = _mm256_add_ps(reg_4_1, reg_5_1); //sum_45_1
            // reg_5_2 = _mm256_add_ps(reg_4_2, reg_5_2); //sum_45_2

            // for (int k = 0; k < last_size; k++)
            // {
            //     sum_123_1[k] = dat_1_1[k] + sum_23_1[k];
            //     sum_123_2[k] = dat_1_2[k] + sum_23_2[k];
            //     sum_234_1[k] = dat_4_1[k] + sum_23_1[k];
            //     sum_234_2[k] = dat_4_2[k] + sum_23_2[k];
            //     sum_345_1[k] = dat_3_1[k] + sum_45_1[k];
            //     sum_345_2[k] = dat_3_2[k] + sum_45_2[k];
            //     sum_456_1[k] = dat_6_1[k] + sum_45_1[k];
            //     sum_456_2[k] = dat_6_2[k] + sum_45_2[k];
            // }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_2_1); // sum_123_1
            reg_1_2 = _mm256_add_ps(reg_1_2, reg_2_2); // sum_123_2
            reg_2_1 = _mm256_add_ps(reg_4_1, reg_2_1); // sum_234_1
            reg_2_2 = _mm256_add_ps(reg_4_2, reg_2_2); // sum_234_2
            // reg_3_1 = _mm256_add_ps(reg_3_1, reg_5_1); // sum_345_1
            // reg_3_2 = _mm256_add_ps(reg_3_2, reg_5_2); // sum_345_2
            // reg_4_1 = _mm256_add_ps(reg_6_1, reg_5_1); // sum_456_1
            // reg_4_2 = _mm256_add_ps(reg_6_2, reg_5_2); // sum_456_2

            // for (int k = 0; k < last_size; k++)
            // {
            //     sum_23_3[k] = dat_2_3[k] + dat_3_3[k];
            //     sum_45_3[k] = dat_4_3[k] + dat_5_3[k];
            // }
            reg_2_3 = _mm256_add_ps(reg_2_3, reg_3_3); //sum_23_3
            // reg_5_3 = _mm256_add_ps(reg_4_3, reg_5_3); //sum_45_3

            // for (int k = 0; k < last_size; k++)
            // {
            //     sum_123_3[k] = dat_1_3[k] + sum_23_3[k];
            //     sum_234_3[k] = dat_4_3[k] + sum_23_3[k];
            //     sum_345_3[k] = dat_3_3[k] + sum_45_3[k];
            //     sum_456_3[k] = dat_6_3[k] + sum_45_3[k];
            // }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_2);
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_2_2);
            // reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_2);
            // reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_2);

            reg_1_3 = _mm256_add_ps(reg_1_3, reg_2_3); // sum_123_3
            reg_2_3 = _mm256_add_ps(reg_4_3, reg_2_3); // sum_234_3
            // reg_3_3 = _mm256_add_ps(reg_3_3, reg_5_3); // sum_345_3
            // reg_4_3 = _mm256_add_ps(reg_6_3, reg_5_3); // sum_456_3
            // for (int k = 0; k < last_size; k++)
            // {
            //     sum_123_1[k] += sum_123_2[k] + sum_123_3[k];
            //     sum_234_1[k] += sum_234_2[k] + sum_234_3[k];
            //     sum_345_1[k] += sum_345_2[k] + sum_345_3[k];
            //     sum_456_1[k] += sum_456_2[k] + sum_456_3[k];
            // }

            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_3); //sum_123_1
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_2_3); //sum_234_1
            // reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_3); //sum_345_1
            // reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_3); //sum_456_1
            // for (int k = 0; k < last_size; k++)
            // {
            //     out_1[j + k] = 9 * in_1[j + k] - sum_123_1[k];
            //     out_2[j + k] = 9 * in_2[j + k] - sum_234_1[k];
            //     out_3[j + k] = 9 * in_3[j + k] - sum_345_1[k];
            //     out_4[j + k] = 9 * in_4[j + k] - sum_456_1[k];
            // }
            reg_o_1 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_1,reg_1_1);
            reg_o_2 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_2,reg_2_1);
            // reg_o_3 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_3,reg_3_1);
            // reg_o_4 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_4,reg_4_1);

            // for (int k = 0; k < last_size; k++)
            // {
            //     out_1[j + k] = out_1[j + k] < 0.0F ? 0.0F : out_1[j + k];
            //     out_1[j + k] = out_1[j + k] > 255.0F ? 255.0F : out_1[j + k];
            //     out_2[j + k] = out_2[j + k] < 0.0F ? 0.0F : out_2[j + k];
            //     out_2[j + k] = out_2[j + k] > 255.0F ? 255.0F : out_2[j + k];
            //     out_3[j + k] = out_3[j + k] < 0.0F ? 0.0F : out_3[j + k];
            //     out_3[j + k] = out_3[j + k] > 255.0F ? 255.0F : out_3[j + k];
            //     out_4[j + k] = out_4[j + k] < 0.0F ? 0.0F : out_4[j + k];
            //     out_4[j + k] = out_4[j + k] > 255.0F ? 255.0F : out_4[j + k];
            // }
            reg_o_1 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_1);
            reg_o_2 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_2);
            // reg_o_3 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_3);
            // reg_o_4 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_4);
            reg_o_1 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_1);
            reg_o_2 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_2);
            // reg_o_3 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_3);
            // reg_o_4 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_4);
            _mm256_maskstore_ps(out_1 + j, r_mask, reg_o_1);
            _mm256_maskstore_ps(out_2 + j, r_mask, reg_o_2);
            
// 复用寄存器
            reg_1_1 = _mm256_maskload_ps(dat_5_1, r_mask);
            reg_1_2 = _mm256_maskload_ps(dat_5_2, r_mask);
            reg_1_3 = _mm256_maskload_ps(dat_5_3, r_mask);
            reg_2_1 = _mm256_maskload_ps(dat_6_1, r_mask);
            reg_2_2 = _mm256_maskload_ps(dat_6_2, r_mask);
            reg_2_3 = _mm256_maskload_ps(dat_6_3, r_mask);
            reg_o_1 = reg_4_2;
            reg_o_2 = reg_1_2;
            reg_1_1 = _mm256_add_ps(reg_4_1, reg_1_1); //sum_45_1
            reg_1_2 = _mm256_add_ps(reg_4_2, reg_1_2); //sum_45_2
            reg_3_1 = _mm256_add_ps(reg_3_1, reg_1_1); // sum_345_1
            reg_3_2 = _mm256_add_ps(reg_1_2, reg_3_2); // sum_345_2
            reg_4_1 = _mm256_add_ps(reg_1_1, reg_2_1); // sum_456_1
            reg_4_2 = _mm256_add_ps(reg_1_2, reg_2_2); // sum_456_2
            reg_1_3 = _mm256_add_ps(reg_4_3, reg_1_3); //sum_45_3
            reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_2);
            reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_2);
            reg_3_3 = _mm256_add_ps(reg_3_3, reg_1_3); // sum_345_3
            reg_4_3 = _mm256_add_ps(reg_2_3, reg_1_3); // sum_456_3
            reg_3_1 = _mm256_add_ps(reg_3_1, reg_3_3); //sum_345_1
            reg_4_1 = _mm256_add_ps(reg_4_1, reg_4_3); //sum_456_1
            reg_o_1 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_1,reg_3_1);
            reg_o_2 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_2,reg_4_1);
            reg_o_1 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_1);
            reg_o_2 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_2);
            reg_o_1 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_1);
            reg_o_2 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_2);
            _mm256_maskstore_ps(out_3 + j, r_mask, reg_o_1);
            _mm256_maskstore_ps(out_4 + j, r_mask, reg_o_1);
        }
    }
    // 注意不要越界了
    // 处理剩下的
    for (; i < actual_m_omp; i++)
    {
        int j = 0;
        // int index_start = (ms_omp + i - 1) * width;
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
        _mm_prefetch((const char *)dat_1_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_2_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
        _mm_prefetch((const char *)dat_3_1 + PREFETCH_DISTANCE, _MM_HINT_T0);
// load data
            reg_1_1 = _mm256_loadu_ps(dat_1_1);
            reg_1_2 = _mm256_loadu_ps(dat_1_2);
            reg_1_3 = _mm256_loadu_ps(dat_1_3);
            reg_2_1 = _mm256_loadu_ps(dat_2_1);
            reg_2_2 = _mm256_loadu_ps(dat_2_2);
            reg_2_3 = _mm256_loadu_ps(dat_2_3);
            reg_3_1 = _mm256_loadu_ps(dat_3_1);
            reg_3_2 = _mm256_loadu_ps(dat_3_2);
            reg_3_3 = _mm256_loadu_ps(dat_3_3);
            reg_o_1 = reg_2_2;

            // float sum_23_1[SIMD_LANE], sum_23_2[SIMD_LANE], sum_23_3[SIMD_LANE];
            // float sum_123_1[SIMD_LANE];
            // float sum_123_2[SIMD_LANE];
            // float sum_123_3[SIMD_LANE];
// float sum_1_12[SIMD_LANE], sum_2_12[SIMD_LANE], sum_3_12[SIMD_LANE], sum_4_12[SIMD_LANE], sum_5_12[SIMD_LANE], sum_6_12[SIMD_LANE];
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_23_1[k] = dat_2_1[k] + dat_3_1[k];
//                 sum_23_2[k] = dat_2_2[k] + dat_3_2[k];
//             }
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_3_1); //sum_23_1
            reg_2_2 = _mm256_add_ps(reg_2_2, reg_3_2); //sum_23_2
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_123_1[k] = dat_1_1[k] + sum_23_1[k];
//                 sum_123_2[k] = dat_1_2[k] + sum_23_2[k];
//             }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_2_1); // sum_123_1
            reg_1_2 = _mm256_add_ps(reg_1_2, reg_2_2); // sum_123_2
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_23_3[k] = dat_2_3[k] + dat_3_3[k];
//             }
            reg_2_3 = _mm256_add_ps(reg_2_3, reg_3_3); //sum_23_3
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_123_3[k] = dat_1_3[k] + sum_23_3[k];
//             }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_2);

            reg_1_3 = _mm256_add_ps(reg_1_3, reg_2_3); // sum_123_3
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 sum_123_1[k] += sum_123_2[k] + sum_123_3[k];
//             }
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_3); //sum_123_1
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 out_1[j + k] = 9 * in_1[j + k] - sum_123_1[k];
//             }
            reg_o_1 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_1,reg_1_1);
// #pragma unroll(SIMD_LANE)
//             for (int k = 0; k < SIMD_LANE; k++)
//             {
//                 out_1[j + k] = out_1[j + k] < 0 ? 0 : out_1[j + k];
//                 out_1[j + k] = out_1[j + k] > 255 ? 255 : out_1[j + k];
//             }
            reg_o_1 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_1);
            reg_o_1 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_1);
            _mm256_storeu_ps(out_1 + j, reg_o_1);
        }

        for (; j < width - 1; j++)
        { // 处理最后的
            // float val =
            //     -in_0[j - 1] - in_0[j] - in_0[j + 1] - in_1[j - 1] + 8 * in_1[j] - in_1[j + 1] - in_2[j - 1] - in_2[j] - in_2[j + 1];

            // val = (val < 0.0F ? 0.0F : val);
            // val = (val > 255.0F ? 255.0F : val);

            // out_1[j] = val;
            int last_size = width - j;
            int mask0[8] = {0};
            for (int k = 0; k < last_size; k++) {
                mask0[k] = 0xffffffff;
            }
            register __m256i r_mask = _mm256_loadu_si256((__m256i const *)mask0);
            reg_1_1 = _mm256_maskload_ps(dat_1_1, r_mask);
            reg_1_2 = _mm256_maskload_ps(dat_1_2, r_mask);
            reg_1_3 = _mm256_maskload_ps(dat_1_3, r_mask);
            reg_2_1 = _mm256_maskload_ps(dat_2_1, r_mask);
            reg_2_2 = _mm256_maskload_ps(dat_2_2, r_mask);
            reg_2_3 = _mm256_maskload_ps(dat_2_3, r_mask);
            reg_3_1 = _mm256_maskload_ps(dat_3_1, r_mask);
            reg_3_2 = _mm256_maskload_ps(dat_3_2, r_mask);
            reg_3_3 = _mm256_maskload_ps(dat_3_3, r_mask);
            reg_o_1 = reg_2_2;
            
            reg_2_1 = _mm256_add_ps(reg_2_1, reg_3_1); //sum_23_1
            reg_2_2 = _mm256_add_ps(reg_2_2, reg_3_2); //sum_23_2

            reg_1_1 = _mm256_add_ps(reg_1_1, reg_2_1); // sum_123_1
            reg_1_2 = _mm256_add_ps(reg_1_2, reg_2_2); // sum_123_2

            reg_2_3 = _mm256_add_ps(reg_2_3, reg_3_3); //sum_23_3

            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_2);

            reg_1_3 = _mm256_add_ps(reg_1_3, reg_2_3); // sum_123_3
            reg_1_1 = _mm256_add_ps(reg_1_1, reg_1_3); //sum_123_1
            reg_o_1 = _mm256_fmsub_ps(_mm256_set1_ps(9.0F),reg_o_1,reg_1_1);

            reg_o_1 = _mm256_min_ps(_mm256_set1_ps(255.0F),reg_o_1);
            reg_o_1 = _mm256_max_ps(_mm256_set1_ps(0.0F),reg_o_1);
            _mm256_maskstore_ps(out_1 + j, r_mask, reg_o_1);
        }
    }
}
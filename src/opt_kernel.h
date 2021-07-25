#include <omp.h>
#include <immintrin.h>
#include <cstring>
#include <cstdio>

inline void ApplyStencil_simple_omp_avx2_float(float *in, float *out, int width, int height)
{
    // 开头对齐
    int counts = width * height - 2 * (width + 1);
    float *alignPtr = (float *)(((unsigned long long)(out + width + 1) + 31ULL) & ~31ULL);
    int alignOffset = alignPtr - (out + width + 1);
    counts -= alignOffset;
    in += width + 1;
    out += width + 1;
    for (int i = 0; i < alignOffset; ++i, ++in, ++out)
    {
        float val = -in[-width - 1] - in[-width] - in[-width + 1] - in[-1] + 8 * in[0] - in[1] - in[width - 1] - in[width] - in[width + 1];
        val = (val < 0 ? 0 : val);
        val = (val > 255 ? 255 : val);
        out[0] = val;
    }

#pragma omp parallel
    {
        int threadsNum = omp_get_num_threads();
        int threadsId = omp_get_thread_num();
        int countPerThread = (counts / threadsNum + 7) & (~7); // 对齐到16Bytes
        int startCount = threadsId * countPerThread;
        // 超出的线程直接join
        if (startCount < counts)
        {
            bool allVector = (startCount + countPerThread <= counts);
            if (!allVector)
                countPerThread -= 8;

            int i = 0;
            int index = startCount;
            for (; i < countPerThread; i += 8, index += 8)
            {
                // x1+x2
                __m256 val1 = _mm256_add_ps(_mm256_loadu_ps(&in[index - width - 1]),
                                            _mm256_loadu_ps(&in[index - width]));
                // x3+x4
                __m256 val2 = _mm256_add_ps(_mm256_loadu_ps(&in[index - width + 1]),
                                            _mm256_loadu_ps(&in[index - 1]));
                // 8 * x5 - x6
                __m256 val3 = _mm256_fmsub_ps(_mm256_set1_ps(8.0),
                                              _mm256_loadu_ps(&in[index]),
                                              _mm256_loadu_ps(&in[index + 1]));
                // x7+x8
                __m256 val4 = _mm256_add_ps(_mm256_loadu_ps(&in[index + width - 1]),
                                            _mm256_loadu_ps(&in[index + width]));
                // x9
                __m256 val5 = _mm256_loadu_ps(&in[index + width + 1]);

                // x1+x2+x3+x4
                __m256 val10 = _mm256_add_ps(val1, val2);
                // 8*x5-x6-x7-x8
                __m256 val11 = _mm256_sub_ps(val3, val4);
                // x1+x2+x3+x4+x9
                __m256 val20 = _mm256_add_ps(val10, val5);
                // val
                __m256 val = _mm256_sub_ps(val11, val20);

                val = _mm256_max_ps(val, _mm256_set1_ps(0));
                val = _mm256_min_ps(val, _mm256_set1_ps(255));

                _mm256_stream_ps(&out[index], val);
                // _mm_prefetch(&in[index + 2 * width], _MM_HINT_T1);
            }

            if (!allVector)
            {
                for (; i < countPerThread + 8; i++, index++)
                {
                    float val = -in[index - width - 1] - in[index - width] - in[index - width + 1] - in[index - 1] + 8 * in[index] - in[index + 1] - in[index + width - 1] - in[index + width] - in[index + width + 1];
                    val = (val < 0 ? 0 : val);
                    val = (val > 255 ? 255 : val);
                    out[index] = val;
                }
            }
        }
    }
}
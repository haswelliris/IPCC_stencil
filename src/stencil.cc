#include "stencil.h"
#include "my_kernel.h"
#include "opt_kernel.h"

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {
  
  const int width  = img_in.width;
  const int height = img_in.height;

  P * in  = img_in.pixel;
  P * out = img_out.pixel;
  // 高效简洁的实现
  ApplyStencil_simple_omp_avx2_float(in, out, width, height);

  // 纯手动向量化
  // ApplyStencil_simd_avx2_float(in, out, width, height);

  // 不用simd
  // ApplyStencil_avx256_float(in, out, width, height);

}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);

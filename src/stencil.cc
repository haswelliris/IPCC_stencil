#include "stencil.h"
#include "my_kernel.h"

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {
  
  const int width  = img_in.width;
  const int height = img_in.height;

  P * in  = img_in.pixel;
  P * out = img_out.pixel;
  ApplyStencil_avx256_float(in, out, width, height);

}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);

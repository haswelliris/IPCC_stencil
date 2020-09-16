#include <omp.h>

#define ALIGN 64
#define NTHREADS 64

inline void ApplyStencil_avx256_float(float* in, float* out, int width, int height) {
omp_set_num_threads(NTHREADS);
#pragma omp parallel for default(shared)
  for (int j = 1; j < width-1; j++) {
      for (int i = 1; i < height-1; i++) {

	int im1jm1 =(i-1)*width + j-1;
	int im1j   =(i-1)*width + j;
	int im1jp1 =(i-1)*width + j+1;
	int ijm1   =(i  )*width + j-1;
	int ij     =(i  )*width + j;
	int ijp1   =(i  )*width + j+1;
	int ip1jm1 =(i+1)*width + j-1;
	int ip1j   =(i+1)*width + j;
	int ip1jp1 =(i+1)*width + j+1;
	float val = 
	  -in[im1jm1] -   in[im1j] - in[im1jp1] 
	  -in[ijm1]   +   8*in[ij] - in[ijp1] 
	  -in[ip1jm1] -   in[ip1j] - in[ip1jp1];

	val = (val < 0   ? 0   : val);
	val = (val > 255 ? 255 : val);

	out[i*width + j] = val;

      }

  }
}
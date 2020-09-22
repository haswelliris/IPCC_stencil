#include "stencil.h"
#include "my_kernel.h"
#include "mpi.h"

#define SPLIT_RATIO 0.52

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {
  
  //printf("RANK#0: breakpoint 1 \n");
  const int width  = img_in.width;
  const int height = img_in.height;

  P * in  = img_in.pixel;
  P * out = img_out.pixel;
  //printf("RANK#0: breakpoint 2 \n");
  // 划分任务
  int height1 = ((int) (height * SPLIT_RATIO / 4)) * 4 + 1;
  int height2 = height - height1 + 1;
  int height2_start = height1 - 1;
  int width1 = width;
  int width2 = width;
  int *params = (int*)malloc(10 * sizeof(int));
  params[0] = height2; params[1] = width;
  
  //printf("RANK#0: height1:%d height2:%d height2_start:%d \n", height1,height2,height2_start);
  //printf("RANK#0: breakpoint 3 \n");
  // MPI发送数据
  MPI_Request request_sending1;
  MPI_Request request_sending2;
  MPI_Request request_recving;
  MPI_Isend(params, 2,
      MPI_INT, 1, 11, MPI_COMM_WORLD, &request_sending1);
  MPI_Isend(in + height2_start * width, height2 * width,
      MPI_FLOAT, 1, 12, MPI_COMM_WORLD, &request_sending2);
  // 先把接收操作挂出去
  MPI_Irecv( out + height2_start + width, (height2 - 2) * width,
      MPI_FLOAT, 1, 20, MPI_COMM_WORLD, &request_recving);
        //printf("RANK#0: breakpoint 4 \n");
  // 计算
  ApplyStencil_avx256_float(in, out, width1, height1);
        //printf("RANK#0: breakpoint 5 \n");
  // 等待写回
  MPI_Status status;
  MPI_Wait(&request_recving, &status);
  free(params);
        //printf("RANK#0: breakpoint 6 \n");

}

void ApplyStencil_worker() {
  //printf("RANK#1: breakpoint 1 \n");
  int* params = (int*)malloc(10 * sizeof(int));
  int height, width;
  MPI_Status status;
  MPI_Request request_recving;
  MPI_Recv(params, 2,
      MPI_INT, 0, 11, MPI_COMM_WORLD, &status);
  height = params[0];
  width = params[1];
  // 申请空间
  float* in = (float *)_mm_malloc(height * width * sizeof(float), ALIGN);
  //printf("RANK#1: breakpoint 2 \n");
  // 异步接收
  MPI_Irecv( in, height * width ,
      MPI_FLOAT, 0, 12, MPI_COMM_WORLD, &request_recving);
  float* out = (float *)_mm_malloc(height * width * sizeof(float) , ALIGN);
  // 接收完毕
  MPI_Wait(&request_recving, &status);
  //printf("RANK#1: breakpoint 3 \n");
  // 计算
  ApplyStencil_avx256_float(in, out, width, height);
  //printf("RANK#1: breakpoint 4 \n");
  // 数据传回
  // 注意第一行和最后一行没有数据
  MPI_Request request_sending;
  MPI_Isend(out + width, (height - 2) * width,
      MPI_FLOAT, 0, 20, MPI_COMM_WORLD, &request_sending);
  free(params);
  _mm_free(in);
  //printf("RANK#1: breakpoint 5 \n");
  MPI_Wait(&request_sending, &status);
  _mm_free(out);
  //printf("RANK#1: breakpoint 6 \n");
}

template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);

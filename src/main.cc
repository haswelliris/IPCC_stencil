#include <cmath>
#include <cstdlib>
#include <sys/time.h>
#include "image.h"
#include "stencil.h"
#include "mpi.h"

#define P float

#define max(a,b) ((a) > (b) ? (a) : (b))

/**
 * insert MPI support
 */
int myid, source, numprocs;
/**
 * insert MPI support
 */
int nTrials;
int skipTrials;

double wtime() {
  struct timeval v;
  struct timezone z;

  int i = gettimeofday(&v,&z);
  return ((double)v.tv_sec + (double)v.tv_usec*1.0e-6);
}



int main(int argc, char** argv) {
  
  nTrials = 10;

  /**
   * insert MPI support
   */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  // Main Process for timing
  if(myid != 0 ) {
    for (int iTrial = 1; iTrial <= nTrials; iTrial++) {
      // printf("RANK#%d: Trial: %d\n", myid, iTrial);
      ApplyStencil_worker();
    }
    MPI_Finalize();
    return 0;

  } else {
  // Main Process for timing

  ImageClass<P> img_in (argv[1]);
  ImageClass<P> img_out(img_in.width, img_in.height);

  printf("\n\033[1mEdge detection with a 3x3 stencil\033[0m\n");
  printf("\nImage size: %d x %d\n\n", img_in.width, img_in.height);
  printf("\033[1m%5s %15s \n", "Trial", "Time, ms"); fflush(stdout);
    double t, dt;

    for (int iTrial = 1; iTrial <= nTrials; iTrial++) {

      // add for test
      // printf("RANK#%d: Trial: %d\n", myid, iTrial);

      const double t0 = wtime();
      ApplyStencil(img_in, img_out);
      const double t1 = wtime();

      const double ts   = t1-t0; // time in seconds
      const double tms  = ts*1.0e3; // time in milliseconds

        t  += tms; 
        dt += tms*tms;

        printf("%5d %15.3f \n", 
        iTrial, tms);
      fflush(stdout);
    }

  /**
   * insert MPI support
   */
    MPI_Finalize();
  /**
   * insert MPI support
   */

    printf("-----------------------------------------------------\n");
    printf("%s   %8.1f %s\n",
    "Total :", t,"ms");
    printf("-----------------------------------------------------\n");
    FILE * fp;

    fp = fopen ("data.txt", "w+");
    P * out = img_out.pixel;
    for (int j = 1; j < img_in.width-1; j++) {
        for (int i = 1; i < img_in.height-1; i++) {
            fprintf(fp,"%lf\n",out[i*img_in.width + j]);
        }
    }
    fclose(fp);
    img_out.WriteToFile("output.png");
    printf("\nOutput written into data.txt and output.png\n");
    return 0;
  }
}

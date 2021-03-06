/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <xmmintrin.h>
#include <math.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

struct complex {
  float real;
  float imag;
};

/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%.3f + %.3fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%.3f + %.3fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
  }
}
void write_out_check(struct complex ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("[%.3f,%.3fi|%p]", a[i][j].real, a[i][j].imag,&a[i][j]);
    }
    printf("[%.3f,%.3fi|%p]\n", a[i][dim2-1].real, a[i][dim2-1].imag,&a[i][dim2-1]);
  }
}

/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);
  memset(new_matrix,0,sizeof(struct complex) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}

void free_matrix(struct complex ** matrix) {
  free (matrix[0]); /* free the contents */
  free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
  int i, j;
  struct complex ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
  struct complex ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
    /* evenly generate values in the range [0, 256)*/
      result[i][j].real = (float)(random()) / 0xFFFFFF;
      result[i][j].imag = (float)(random()) / 0xFFFFFF;

    /* at no loss of precision, negate the values sometimes */
    /* so the range is now (-256, 256)*/
    if (random() & 1) result[i][j].real = -result[i][j].real;
    if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
/* returns number of differing values */
int check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
  int i, j;
  double sum_abs_diff = 0.0;
  int errs = 0;
  const double EPSILON = 0.0625;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      double diff;
      diff = abs(control[i][j].real - result[i][j].real);
      sum_abs_diff = sum_abs_diff + diff;
      if (diff > EPSILON) errs++;

      diff = abs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff = sum_abs_diff + diff;
      if (diff > EPSILON) errs++;
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
      sum_abs_diff, EPSILON);
  }

  return errs;
}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int i, j, k;

  for ( i = 0; i < a_dim1; i++ ) {
    for( j = 0; j < b_dim2; j++ ) {
      struct complex sum;
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        // the following code does: sum += A[i][k] * B[k][j];
        struct complex product;
        product.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        product.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
        sum.real += product.real;
        sum.imag += product.imag;
      }
      C[i][j] = sum;
    }
  }
}
/*
* This method is called when the matrix size is too small to be affected by threading
*/
void no_omp_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2){
  for (int i = 0; i < a_dim1; i++ ) {
    for(int j = 0; j < b_dim2; j++ ) {
      float sum_real = 0.0;
      float sum_imag = 0.0;
      for (int k = 0; k < a_dim2; k++ ) {
        sum_real += A[i][k].real * B[j][k].real - A[i][k].imag * B[j][k].imag;
        sum_imag += A[i][k].real * B[j][k].imag + A[i][k].imag * B[j][k].real;
      }
      C[i][j].real = sum_real;
      C[i][j].imag = sum_imag;
    }
  }
}
/*
* Precondition = B is transposed
*/
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2) {
  int s;
  /*
  * If there's less than 35 rows then dont use any threads.
  * Otherwise use the fancy equations of a_row * 0.12 to get the number
  * of threads needed. This number didn't come out of the air. I ran the program
  * over and over with different matrix sizes and different number of threads
  * and used the data gathered to calculate that value
  *
  * There is no logical reason to use more thant 64 threads for this program on 
  * stoker "four processors. Each processor has eight out-of-order pipelined, superscalar cores"
  */
  if(a_dim1<35){
    no_omp_matmul(A,B,C,a_dim1,a_dim2,b_dim2);
    return;
  } else {
    s = (int) floor(((float)a_dim1*0.12));
    if(s>64){
      s=64;
    }
  }
  /*
  * The workload is always the same, so no point in usign guided or dynamic scheduling
  */
  omp_set_dynamic(0);
  omp_set_num_threads(s); 
  #pragma omp parallel for
  for (int i = 0; i < a_dim1; i++ ) {
    for(int j = 0; j < b_dim2; j++ ) {
      /*
      * Use these variables to store intermediate results, these are unique to each thread
      * These will most likely be in register all of the time since the compiler isn't stoopid
      * enough to let these sit in memory.
      * Notice the lack of struct complex ** sum and struct complex ** product in the code.
      * We got rid of these to reduce memory allocation overhears.
      * Each thread will only ever write to 1 location, and will read from many so there's
      * no need for a critical section
      */
      float sum_real = 0.0;
      float sum_imag = 0.0;
      for (int k = 0; k < a_dim2; k++ ) {
        /*
        * Vectorisation is too mainstream. We tried it but it just slowed out code down.
        * GCC -O3 is too smart.
        */
        sum_real += A[i][k].real * B[j][k].real - A[i][k].imag * B[j][k].imag;
        sum_imag += A[i][k].real * B[j][k].imag + A[i][k].imag * B[j][k].real;
      }
      C[i][j].real = sum_real;
      C[i][j].imag = sum_imag;
    }
  }
}
/*
* Transposes matrix B into T.
*/
void transpose(struct complex ** B,struct complex **T,int r, int c){
  for(int i = 0;i< r;i++){
    for(int j = 0; j < c;j++){
      T[j][i] = B[i][j];
    }
  }
}
long long time_diff(struct timeval * start, struct timeval * end) {
  return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_usec - start->tv_usec);
}
int main(int argc, char ** argv)
{
  struct complex ** A, ** B, ** C,**T;
  struct complex ** control_matrix;
  long long control_time, mul_time,mul_t;
  double speedup,speedup_2;
  int a_dim1, a_dim2, b_dim1, b_dim2, errs;
  struct timeval pre_time, start_time,stop_time_t, stop_time;

  if ( argc != 5 ) {
    fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /* check the matrix sizes are compatible */
  if ( a_dim2 != b_dim1 ) {
    fprintf(stderr,
      "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
      a_dim2, b_dim1);
    exit(1);
  }

  /* allocate the matrices */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  T = new_empty_matrix(b_dim2, b_dim1);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  for (int i = 0; i < a_dim1; i++ ) {
    for(int j = 0; j < b_dim2; j++ ) {
      struct complex sum;
      sum.real = 0.0;
      sum.imag = 0.0;
      C[i][j] = sum;
    }
  }


  DEBUGGING( {
    printf("matrix A:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\nmatrix B:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\n");
  } )

  /* record control start time */
  gettimeofday(&pre_time, NULL);

  /* use a simple matmul routine to produce control result */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* perform matrix multiplication */
  transpose(B,T,b_dim1,b_dim2);
  team_matmul(A, T, C, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL); 

  /* compute elapsed times and speedup factor */
 control_time = time_diff(&pre_time, &start_time);
  mul_time = time_diff(&start_time, &stop_time);
  speedup = (float) control_time / mul_time;

  printf("A:%lld\n", mul_time);
  printf("B:%lld\n", control_time);
  if (mul_time > 0 && control_time > 0) {
    printf("speedup: %.2fx\n", speedup);
  }
  /* now check that the team's matmul routine gives the same answer
     as the known working version */
  errs = check_result(C, control_matrix, a_dim1, b_dim2);

  DEBUGGING( {
    if (errs > 0) {
      printf("reference matmul produces:\n");
      write_out(control_matrix, a_dim1, b_dim2);
      printf("\nteam_matmul gives:\n");
      write_out(C, a_dim1, b_dim2);
    }
  } )

  /* free all matrices */
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(control_matrix);

  return 0;
}

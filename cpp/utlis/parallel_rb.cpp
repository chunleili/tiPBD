#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include <sys/time.h>
#include <omp.h>

#define Tolerance 0.000012
#define TRUE 1
#define FALSE 0

#define N 8001
#define THREAD_COUNT 16

double ** A, **B, diff;

void display(double **V, int n)
{
  for (int i = 0; i < n+1; ++i)
  {
    for (int k = 0; k < n+1; ++k)
    {
        printf("%f ", V[i][k]);
    }
    printf("\n");
  }
  printf("\n");
}

void initialize (double **A, int n)
{
   int i,j;

   for (j=0;j<n+1;j++)
   {
     A[0][j]=1.0;
   }
   for (i=1;i<n+1;i++)
   {
      A[i][0]=1.0;
      for (j=1;j<n+1;j++) A[i][j]=0.0;
   }
}

void solve(double **B, int n)
{
   printf("\n\n-----------------------Serial Solver-----------------------\n\n\n");
   int convergence=FALSE;
   double diff, tmp;
   int i, j, iters=0;
   int for_iters;

   for(for_iters = 1; for_iters < 21; for_iters++)
   { 
     diff = 0.0;

     for (i=1;i<n;i++)
     {
       for (j=1;j<n;j++)
       {
         tmp = B[i][j];
         B[i][j] = 0.2*(B[i][j] + B[i][j-1] + B[i-1][j] + B[i][j+1] + B[i+1][j]);
         diff += fabs(B[i][j] - tmp);
       }
     }
     iters++;
     printf("Difference after %3d iterations: %f\n", iters, diff);
     if (diff/((double)N*(double)N) < Tolerance)
     {
       printf("\nConvergence achieved after %d iterations....Now exiting\n\n", iters);
       return;
     }
   }
   printf("\n\nIteration LIMIT Reached...Exiting\n\n");
}

// long usecs (void)
// {
//   struct timeval t;
//   gettimeofday(&t,NULL);
//   return t.tv_sec*1000000+t.tv_usec;
// }


/// @brief Usage: Timer t("timer_name");
///               t.start();
///               //do something
///               t.end();
/// You need to include <chrono> and <string> for this to work

#include <chrono>
#include <string>
#include <iostream>
class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
    std::chrono::duration<double, std::milli> elapsed;
    std::string name = "";

    Timer(std::string name = "") : name(name){};
    inline void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    inline void end()
    {
        m_end = std::chrono::steady_clock::now();
        elapsed = (m_end - m_start);
    }
    inline void report()
    {
        printf("(%s): %.0f(ms)", name.c_str(), elapsed.count());
    };
    
};


void solve_parallel(double **A, int n)
{ 
  printf("\n\n-----------------------Parallel Red Black Solver-----------------------\n\n\n");
  int for_iters;
  int iters = 0;
  int convergence=FALSE;
  double tmp;
  double diff;
  int i, j;
  for (for_iters = 1; for_iters < 21; ++for_iters)
  { 
    diff = 0;
    
    #pragma omp parallel num_threads(THREAD_COUNT) private(tmp, i, j) reduction(+:diff)
    {
      #pragma omp for
      for (i = 1; i <= n; ++i)
      {    
        for (j = 1; j <= n; ++j)
        { 
          if ((i + j) % 2 == 1)
          {
            // printf("Thread: %d on (%d, %d)\n", omp_get_thread_num(), i, j);
            tmp = A[i][j];
            A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
            diff += fabs(A[i][j] - tmp);
          }
        }
      }
      #pragma omp barrier
    }

    #pragma omp parallel num_threads(THREAD_COUNT) private(tmp, i, j) reduction(+:diff)
    {
      #pragma omp for
      for (i = 1; i <= n; ++i)
      {    
        for (j = 1; j <= n; ++j)
        { 
          if ((i + j) % 2 == 0)
          {
            // printf("Thread: %d on (%d, %d)\n", omp_get_thread_num(), i, j);
            tmp = A[i][j];
            A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
            diff += fabs(A[i][j] - tmp);
          }
        }
      }
      #pragma omp barrier
    }
   iters++;
   printf("Difference after %3d iterations: %f\n", iters, diff);
   if (diff/((double)N*(double)N) < Tolerance)
    {
      printf("\nConvergence achieved after %d iterations....Now exiting\n\n", iters);
      return;
    }
  }
  printf("\n\nIteration LIMIT Reached...Exiting\n\n");
}

int main(int argc, char * argv[])
{
   int i;
   long t_start,t_end;
   double time;
   A = new double *[N+2];
   B = new double *[N+2];
   for (i=0; i<N+2; i++) 
   {
     A[i] = new double[N+2];
     B[i] = new double[N+2];
   }

   initialize(B, N);

  //  t_start=usecs();
  Timer t("Serial");
  t.start();
   solve(B, N);
  t.end();
  t.report();
  //  t_end=usecs();

  //  time = ((double)(t_end-t_start))/1000000;
  //  printf("Computation time for Serial approach(secs): %f\n\n", time);
   
   // display(B, N);

   initialize(A, N);

  //  t_start=usecs();
  Timer t2("Parallel");
  t2.start();
   solve_parallel(A, N - 1);
  t2.end();
  t2.report();
  //  t_end=usecs();

  //  time = ((double)(t_end-t_start))/1000000;
  //  printf("Computation time for Parallel approach(secs): %f\n\n", time);

   // display(A, N);
}
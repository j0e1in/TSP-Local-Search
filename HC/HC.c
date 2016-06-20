#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "../common/city.h"

#define ITER_TIMES 20


int* twoOptSwap(int *route, int dim, int m, int n)
{
  /** Neighbor is defined as two of cities swapped
   *  (2-Opt swap)
   **/
  int i, j;
  int *route_new;

  route_new = (int*) malloc(sizeof(int)*dim);

  // Add route[0] to route[m-1] to route_new in order
  for (i = 0; i <= m-1; ++i)
  {
    route_new[i] = route[i];
  }

  // Add route[m] to route[n] to route_new in reverse order
  for (i = m, j = n; i <= n; ++i, --j)
  {
    route_new[i] = route[j];
  }

  // Add route[n+1] to route[dim-1] to route_new in order
  for (i = n+1; i < dim; ++i)
  {
    route_new[i] = route[i];
  }

  return route_new;
}

int* bestChild(int *route, float **dist, int dim)
{
  int i, j;
  int *route_new;
  int *route_best;
  float cur_dist;
  float best_dist = (float)INF;

  route_best = twoOptSwap(route, dim, 0, 1);
  best_dist = getDist(route_best, dist, dim);

  for (i = 0; i < dim; ++i)
  {
    for (j = 0; j < dim; ++j)
    {
      if (i >= j) continue;

      route_new = twoOptSwap(route, dim, i, j);
      cur_dist = getDist(route_new, dist, dim);

      if (cur_dist < best_dist)
      {
        best_dist = cur_dist;
        free(route_best);
        route_best = route_new;
      }
      else // Abandon the new route
      {
        free(route_new);
      }
    }
  }

  return route_best;
}

float HillClimbing(float **dist, int dim)
{
  int *route;
  int *route_child_best;
  int improved = 1;
  float best_so_far = (float)INF;
  float child_best_dist = (float)INF;

  // CREATE A RANDOM ROUTE AT START
  route = randRoute(dim);
  best_so_far = getDist(route, dist, dim);

  // ITERATE UNTIL HIT THE LOCAL OPTIMA
  while(improved)
  {
    improved = 0;
    route_child_best = bestChild(route, dist, dim);
    child_best_dist = getDist(route_child_best, dist, dim);
    printf("child_best_dist: %f\n", child_best_dist);

    if (child_best_dist < best_so_far)
    {
      best_so_far = child_best_dist;
      free(route);
      route = route_child_best;
      improved = 1;
    }
    else
    {
      free(route_child_best);
    }
  }

  return best_so_far;
}


int main(int argc, char const *argv[])
{
  int i;
  int dim;
  int **city;
  float opt_value;
  float prec_err;
  float **dist;
  float best_dist;
  char header[LINE_LEN];
  char *tmp;

  clock_t start;
  clock_t end;

  /** FOR PERFORMANCE MEASURING **/
  int succ_times;      // = times of finding optima / trials
  float run_time;
  float total_best_dist;
  /** FOR PERFORMANCE MEASURING **/

  FILE *f, *fw;
  if (argc == 2)
  {
    f = fopen(argv[1], "r");
  }
  else
  {
    printf("ERROR: missing input argument\n");
  }

  fw = fopen("result.txt", "w");

  // READ HEADER
  for (i = 0; i < 5; ++i)
  {
    fgets(header, LINE_LEN, f);
    tmp = strtok(header, ":");
    trimWS(tmp);
    fprintf(fw, "%s", tmp);

    // get number of cities
    if (strcmp(header, "DIMENSION") == 0)
    {
      tmp = strtok(NULL, "\n");
      trimWS(tmp);
      dim = atoi(tmp);
      fprintf(fw, ": -%d-\n", dim);
    }else{
      tmp = strtok(NULL, "\n");
      fprintf(fw, ": %s\n", tmp);
    }
  }

  // READ NODES
  if (dim == 442) // 442 NODES (Exponential format)
  {
    city = readExp(f, dim);
  }
  else // (Integer format)
  {
    city = readNorm(f, dim);
  }

  // CREATE A MATRIX WITH CITY DISTANCE
  dist = getDistMatrix(city, dim);

  // GET OPTIMAL VALUE
  switch(dim)
  {
    case 51:  opt_value = (float)OPTIMA_51;
              break;
    case 105: opt_value = (float)OPTIMA_105;
              break;
    case 442: opt_value = (float)OPTIMA_442;
              break;
  }
  prec_err = opt_value*0.01;
  total_best_dist = 0;
  run_time = 0;
  succ_times = 0;
  // ITER SEVERAL TIMES TO MEASURE PERFORMANCE
  for (i = 0; i < ITER_TIMES; ++i)
  {
    start = clock();
    best_dist = HillClimbing(dist, dim);
    end = clock();
    run_time += (float)(end-start);
    total_best_dist += best_dist;
    printf(".");
    // fprintf(fw, "Shortest distance: %f\n", best_dist);

    if (best_dist <= opt_value+prec_err && best_dist >= opt_value-prec_err)
    {
      succ_times++;
    }
  }
  printf("\n");
  fprintf(fw, "Search Algorithm: Hill Climbing\n");
  fprintf(fw, "Trials: %d\n", ITER_TIMES);
  fprintf(fw, "Average Best Distance: %.2f\n", ((float)total_best_dist/(float)ITER_TIMES));
  fprintf(fw, "Average Run Time: %.2f\n", (float)(run_time/ITER_TIMES)/CLOCKS_PER_SEC);
  fprintf(fw, "Success Rate: %f\n", (float)((float)succ_times/(float)ITER_TIMES*(float)100));


  free(dist);
  free(city);

  return 0;
}
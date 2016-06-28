#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "../common/city.h"


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
  int trials = 1;
  int i, dim, **city;
  float opt_value, prec_err, best_dist, **dist;

  clock_t start;
  clock_t end;

  int succ_times; // = times of finding optima / trials
  float run_time;
  float total_best_dist;

  FILE *f, *fw;
  if (argc == 2)
  {
    f = fopen(argv[1], "r");
  }
  else if (argc == 3)
  {
    f = fopen(argv[1], "r");
    trials = atoi(argv[2]);
  }
  else
  {
    printf("Usage: hc_cpu [data file] [trials]\n");
    return -1;
  }

  fw = fopen("result_cpu.txt", "w");

  dim = readHeader(f, fw);

  // read nodes
  if (dim == 442)
    city = readExp(f, dim);
  else
    city = readNorm(f, dim);

  dist = genDistMatrix(city, dim);

  // get optimal value
  opt_value = getOptValue(dim);

  prec_err = opt_value*0.01;
  total_best_dist = 0;
  run_time = 0;
  succ_times = 0;

  // run serveral times to get average results
  for (i = 0; i < trials; ++i)
  {
    start = clock();
    best_dist = HillClimbing(dist, dim);
    end = clock();

    run_time += (float)(end-start);
    total_best_dist += best_dist;
    printf("Shortest distance: %f\n", best_dist);

    if (best_dist <= opt_value+prec_err && best_dist >= opt_value-prec_err)
    {
      succ_times++;
    }
  }

  printf("\n");
  fprintf(fw, "Search Algorithm: Hill Climbing\n");
  fprintf(fw, "Trials: %d\n", trials);
  fprintf(fw, "Average Best Distance: %.2f\n", ((float)total_best_dist/(float)trials));
  fprintf(fw, "Average Run Time: %.2f\n", (float)(run_time/trials)/CLOCKS_PER_SEC);
  fprintf(fw, "Success Rate: %f\n", (float)((float)succ_times/(float)trials*(float)100));

  free(dist);
  free(city);

  return 0;
}
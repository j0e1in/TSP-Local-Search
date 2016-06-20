#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../common/city.h"

#define MAX_TEMP 99
#define E 2.71828183

float schedule(float temp, int iter)
{
  if (iter%10 == 0)
  {
    temp = temp*0.95;
  }
  return temp;
}

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

int* randChild(int *route, float **dist, int dim)
{
  int i, j;
  int *route_new;

  i = j = 0;
  // Choose a random child
  while(i == j)
  {
    j = rand()%dim;
    if (j == 0) continue;
    i = rand()%j;
  }
  route_new = twoOptSwap(route, dim, i, j);

  return route_new;
}

float SimulatedAnnealing(float **dist, int dim)
{
  int iter;
  int no_improve;
  int *route, *route_new;
  float prob, rnd;
  float temp = MAX_TEMP;
  float delta_E;
  float best_so_far = (float)INF;
  float cur_dist = (float)INF;
  float next_dist = (float)INF;

  srand(time(NULL));

  route = randRoute(dim);
  cur_dist = best_so_far = getDist(route, dist, dim);

  no_improve = 0;
  iter = 0;
  while(no_improve < 600)
  {
    temp = schedule(temp, iter);
    iter++;
    route_new = randChild(route, dist, dim);
    next_dist = getDist(route_new, dist, dim);
    delta_E = cur_dist - next_dist;

    if (delta_E > 0)
    {
      cur_dist = next_dist;
      free(route);
      route = route_new;
      no_improve = 0;
      // printf("delt-- %f\n", delta_E);
    }
    else
    {
      prob = (float)pow(E, (float)delta_E/temp)*100;
      if (delta_E != 0)
      {
        // printf("delt-- %f \ttemp-- %f \tprob-- %f\n", delta_E, temp, prob);
      }
      rnd = rand() % 100;
      if (rnd <= prob)
      {
        cur_dist = next_dist;
        free(route);
        route = route_new;
      }
      else // Abandon the new route
      {
        free(route_new);
      }
      no_improve++;
    }

    if (cur_dist < best_so_far)
    {
      best_so_far = cur_dist;
      no_improve = 0;
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
  else
  {
    printf("Usage: main_cpu [alg] [data file]\n");
    return -1;
  }

  fw = fopen("result.txt", "w");

  dim = readHeader(f, fw);

  // read nodes
  if (dim == 442)
    city = readExp(f, dim);
  else
    city = readNorm(f, dim);

  dist = getDistMatrix(city, dim);

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
    best_dist = SimulatedAnnealing(dist, dim);
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
  fprintf(fw, "Search Algorithm: Simulated Annealing\n");
  fprintf(fw, "Trials: %d\n", trials);
  fprintf(fw, "Average Best Distance: %.2f\n", ((float)total_best_dist/(float)trials));
  fprintf(fw, "Average Run Time: %.2f\n", (float)(run_time/trials)/CLOCKS_PER_SEC);
  fprintf(fw, "Success Rate: %f\n", (float)((float)succ_times/(float)trials*(float)100));

  free(dist);
  free(city);

  return 0;
  return 0;
}
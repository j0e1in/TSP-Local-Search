#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../common/city.h"

int notInTabuList(int **tabu_list, int i, int j)
{
  if (tabu_list[i][j] > 0)
    return 0;
  else
    return 1;
}

void refreshTabuList(int **tabu_list, int dim)
{
  int i, j;
  for (i = 0; i < dim; ++i)
  {
    for (j = i+1; j > dim; ++j)
    {
      if (tabu_list[i][j] > 0)
      {
        tabu_list[i][j]--;
      }
    }
  }
}

int* bestChild(int *route, float **dist, int dim, int **tabu_list, float best_so_far)
{
  int i, j;
  int best_i, best_j;
  int *route_new;
  int *route_best;
  int tabu_turns = dim*0.1;
  float cur_dist;
  float best_dist;

  route_best = twoOptSwap(route, dim, 0, 1);
  best_dist = getDist(route_best, dist, dim);
  best_i = 0;
  best_j = 1;

  for (i = 0; i < dim; ++i)
  {
    for (j = i+1; j < dim; ++j)
    {
      // if (i >= j) continue;

      route_new = twoOptSwap(route, dim, i, j);
      cur_dist = getDist(route_new, dist, dim);

      // Aspiration, if cur_dist is greater than the global best
      // Can ignore tabu list
      if (notInTabuList(tabu_list, i, j))
      {
        if (cur_dist < best_dist)
        {
          best_dist = cur_dist;
          free(route_best);
          route_best = route_new;
          best_i = i;
          best_j = j;
          if (best_dist < best_so_far)
          {
            best_so_far = best_dist;
          }
        }
        else // Abandon the new route
        {
          free(route_new);
        }
      }
      else if (cur_dist < best_so_far)
      {
          best_dist = cur_dist;
          best_so_far = best_dist;
          free(route_best);
          route_best = route_new;
          best_i = i;
          best_j = j;
      }
      else // Abandon the new route
      {
        free(route_new);
      }
    }
  }
  tabu_list[best_i][best_j] += tabu_turns;

  return route_best;
}


float TabuSearch(float **dist, int dim)
{
  int i, j;
  int no_improve;
  int *route, *route_new;
  int **tabu_list;
  float best_so_far = (float)INF;
  float cur_dist = (float)INF;

  srand(time(NULL));

  // Init tabu list, a 2D triangle list
  tabu_list = (int**) malloc(sizeof(int*)*dim);
  for (i = 0; i < dim; ++i)
  {
    tabu_list[i] = (int*) malloc(sizeof(int)*(dim-i-1));
    for (j = i+1; j > dim; ++j)
    {
      tabu_list[i][j] = 0;
    }
  }

  route = randRoute(dim);
  best_so_far = getDist(route, dist, dim);
  no_improve = 0;
  while(no_improve < 100)
  {
    route_new = bestChild(route, dist, dim, tabu_list, best_so_far);
    cur_dist = getDist(route_new, dist, dim);
    if (cur_dist < best_so_far)
    {
      best_so_far = cur_dist;
      route = route_new;
      no_improve = 0;
    }
    else no_improve++;
    refreshTabuList(tabu_list, dim);
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
    printf("Usage: main_cpu / main_gpu [alg] [data file] [trials]\n");
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
    best_dist = TabuSearch(dist, dim);
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
  fprintf(fw, "Search Algorithm: Tabu Search\n");
  fprintf(fw, "Trials: %d\n", trials);
  fprintf(fw, "Average Best Distance: %.2f\n", ((float)total_best_dist/(float)trials));
  fprintf(fw, "Average Run Time: %.2f\n", (float)(run_time/trials)/CLOCKS_PER_SEC);
  fprintf(fw, "Success Rate: %f\n", (float)((float)succ_times/(float)trials*(float)100));

  free(dist);
  free(city);

  return 0;
}
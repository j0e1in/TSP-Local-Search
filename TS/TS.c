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

int notInTabuList(int **tabu_list, int i, int j)
{
  if (i >= j)
  {
    printf("ERROR: i >= j exception!\n");
    exit(1);
  }
  if (tabu_list[i][j] > 0)
    return 0;

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
    for (j = 0; j < dim; ++j)
    {
      if (i >= j) continue;

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
    best_dist = TabuSearch(dist, dim);
    end = clock();
    run_time += (float)(end-start);
    total_best_dist += best_dist;
    // fprintf(fw, "Shortest distance: %f\n", best_dist);
    printf(".");
    fflush(stdout);

    if (best_dist <= opt_value+prec_err && best_dist >= opt_value-prec_err)
    {
      succ_times++;
    }
    // else if (best_dist < opt_value-prec_err)
    // {
    //   fprintf(fw, "best distance < OPTMA !!!\n");
    //   exit(1);
    // }
  }
  printf("\n");
  fprintf(fw, "Search Algorithm: Tabu Search\n");
  fprintf(fw, "Average Best Distance: %f\n", (float)(total_best_dist/ITER_TIMES));
  fprintf(fw, "Average Run Time: %f\n", (float)(run_time/ITER_TIMES)/CLOCKS_PER_SEC);
  fprintf(fw, "Success Rate: %f\n", (float)((float)succ_times/(float)ITER_TIMES*(float)100));


  free(dist);
  free(city);

  return 0;
}

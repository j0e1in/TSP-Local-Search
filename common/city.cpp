#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #include <ctype.h>
#include <random>
#include <chrono>
#include "city.h"

// READ NORMAL FORMAT OF CITIES (integer, 51 & 105 cities)
int** readNorm(FILE *f, int dim)
{
  int i;
  int **city;
  char string[LEN_MAX];

  city = (int**) malloc(sizeof(int*)*dim);
  for (i = 0; i < dim; ++i)
  {
    city[i] = (int*) malloc(sizeof(int)*2);
  }

  fgets(string, LEN_MAX, f); // to skip "NODE_COORD_SECTION"
  for (i = 0; i < dim; ++i)
  {
    fgets(string, LEN_MAX, f);
    strtok(string, " "); // to remove first number

    city[i][0] = atoi(strtok(NULL, " \n"));
    city[i][1] = atoi(strtok(NULL, " \n"));

  }

  return city;
}

// READ EXPONENTIAL FORMAT OF CITIES (442 cities)
int** readExp(FILE *f, int dim)
{
  int i;
  int **city;
  char string[LEN_MAX];
  char tmp[2][50];

  city = (int**) malloc(sizeof(int*)*dim);
  for (i = 0; i < dim; ++i)
  {
    city[i] = (int*) malloc(sizeof(int)*2);
  }

  fgets(string, LEN_MAX, f);     // to skip "NODE_COORD_SECTION"
  for (i = 0; i < dim; ++i)
  {
    fgets(string, LEN_MAX, f);
    strtok(string, " ");          // to remove first number

    strcpy(tmp[0], strtok(NULL, " \n"));
    strcpy(tmp[1], strtok(NULL, " \n"));

    city[i][0] = convertExptoInt(tmp[0]);
    city[i][1] = convertExptoInt(tmp[1]);
  }
  return city;
}

float** genDistMatrix(int **city, int dim)
{
  int i, j;
  float **dist;

  dist = (float**) malloc(sizeof(float*)*dim);
  for (i = 0; i < dim; ++i)
  {
    dist[i] = (float*) malloc(sizeof(float)*dim);
  }

  for (i = 0; i < dim; ++i)
  {
    for (j = 0; j < dim; ++j)
    {
      dist[i][j] = (float)sqrt((double)pow(city[i][0] - city[j][0], 2)
                              +(double)pow(city[i][1] - city[j][1], 2));
    }
  }
  return dist;
}

float getDist(int *route, float **dist, int dim)
{
  int i;
  float total_dist = 0;

  for (i = 0; i < dim-1; ++i)
  {
    total_dist += dist[route[i]][route[i+1]];
  }
  total_dist += dist[route[i]][route[0]];

  return total_dist;
}

int* randRoute(int dim)
{
  int i;
  int *route;

  route = (int*) malloc(sizeof(int)*dim);
  for (i = 0; i < dim; ++i)
  {
    route[i] = i;
  }

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto gen = std::default_random_engine(seed);
  std::shuffle(&route[0], &route[dim-1], gen);

  return route;
}

int readHeader(FILE* inputFile, FILE* resultFile)
{
  int i, dim = 0;
  char header[LEN_MAX];
  char *tmp;
  for (i = 0; i < 5; ++i)
  {
    fgets(header, LEN_MAX, inputFile);
    tmp = strtok(header, ":");
    trim(tmp);
    fprintf(resultFile, "%s", tmp);

    // get number of cities
    if (strcmp(header, "DIMENSION") == 0)
    {
      tmp = strtok(NULL, "\n");
      trim(tmp);
      dim = atoi(tmp);
      fprintf(resultFile, ": -%d-\n", dim);
    }else{
      tmp = strtok(NULL, "\n");
      fprintf(resultFile, ": %s\n", tmp);
    }
  }
  return dim;
}

float getOptValue(int dim)
{
  switch(dim)
  {
    case 51:  return (float)OPTIMA_51;
    case 105: return (float)OPTIMA_105;
    case 442: return (float)OPTIMA_442;
    default: return 0;
  }
}

char* trim(char *str)
{
  char *end;

  // Trim leading space
  while(isspace(*str)) str++;

  if(*str == 0)  // All spaces?
    return str;

  // Trim trailing space
  end = str + strlen(str) - 1;
  while(end > str && isspace(*end)) end--;

  // Write new null terminator
  *(end+1) = 0;

  return str;
}

int convertExptoInt(char *str)
{
  int num;
  int tmp_exp;
  float tmp_float;
  char *tmp_str;

  trim(str);                            // to clean string
  tmp_str = strtok(str, "e");             // to get float part
  tmp_float = (float)atof(tmp_str);       // convert to float
  tmp_str = strtok(NULL, "+");            // to get exp part
  tmp_exp = atoi(tmp_str);                // convert to int

  tmp_float = tmp_float * (float)pow(10, tmp_exp);
  num = (int)tmp_float;

  return num;
}




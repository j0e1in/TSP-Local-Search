#ifndef CITY_CUDA_H
#define CITY_CUDA_H

#include <cstdio>

#define LEN_MAX 50
#define INF 9999999
#define NUM_ELEMS(x)  (sizeof(x) / sizeof(x[0]))
#define OPTIMA_51   426
#define OPTIMA_105  14379
#define OPTIMA_442  50778

typedef struct child_
{
    int *route;
    float dist;
} child_t;

typedef struct simpleChild_
{
    int i; // twoOptSwap index
    int j; // twoOptSwap index
    float dist;
} simpleChild_t;

int** readNorm(FILE *f, int dim);
int** readExp(FILE *f, int dim);
int readHeader(FILE* inputFile, FILE* resultFile);
float* genDistMatrix_cuda(int **city, int dim);
float getOptValue(int dim);
float getDist(int *seq_city, float *dist, int dim);
int* randRoute(int dim);
int* twoOptSwap(int *route, int dim, int m, int n);

// utils
char* trim(char *str);
int convertExptoInt(char *str);

#endif
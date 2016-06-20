#ifndef CITY_H
#define CITY_H

#define LINE_LEN 100
#define INF 9999999
#define NUM_ELEMS(x)  (sizeof(x) / sizeof(x[0]))
#define OPTIMA_51   426
#define OPTIMA_105  14379
#define OPTIMA_442  50778

int** readNorm(FILE *f, int dim);
int** readExp(FILE *f, int dim);
float** getDistMatrix(int **city, int dim);
float getDist(int *seq_city, float **dist, int dim);
int* randRoute(int dim);
void shuffle(int *arry, int n);


/** ADDTIONAL FUNCTIONS **/
char* trimWS(char *str);
int convertExptoInt(char *str);

#endif
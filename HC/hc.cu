#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "../common/city_cuda.h"
#include "../common/helper_cuda.h"

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



int* twoOptSwap(int *route, int dim, int m, int n)
{
    /** Neighbor is defined as two of cities swapped
    *  (2-Opt swap)
    **/
    int i, j;
    int *route_new;

    route_new = new int[dim];

    // Add route[0] to route[m-1] to route_new in order
    for (i = 0; i < m; ++i)
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

__device__ float twoOptSwap_dist(int *route, float *distArr, int dim, int m, int n)
{
    float newRouteDist = 0;

    /*
    - 0 < m < n < dim
    - connect (m-1) => (n) , (m) => (n+1) , (dim-1) => (0)
    - the rest remain the same
    */

    for (int i = 0; i < dim-1; ++i)
    {
        if (i < m-1 || i > n || (i >= m && i < n))
            newRouteDist += distArr[route[i] * dim + route[i+1]];
    }
    newRouteDist += distArr[route[m-1] * dim + route[n]];
    newRouteDist += distArr[route[m] * dim + route[n+1]];
    newRouteDist += distArr[route[dim-1] * dim + route[0]];
    return newRouteDist;
}

__global__ void searchChild(int *route, float *distArr, int dim, simpleChild_t *d_children)
{
    int j = blockIdx.x;
    int i = threadIdx.x;
    int idx = i * blockDim.x + j;
    d_children[idx].i = i;
    d_children[idx].j = j;
    if (i < j) {
        d_children[idx].dist = twoOptSwap_dist(route, distArr, dim, i, j);
    } else {
        d_children[idx].dist = 0;
    }
}

child_t* bestChild(child_t *parent, float *distArr, int dim)
{
    child_t *bestChild;
    bestChild = new child_t();

    size_t routeSize = sizeof(int)*dim;
    size_t distArrSize = sizeof(float)*dim*dim;
    size_t childrenSize = sizeof(simpleChild_t)*dim*dim;

    int *d_route;
    float *d_distArr;

    simpleChild_t *d_children, *h_children;
    h_children = new simpleChild_t[dim*dim];

    checkCudaErrors(cudaMalloc((void**)&d_route, routeSize));
    checkCudaErrors(cudaMalloc((void**)&d_distArr, distArrSize));
    checkCudaErrors(cudaMalloc((void**)&d_children, childrenSize));

    checkCudaErrors(cudaMemcpy(d_route, parent->route, routeSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_distArr, distArr, distArrSize, cudaMemcpyHostToDevice));

    searchChild<<<dim, dim>>>(d_route, d_distArr, dim, d_children);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_children, d_children, childrenSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_route));
    checkCudaErrors(cudaFree(d_distArr));
    checkCudaErrors(cudaFree(d_children));

    simpleChild_t tmpChild = h_children[1];
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            // if (i < j)
            // {
            //     printf("%f\n", h_children[i*dim+j].dist);
            // }
            if (i < j && h_children[i*dim+j].dist < tmpChild.dist)
            {
                tmpChild = h_children[i*dim+j];
            }
        }
    }
    bestChild->route = twoOptSwap(parent->route, dim , tmpChild.i, tmpChild.j);
    bestChild->dist = getDist(bestChild->route, distArr, dim);

    free(h_children);

    return bestChild;
}

float HillClimbing(float *distArr, int dim)
{
    int improved = true;
    float best_dist;
    child_t *bestSoFar, *tmpChild;

    bestSoFar = new child_t();
    bestSoFar->route = randRoute(dim);
    bestSoFar->dist = getDist(bestSoFar->route, distArr, dim);

    // ITERATE UNTIL HIT THE LOCAL OPTIMA
    while(improved)
    {
        improved = false;
        tmpChild = bestChild(bestSoFar, distArr, dim);

        if (tmpChild->dist < bestSoFar->dist)
        {
            free(bestSoFar);
            bestSoFar = tmpChild;
            improved = true;
        }
        else
        {
            free(tmpChild);
        }
    }
    best_dist = bestSoFar->dist;
    free(bestSoFar);
    return best_dist;
}

int main(int argc, char const *argv[])
{
    int trials = 1;
    int i, dim, **city;
    float opt_value, prec_err, best_dist;
    float *distArr;

    clock_t start;
    clock_t end;

    int succ_times; // == times of finding optima / trials
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

    fw = fopen("result_gpu.txt", "w");
    dim = readHeader(f, fw);

    // read nodes
    if (dim == 442)
        city = readExp(f, dim);
    else
        city = readNorm(f, dim);

    distArr = genDistMatrix_cuda(city, dim);

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
        best_dist = HillClimbing(distArr, dim);
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

    free(distArr);
    cudaDeviceReset();

    return 0;
}
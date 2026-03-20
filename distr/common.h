#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <mpi.h>

void find_revcounts_displs(int my_elems, int total_ranks, int** recvcounts_out, int** displs_out, MPI_Comm comm) {
    int* recvcounts = (int*)malloc(total_ranks * sizeof(int));
    int* displs = (int*)malloc(total_ranks * sizeof(int));
    MPI_Gather(&my_elems, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);
    MPI_Bcast(recvcounts, total_ranks, MPI_INT, 0, comm);
    displs[0] = 0;
    for (int i = 1; i < total_ranks; i++)
        displs[i] = displs[i-1] + recvcounts[i-1];
    *recvcounts_out = recvcounts;
    *displs_out = displs;
}

#endif // COMMON_H
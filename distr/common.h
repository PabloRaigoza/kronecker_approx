#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <stdint.h>
#include <cblas.h>
#include <mpi.h>

// ------------------------------------------------------------------
// Seeded, partition-independent data generation.
//
// Every entry of Ã and of the starting vectors is a pure function of
// (seed, global index), so every partitioning scheme -- at any rank
// count -- sees exactly the same matrix and vectors.
// ------------------------------------------------------------------
enum SeedStream : uint64_t {
    STREAM_A_TILDE = 0,
    STREAM_V_INIT  = 1,  // initial v_send contents (kernel benchmarks)
    STREAM_U_INIT  = 2,  // initial u_send contents (kernel benchmarks)
    STREAM_V_START = 3,  // Golub-Kahan starting vector
    STREAM_VERIFY  = 4,  // kernel verification input
};

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Uniform double in [0, 1)
static inline double seeded_uniform(uint64_t seed, uint64_t stream, uint64_t idx) {
    uint64_t h = splitmix64(seed ^ splitmix64(stream ^ splitmix64(idx)));
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

// Ã[r][c], Ã of size (m1*n1) x (m2*n2)
static inline double a_tilde_entry(uint64_t seed, uint64_t r, uint64_t c, uint64_t num_cols) {
    return seeded_uniform(seed, STREAM_A_TILDE, r * num_cols + c);
}

// Fill a row-major local block of Ã given the global row/col index of
// each local row/col.
static inline void fill_a_local(double* A_local, uint64_t seed,
                                const long* row_gidx, int num_rows,
                                const long* col_gidx, int num_cols,
                                uint64_t global_num_cols) {
    for (size_t a = 0; a < (size_t)num_rows; a++)
        for (size_t b = 0; b < (size_t)num_cols; b++)
            A_local[a * num_cols + b] = a_tilde_entry(seed, row_gidx[a], col_gidx[b], global_num_cols);
}

static inline void fill_seeded_vec(double* x, const long* gidx, int n, uint64_t seed, uint64_t stream) {
    for (int i = 0; i < n; i++) x[i] = seeded_uniform(seed, stream, gidx[i]);
}

// ------------------------------------------------------------------
// Per-phase timers for one Ax / ATx kernel (accumulated across calls).
// ------------------------------------------------------------------
struct KernelTimers {
    double all_gather = 0.0;
    double computation = 0.0;
    double reduce_scatter = 0.0;
    double total() const { return all_gather + computation + reduce_scatter; }
};

// The slices of u (length m1*n1) and v (length m2*n2) this rank owns.
// Across the ranks of `comm`, every global element is owned by exactly
// one rank, so global reductions are a local reduction + MPI_Allreduce.
// u_gidx[i] / v_gidx[i] give the global index of u[i] / v[i].
struct OwnedVecs {
    double* u; int nu; const long* u_gidx;
    double* v; int nv; const long* v_gidx;
    MPI_Comm comm;
};

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

void find_revcounts_displs_ordered(int my_elems, int total_ranks, const int *order, int **recvcounts_out, int **displs_out, MPI_Comm comm) {
    int *recvcounts = (int*)malloc(total_ranks * sizeof(int));
    int *displs     = (int*)malloc(total_ranks * sizeof(int));
    int *tmpcounts  = (int*)malloc(total_ranks * sizeof(int));

    MPI_Gather(&my_elems, 1, MPI_INT, tmpcounts, 1, MPI_INT, 0, comm);
    MPI_Bcast(tmpcounts, total_ranks, MPI_INT, 0, comm);

    for (int i = 0; i < total_ranks; i++)
        recvcounts[i] = tmpcounts[order[i]];
    
    displs[0] = 0;
    for (int i = 1; i < total_ranks; i++)
        displs[i] = displs[i-1] + recvcounts[i-1];

    free(tmpcounts);

    *recvcounts_out = recvcounts;
    *displs_out = displs;
}

#endif // COMMON_H
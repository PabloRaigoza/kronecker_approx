#ifndef BCP_KERNEL_H
#define BCP_KERNEL_H
#include "../common.h"
#include "bcp_distr.h"
#include <cblas.h>
#include <cmath>
#include <assert.h>

void bcp_ax(BCPContext* ctx, int num_trails, bool per_rank_timings) {
    double total_all_gather_time = 0.0;
    double total_computation_time = 0.0;
    double total_reduce_scatter_time = 0.0;
    for (int trial = 0; trial < num_trails; trial++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double all_gather_start = MPI_Wtime();
        MPI_Allgatherv(ctx->v_send, ctx->v_send_size, MPI_DOUBLE,
               ctx->v_recv, ctx->recvcounts_v, ctx->displs_v, MPI_DOUBLE,
               ctx->v_comm);
        total_all_gather_time += (MPI_Wtime() - all_gather_start);

        MPI_Barrier(MPI_COMM_WORLD);
        double computation_start = MPI_Wtime();
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    ctx->u_recv_size, // rows of local_A
                    ctx->v_recv_size,     // cols of local_A
                    1.0,                   // alpha
                    ctx->A_local,          // local_A
                    ctx->v_recv_size,     // lda
                    ctx->v_recv,          // x
                    1,                     // incx
                    0.0,                   // beta
                    ctx->u_recv,          // y
                    1);                    // incy
        total_computation_time += (MPI_Wtime() - computation_start);

        double reduce_scatter_start = MPI_Wtime();
        MPI_Reduce_scatter(ctx->u_recv, ctx->u_send, ctx->recvcounts_u, MPI_DOUBLE, MPI_SUM, ctx->u_comm);
        total_reduce_scatter_time += (MPI_Wtime() - reduce_scatter_start);
    }

    double total_all_gather = 0.0, total_computation = 0.0, total_reduce_scatter = 0.0;
    MPI_Reduce(&total_all_gather_time, &total_all_gather, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_computation_time, &total_computation, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_reduce_scatter_time, &total_reduce_scatter, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        total_all_gather /= num_trails * ctx->world_size;
        total_computation /= num_trails * ctx->world_size;
        total_reduce_scatter /= num_trails * ctx->world_size;
        printf("Mean All Gather: %.6f | Mean Computation: %.6f | Mean Reduce Scatter: %.6f\n", total_all_gather, total_computation, total_reduce_scatter);
    }

    if (per_rank_timings) {
        double local_all_gather_time = total_all_gather_time / num_trails;
        double local_computation_time = total_computation_time / num_trails;
        double local_reduce_scatter_time = total_reduce_scatter_time / num_trails;
        printf("Rank %d: Local All Gather Time: %.9f | Local Computation Time: %.9f | Local Reduce Scatter Time: %.9f\n",
            ctx->world_rank, local_all_gather_time, local_computation_time, local_reduce_scatter_time);
    }
}

void bcp_atx(BCPContext* ctx, int num_trails, bool per_rank_timings) {
    double total_all_gather_time = 0.0;
    double total_computation_time = 0.0;
    double total_reduce_scatter_time = 0.0;
    for (int trial = 0; trial < num_trails; trial++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double all_gather_start = MPI_Wtime();
        MPI_Allgatherv(ctx->u_send, ctx->u_send_size, MPI_DOUBLE,
               ctx->u_recv, ctx->recvcounts_u, ctx->displs_u, MPI_DOUBLE,
               ctx->u_comm);
        total_all_gather_time += (MPI_Wtime() - all_gather_start);

        MPI_Barrier(MPI_COMM_WORLD);
        double computation_start = MPI_Wtime();
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    ctx->u_recv_size, // rows of local_A
                    ctx->v_recv_size,     // cols of local_A
                    1.0,                   // alpha
                    ctx->A_local,          // local_A
                    ctx->v_recv_size,     // lda
                    ctx->u_recv,          // x
                    1,                     // incx
                    0.0,                   // beta
                    ctx->v_recv,          // y
                    1);                    // incy
        total_computation_time += (MPI_Wtime() - computation_start);

        double reduce_scatter_start = MPI_Wtime();
        MPI_Reduce_scatter(ctx->v_recv, ctx->v_send, ctx->recvcounts_v, MPI_DOUBLE, MPI_SUM, ctx->v_comm);
        total_reduce_scatter_time += (MPI_Wtime() - reduce_scatter_start);
    }

    double total_all_gather = 0.0, total_computation = 0.0, total_reduce_scatter = 0.0;
    MPI_Reduce(&total_all_gather_time, &total_all_gather, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_computation_time, &total_computation, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_reduce_scatter_time, &total_reduce_scatter, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        total_all_gather /= num_trails * ctx->world_size;
        total_computation /= num_trails * ctx->world_size;
        total_reduce_scatter /= num_trails * ctx->world_size;
        printf("Mean All Gather: %.6f | Mean Computation: %.6f | Mean Reduce Scatter: %.6f\n", total_all_gather, total_computation, total_reduce_scatter);
    }

    if (per_rank_timings) {
        double local_all_gather_time = total_all_gather_time / num_trails;
        double local_computation_time = total_computation_time / num_trails;
        double local_reduce_scatter_time = total_reduce_scatter_time / num_trails;
        printf("Rank %d: Local All Gather Time: %.9f | Local Computation Time: %.9f | Local Reduce Scatter Time: %.9f\n",
            ctx->world_rank, local_all_gather_time, local_computation_time, local_reduce_scatter_time);
    }
}

void bcp_verify(BCPContext* ctx) {
    double *A_full = NULL, *v_full = NULL, *u_full = NULL, *u_serial = NULL, *v_serial = NULL;
    double *v_full2 = NULL;
    bool failed = false;
    // Intial Reconstructions
    bcp_reconstruct_A_tilde(&A_full, ctx);
    bcp_reconstruct_v(&v_full, ctx);

    // Run Ax
    bcp_ax(ctx, 1, false);
    bcp_reconstruct_u(&u_full, ctx);

    // // Run A^T x
    bcp_atx(ctx, 1, false);
    bcp_reconstruct_v(&v_full2, ctx);

    if (ctx->world_rank == 0) {
        u_serial = (double*)calloc(ctx->m1 * ctx->n1, sizeof(double));
        v_serial = (double*)calloc(ctx->m2 * ctx->n2, sizeof(double));
        
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    ctx->m1 * ctx->n1, // rows of A_full
                    ctx->m2 * ctx->n2, // cols of A_full
                    1.0,               // alpha
                    A_full,            // A_full
                    ctx->m2 * ctx->n2, // lda
                    v_full,            // x
                    1,                 // incx
                    0.0,               // beta
                    u_serial,          // y
                    1);                // incy         
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    ctx->m1 * ctx->n1, // rows of A_full
                    ctx->m2 * ctx->n2, // cols of A_full
                    1.0,               // alpha
                    A_full,            // A_full
                    ctx->m2 * ctx->n2, // lda
                    u_full,            // x
                    1,                 // incx
                    0.0,               // beta
                    v_serial,          // y
                    1);                // incy

        for (int i = 0; i < ctx->m1 * ctx->n1; ++i) {
            if (fabs(u_full[i] - u_serial[i]) > 1e-6) {
                // printf("Verification failed at index %d: expected %.6f, got %.6f\n", i, u_serial[i], u_full[i]);
                failed = true;
            }
        }
        printf("Verification for Ax: %s\n", failed ? "FAILED" : "PASSED");

        failed = false;
        for (int i = 0; i < ctx->m2 * ctx->n2; ++i) {
            if (fabs(v_full2[i] - v_serial[i]) > 1e-6) {
                // printf("Verification failed at index %d: expected %.6f, got %.6f\n", i, v_serial[i], v_full2[i]);
                failed = true;
            }
        }
        printf("Verification for A^T x: %s\n", failed ? "FAILED" : "PASSED");
            
        free(A_full);
        free(v_full);
        free(v_full2);
        free(u_full);
        free(u_serial);
        free(v_serial);
    }
}

#endif // BCP_KERNEL_H
#ifndef WBP_KERNEL_H
#define WBP_KERNEL_H
#include "../common.h"
#include "wbp_distr.h"
#include <cblas.h>
#include <cmath>
#include <assert.h>

void wbp_ax(WBPContext* ctx, int num_trails, bool per_rank_timings) {
    double total_all_gather_time = 0.0;
    double total_computation_time = 0.0;
    for (int trial = 0; trial < num_trails; trial++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double all_gather_start = MPI_Wtime();
        MPI_Allgatherv(ctx->v_send, ctx->v_send_size, MPI_DOUBLE,
               ctx->v_recv, ctx->recvcounts_v, ctx->displs_v, MPI_DOUBLE,
               MPI_COMM_WORLD);
        total_all_gather_time += (MPI_Wtime() - all_gather_start);

        MPI_Barrier(MPI_COMM_WORLD);
        
        double computation_start = MPI_Wtime();
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    ctx->num_local_blocks, // rows of local_A
                    ctx->m2 * ctx->n2,     // cols of local_A
                    1.0,                   // alpha
                    ctx->A_local,          // local_A
                    ctx->m2 * ctx->n2,     // lda
                    ctx->v_recv,          // x
                    1,                     // incx
                    0.0,                   // beta
                    ctx->u_send,          // y
                    1);                    // incy
        total_computation_time += (MPI_Wtime() - computation_start);
    }

    double total_all_gather = 0.0, total_computation = 0.0;
    MPI_Reduce(&total_all_gather_time, &total_all_gather, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_computation_time, &total_computation, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        total_all_gather /= num_trails * ctx->world_size;
        total_computation /= num_trails * ctx->world_size;
        printf("Mean All Gather: %.6f | Mean Computation: %.6f\n", total_all_gather, total_computation);
    }

    if (per_rank_timings) {
        double local_all_gather_time = total_all_gather_time / num_trails;
        double local_computation_time = total_computation_time / num_trails;
        printf("Rank %d: Local All Gather Time: %.9f | Local Computation Time: %.9f\n",
            ctx->world_rank, local_all_gather_time, local_computation_time);
    }
}

void wbp_atx(WBPContext* ctx, int num_trails, bool per_rank_timings) {
    double total_computation_time = 0.0;
    double total_reduce_scatter_time = 0.0;
    for (int trial = 0; trial < num_trails; trial++) {
        double computation_start = MPI_Wtime();
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    ctx->num_local_blocks, // rows of local_A
                    ctx->m2 * ctx->n2,     // cols of local_A
                    1.0,                   // alpha
                    ctx->A_local,          // local_A
                    ctx->m2 * ctx->n2,     // lda
                    ctx->u_send,          // x
                    1,                     // incx
                    0.0,                   // beta
                    ctx->v_recv,          // y
                    1);                    // incy
        total_computation_time += (MPI_Wtime() - computation_start);

        double reduce_scatter_start = MPI_Wtime();
        MPI_Reduce_scatter(ctx->v_recv, ctx->v_send, ctx->recvcounts_v, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        total_reduce_scatter_time += (MPI_Wtime() - reduce_scatter_start);
    }

    double total_computation = 0.0, total_reduce_scatter = 0.0;
    MPI_Reduce(&total_computation_time, &total_computation, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_reduce_scatter_time, &total_reduce_scatter, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        total_computation /= num_trails * ctx->world_size;
        total_reduce_scatter /= num_trails * ctx->world_size;
        printf("Mean Computation: %.6f | Mean Reduce Scatter: %.6f\n", total_computation, total_reduce_scatter);
    }

    if (per_rank_timings) {
        double local_computation_time = total_computation_time / num_trails;
        double local_reduce_scatter_time = total_reduce_scatter_time / num_trails;
        printf("Rank %d: Local Computation Time: %.9f | Local Reduce Scatter Time: %.9f\n",
            ctx->world_rank, local_computation_time, local_reduce_scatter_time);
    }
}

void wbp_verify(WBPContext* ctx) {
    double *A_full = NULL, *v_full = NULL, *u_full = NULL, *u_serial = NULL, *v_serial = NULL;
    double *v_full2 = NULL;
    bool failed = false;
    // Intial Reconstructions
    wbp_reconstruct_A_tilde(&A_full, ctx);
    wbp_reconstruct_v(&v_full, ctx);

    // Run Ax
    wbp_ax(ctx, 1, false);
    wbp_reconstruct_u(&u_full, ctx);

    // // Run A^T x
    wbp_atx(ctx, 1, false);
    wbp_reconstruct_v(&v_full2, ctx);

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
                printf("Verification failed at index %d: expected %.6f, got %.6f\n", i, u_serial[i], u_full[i]);
                failed = true;
            }
        }
        if (!failed) printf("Verification passed for Ax\n");

        failed = false;
        for (int i = 0; i < ctx->m2 * ctx->n2; ++i) {
            if (fabs(v_full2[i] - v_serial[i]) > 1e-6) {
                printf("Verification failed at index %d: expected %.6f, got %.6f\n", i, v_serial[i], v_full2[i]);
                failed = true;
            }
        }
        if (!failed) printf("Verification passed for A^T x\n");
            
        free(A_full);
        free(v_full);
        free(v_full2);
        free(u_full);
        free(u_serial);
        free(v_serial);
    }
}

#endif // WBP_KERNEL_H
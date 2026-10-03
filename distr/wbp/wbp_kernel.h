#ifndef WBP_KERNEL_H
#define WBP_KERNEL_H
#include "../common.h"
#include "wbp_distr.h"
#include <cblas.h>
#include <cmath>
#include <assert.h>

// ------------------------------------------------------------------
// Single-shot kernels used by both the benchmarks and Golub-Kahan.
//   do_ax : v_send (owned v slice) -> u_send (owned u slice)
//   do_atx: u_send (owned u slice) -> v_send (owned v slice)
// With sync=true every phase is preceded by a barrier so per-phase
// timings are not polluted by load imbalance from the previous phase.
// ------------------------------------------------------------------
void do_ax(WBPContext* ctx, KernelTimers* t, bool sync) {
    double t0 = phase_start(&t->barrier, sync);
    MPI_Allgatherv(ctx->v_send, ctx->v_send_size, MPI_DOUBLE,
           ctx->v_recv, ctx->recvcounts_v, ctx->displs_v, MPI_DOUBLE,
           MPI_COMM_WORLD);
    double e1 = MPI_Wtime();

    double t1 = phase_start(&t->barrier, sync);
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
    double t2 = MPI_Wtime();

    t->all_gather += e1 - t0;
    t->computation += t2 - t1;
}

void do_atx(WBPContext* ctx, KernelTimers* t, bool sync) {
    double t0 = phase_start(&t->barrier, sync);
    // BLAS quick-returns without writing y when this rank owns no rows
    if (ctx->num_local_blocks == 0) memset(ctx->v_recv, 0, (size_t)ctx->v_recv_size * sizeof(double));
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
    double e1 = MPI_Wtime();

    double t1 = phase_start(&t->barrier, sync);
    MPI_Reduce_scatter(ctx->v_recv, ctx->v_send, ctx->recvcounts_v, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    t->computation += e1 - t0;
    t->reduce_scatter += t2 - t1;
}

OwnedVecs owned_vecs(WBPContext* ctx) {
    return { ctx->u_send, ctx->u_send_size, ctx->u_gidx.data(),
             ctx->v_send, ctx->v_send_size, ctx->v_gidx.data(), MPI_COMM_WORLD };
}

void wbp_ax(WBPContext* ctx, int num_trails, bool per_rank_timings) {
    KernelTimers t;
    for (int trial = 0; trial < num_trails; trial++) do_ax(ctx, &t, true);

    double local_all_gather_time = t.all_gather / num_trails;
    double local_computation_time = t.computation / num_trails;

    double max_all_gather = 0.0, max_computation = 0.0;
    MPI_Reduce(&local_all_gather_time, &max_all_gather, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_computation_time, &max_computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        printf("Max All Gather: %.6f | Max Computation: %.6f\n", max_all_gather, max_computation);
    }

    if (per_rank_timings) {
        printf("Rank %d: Local All Gather Time: %.9f | Local Computation Time: %.9f\n",
            ctx->world_rank, local_all_gather_time, local_computation_time);
    }
}

void wbp_atx(WBPContext* ctx, int num_trails, bool per_rank_timings) {
    KernelTimers t;
    for (int trial = 0; trial < num_trails; trial++) do_atx(ctx, &t, true);

    double local_computation_time = t.computation / num_trails;
    double local_reduce_scatter_time = t.reduce_scatter / num_trails;

    double max_computation = 0.0, max_reduce_scatter = 0.0;
    MPI_Reduce(&local_computation_time, &max_computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_reduce_scatter_time, &max_reduce_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        printf("Max Computation: %.6f | Max Reduce Scatter: %.6f\n", max_computation, max_reduce_scatter);
    }

    if (per_rank_timings) {
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
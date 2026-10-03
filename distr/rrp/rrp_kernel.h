#ifndef RRP_KERNEL_H
#define RRP_KERNEL_H
#include "../common.h"
#include "../wbp/wbp_distr.h"
#include "../wbp/wbp_kernel.h"
#include "rrp_distr.h"
#include <cblas.h>
#include <cmath>
#include <assert.h>

// ------------------------------------------------------------------
// Single-shot kernels (see wbp_kernel.h for the contract).
// Falls back to WBP when world_size <= n1.
// ------------------------------------------------------------------
void do_ax(RRPContext* ctx, KernelTimers* t, bool sync) {
    if (ctx->simple_case) { do_ax(&ctx->wbp_ctx, t, sync); return; }

    double t0 = phase_start(&t->barrier, sync);
    if (!ctx->is_on_edge_natural) {
        MPI_Allgatherv(ctx->v_send_main, ctx->v_send_main_size, MPI_DOUBLE,
                    ctx->v_recv, ctx->recvcounts_v_main, ctx->displs_v_main, MPI_DOUBLE,
                    ctx->v_comm_main);
    }
    if (ctx->is_on_edge) {
        MPI_Allgatherv(ctx->v_send_edge, ctx->v_send_edge_size, MPI_DOUBLE,
                    ctx->v_recv + ctx->v_recv_main_size, ctx->recvcounts_v_edge, ctx->displs_v_edge, MPI_DOUBLE,
                    ctx->v_comm_edge);
    }
    double e1 = MPI_Wtime();

    double t1 = phase_start(&t->barrier, sync);
    // BLAS quick-returns without writing y when this rank owns no columns
    if (ctx->v_recv_main_size + ctx->v_recv_edge_size == 0) memset(ctx->u_recv, 0, (size_t)ctx->u_recv_size * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ctx->u_recv_size, // rows of local_A
                ctx->v_recv_main_size + ctx->v_recv_edge_size,     // cols of local_A
                1.0,                   // alpha
                ctx->A_local,          // local_A
                ctx->v_recv_main_size + ctx->v_recv_edge_size,     // lda
                ctx->v_recv,          // x
                1,                     // incx
                0.0,                   // beta
                ctx->u_recv,          // y
                1);                    // incy
    double e2 = MPI_Wtime();

    double t2 = phase_start(&t->barrier, sync);
    MPI_Reduce_scatter(ctx->u_recv, ctx->u_send, ctx->recvcounts_u, MPI_DOUBLE, MPI_SUM, ctx->u_comm);
    double t3 = MPI_Wtime();

    t->all_gather += e1 - t0;
    t->computation += e2 - t1;
    t->reduce_scatter += t3 - t2;
}

void do_atx(RRPContext* ctx, KernelTimers* t, bool sync) {
    if (ctx->simple_case) { do_atx(&ctx->wbp_ctx, t, sync); return; }

    double t0 = phase_start(&t->barrier, sync);
    MPI_Allgatherv(ctx->u_send, ctx->u_send_size, MPI_DOUBLE,
           ctx->u_recv, ctx->recvcounts_u, ctx->displs_u, MPI_DOUBLE,
           ctx->u_comm);
    double e1 = MPI_Wtime();

    double t1 = phase_start(&t->barrier, sync);
    cblas_dgemv(CblasRowMajor, CblasTrans,
                ctx->u_recv_size, // rows of local_A
                ctx->v_recv_main_size + ctx->v_recv_edge_size,     // cols of local_A
                1.0,                   // alpha
                ctx->A_local,          // local_A
                ctx->v_recv_main_size + ctx->v_recv_edge_size,     // lda
                ctx->u_recv,          // x
                1,                     // incx
                0.0,                   // beta
                ctx->v_recv,          // y
                1);                    // incy
    double e2 = MPI_Wtime();

    double t2 = phase_start(&t->barrier, sync);
    if (!ctx->is_on_edge_natural) {
        MPI_Reduce_scatter(ctx->v_recv, ctx->v_send_main, ctx->recvcounts_v_main, MPI_DOUBLE, MPI_SUM, ctx->v_comm_main);
    }
    if (ctx->is_on_edge) {
        MPI_Reduce_scatter(ctx->v_recv + ctx->v_recv_main_size, ctx->v_send_edge, ctx->recvcounts_v_edge, MPI_DOUBLE, MPI_SUM, ctx->v_comm_edge);
    }
    double t3 = MPI_Wtime();

    t->all_gather += e1 - t0;
    t->computation += e2 - t1;
    t->reduce_scatter += t3 - t2;
}

OwnedVecs owned_vecs(RRPContext* ctx) {
    if (ctx->simple_case) return owned_vecs(&ctx->wbp_ctx);
    return { ctx->u_send, ctx->u_send_size, ctx->u_gidx.data(),
             ctx->v_send, ctx->v_send_main_size + ctx->v_send_edge_size, ctx->v_gidx.data(), MPI_COMM_WORLD };
}

static void rrp_print_timings(RRPContext* ctx, const KernelTimers& t, int num_trails, bool per_rank_timings) {
    double local_all_gather_time = t.all_gather / num_trails;
    double local_computation_time = t.computation / num_trails;
    double local_reduce_scatter_time = t.reduce_scatter / num_trails;

    double max_all_gather = 0.0, max_computation = 0.0, max_reduce_scatter = 0.0;
    MPI_Reduce(&local_all_gather_time, &max_all_gather, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_computation_time, &max_computation, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_reduce_scatter_time, &max_reduce_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (ctx->world_rank == 0) {
        printf("Max All Gather: %.6f | Max Computation: %.6f | Max Reduce Scatter: %.6f\n", max_all_gather, max_computation, max_reduce_scatter);
    }

    if (per_rank_timings) {
        printf("Rank %d: Local All Gather Time: %.9f | Local Computation Time: %.9f | Local Reduce Scatter Time: %.9f\n",
            ctx->world_rank, local_all_gather_time, local_computation_time, local_reduce_scatter_time);
    }
}

void rrp_ax(RRPContext* ctx, int num_trails, bool per_rank_timings) {
    if (ctx->simple_case) {
        wbp_ax(&ctx->wbp_ctx, num_trails, per_rank_timings);
        return;
    }
    KernelTimers t;
    for (int trial = 0; trial < num_trails; trial++) do_ax(ctx, &t, true);
    rrp_print_timings(ctx, t, num_trails, per_rank_timings);
}

void rrp_atx(RRPContext* ctx, int num_trails, bool per_rank_timings) {
    if (ctx->simple_case) {
        wbp_atx(&ctx->wbp_ctx, num_trails, per_rank_timings);
        return;
    }
    KernelTimers t;
    for (int trial = 0; trial < num_trails; trial++) do_atx(ctx, &t, true);
    rrp_print_timings(ctx, t, num_trails, per_rank_timings);
}

void rrp_verify(RRPContext* ctx) {
    if (ctx->simple_case) {
        wbp_verify(&ctx->wbp_ctx);
        return;
    }

    double *A_full = NULL, *v_full = NULL, *u_full = NULL, *u_serial = NULL, *v_serial = NULL;
    double *v_full2 = NULL;
    bool failed = false;
    // Intial Reconstructions
    rrp_reconstruct_A_tilde(&A_full, ctx);
    rrp_reconstruct_v(&v_full, ctx);

    // Run Ax
    rrp_ax(ctx, 1, false);
    rrp_reconstruct_u(&u_full, ctx);

    // // Run A^T x
    rrp_atx(ctx, 1, false);
    rrp_reconstruct_v(&v_full2, ctx);

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

#endif // RRP_KERNEL_H
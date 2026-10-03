#ifndef WBP_DISTR_H
#define WBP_DISTR_H

#include "../common.h"
#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <vector>

struct WBPContext {
    int world_rank, world_size;
    int m1, n1, m2, n2;
    int num_local_blocks;
    double *A_local;
    int A_local_size;
    double *v_send, *u_send;
    double *v_recv, *u_recv;
    int v_send_size, u_send_size;
    int v_recv_size, u_recv_size;
    int *recvcounts_v, *displs_v;
    int *recvcounts_u, *displs_u;
    std::vector<long> u_gidx, v_gidx;  // global index of each u_send / v_send entry
};

WBPContext wbp_distribute(int world_rank, int world_size, int m1, int n1, int m2, int n2, uint64_t seed) {
    WBPContext ctx;
    ctx.world_rank = world_rank;
    ctx.world_size = world_size;
    ctx.m1 = m1;
    ctx.n1 = n1;
    ctx.m2 = m2;
    ctx.n2 = n2;

    int total_blocks = m1 * n1;
    int blocks_per_rank = total_blocks / world_size;
    int remainder = total_blocks % world_size;
    ctx.num_local_blocks = blocks_per_rank + (world_rank < remainder ? 1 : 0);

    int total_v_elems = m2 * n2;
    int base_v_elems = total_v_elems / world_size;
    int rem_v_elems = total_v_elems % world_size;
    ctx.v_send_size = base_v_elems + (world_rank < rem_v_elems ? 1 : 0);
    ctx.u_send_size = ctx.num_local_blocks;

    ctx.A_local_size = (size_t)ctx.num_local_blocks * (size_t)m2 * (size_t)n2;
    ctx.v_recv_size = m2 * n2;
    ctx.u_recv_size = ctx.num_local_blocks;

    ctx.A_local = (double*)malloc(ctx.A_local_size * sizeof(double));
    ctx.v_recv = (double*)malloc((size_t)ctx.v_recv_size * sizeof(double));
    ctx.u_recv = (double*)malloc((size_t)ctx.u_recv_size * sizeof(double));
    ctx.v_send = (double*)malloc((size_t)ctx.v_send_size * sizeof(double));
    ctx.u_send = (double*)malloc((size_t)ctx.u_send_size * sizeof(double));

    find_revcounts_displs(ctx.v_send_size, world_size, &ctx.recvcounts_v, &ctx.displs_v, MPI_COMM_WORLD);
    find_revcounts_displs(ctx.u_send_size, world_size, &ctx.recvcounts_u, &ctx.displs_u, MPI_COMM_WORLD);

    // This rank owns rows [row_offset, row_offset + num_local_blocks) of Ã and all columns
    long row_offset = (long)world_rank * blocks_per_rank + (world_rank < remainder ? world_rank : remainder);
    std::vector<long> row_gidx(ctx.u_recv_size), col_gidx(ctx.v_recv_size);
    for (int a = 0; a < ctx.u_recv_size; a++) row_gidx[a] = row_offset + a;
    for (int b = 0; b < ctx.v_recv_size; b++) col_gidx[b] = b;

    ctx.u_gidx = row_gidx;
    ctx.v_gidx.resize(ctx.v_send_size);
    for (int i = 0; i < ctx.v_send_size; i++) ctx.v_gidx[i] = ctx.displs_v[world_rank] + i;

    fill_a_local(ctx.A_local, seed, row_gidx.data(), ctx.u_recv_size, col_gidx.data(), ctx.v_recv_size, (uint64_t)m2 * n2);
    fill_seeded_vec(ctx.v_send, ctx.v_gidx.data(), ctx.v_send_size, seed, STREAM_V_INIT);
    fill_seeded_vec(ctx.u_send, ctx.u_gidx.data(), ctx.u_send_size, seed, STREAM_U_INIT);

    return ctx;
}

void wbp_reconstruct_A_tilde(double **tilde_A, WBPContext* ctx) {
    if (ctx->world_rank == 0) {
        *tilde_A = (double*)malloc((size_t)ctx->m1 * (size_t)ctx->n1 * (size_t)ctx->m2 * (size_t)ctx->n2 * sizeof(double));
    }

    int data_per_rank = ctx->num_local_blocks * ctx->m2 * ctx->n2;
    int *recvcounts_A, *displs_A;
    find_revcounts_displs(data_per_rank, ctx->world_size, &recvcounts_A, &displs_A, MPI_COMM_WORLD);
    MPI_Gatherv(ctx->A_local, data_per_rank, MPI_DOUBLE,
                *tilde_A, recvcounts_A, displs_A, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    free(recvcounts_A);
    free(displs_A);
}

void wbp_reconstruct_v(double **v_g, WBPContext* ctx) {
    if (ctx->world_rank == 0) {
        *v_g = (double*)malloc((size_t)ctx->m2 * (size_t)ctx->n2 * sizeof(double));
    }

    MPI_Gatherv(ctx->v_send, ctx->v_send_size, MPI_DOUBLE,
                *v_g, ctx->recvcounts_v, ctx->displs_v, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

void wbp_reconstruct_u(double **u_g, WBPContext* ctx) {
    if (ctx->world_rank == 0) {
        *u_g = (double*)malloc((size_t)ctx->m1 * (size_t)ctx->n1 * sizeof(double));
    }

    MPI_Gatherv(ctx->u_send, ctx->u_send_size, MPI_DOUBLE,
                *u_g, ctx->recvcounts_u, ctx->displs_u, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

void wbp_free_context(WBPContext* ctx) {
    free(ctx->A_local);
    free(ctx->v_send);
    free(ctx->u_send);
    free(ctx->v_recv);
    free(ctx->u_recv);
    free(ctx->recvcounts_v);
    free(ctx->displs_v);
    free(ctx->recvcounts_u);
    free(ctx->displs_u);
}

#endif // WBP_DISTR_H
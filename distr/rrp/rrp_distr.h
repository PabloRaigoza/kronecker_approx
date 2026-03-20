#ifndef RRP_DISTR_H
#define RRP_DISTR_H

#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <cassert>
#include "../common.h"
#include "../wbp/wbp_distr.h"

// If world_size <= n1 we essentially wrap WBP distribution
struct RRPContext {
    WBPContext wbp_ctx;
    int world_rank, world_size;
    int m1, n1, m2, n2;
    int cols_owned_main, cols_owned_edge;
    double *A_local;
    int A_local_size;
    
    double *v_send_main;
    int v_send_main_size;

    double *v_send_edge;
    int v_send_edge_size;

    double *v_recv;
    int v_recv_main_size;
    int v_recv_edge_size;

    double *u_recv, *u_send;
    int u_recv_size, u_send_size;
    
    int *recvcounts_v_main, *displs_v_main;
    int *recvcounts_v_edge, *displs_v_edge;
    int *recvcounts_u, *displs_u;
    std::vector<std::vector<int>> state;
    bool simple_case;
    bool is_on_edge, is_on_edge_natural;
    MPI_Comm v_comm_main, v_comm_edge;
    MPI_Comm u_comm;
    int v_main_rank, v_main_size;
    int v_edge_rank, v_edge_size;
    int u_rank, u_size;
};

void rrp_compute_state(RRPContext* ctx) {
    ctx->state.resize(ctx->n1);
    for (int cur_rank = 0; cur_rank < ctx->world_size; cur_rank++)
        ctx->state[cur_rank % ctx->n1].push_back(cur_rank);
    
    int max_ranks_per_block = (int)ctx->state[0].size();
    for (size_t i = 1; i < ctx->state.size(); i++)
        if ((int)ctx->state[i].size() < max_ranks_per_block)
            ctx->state[i].push_back(ctx->state[i][ctx->state[i].size() - 1]);
}

RRPContext rrp_distribute(int world_rank, int world_size, int m1, int n1, int m2, int n2) {
    RRPContext ctx;
    ctx.simple_case = world_size <= n1;
    if (ctx.simple_case) {
        ctx.wbp_ctx = wbp_distribute(world_rank, world_size, m1, n1, m2, n2);
        return ctx;
    }

    ctx.world_rank = world_rank;
    ctx.world_size = world_size;
    ctx.m1 = m1;
    ctx.n1 = n1;
    ctx.m2 = m2;
    ctx.n2 = n2;
    
    rrp_compute_state(&ctx);

    int intra_block_index = -1, inter_block_index = -1;
    for (size_t i = 0; i < ctx.state.size(); i++) {
        for (size_t j = 0; j < ctx.state[i].size(); j++) {
            if (ctx.state[i][j] == world_rank) {
                intra_block_index = j;
                inter_block_index = i;
                ctx.is_on_edge = j == ctx.state[i].size() - 2 && ctx.state[i][j+1] == world_rank;
                ctx.is_on_edge_natural = j == ctx.state[i].size() - 1;
                if (ctx.is_on_edge_natural) ctx.is_on_edge = true;
                break;
            }
        }
        if (intra_block_index != -1) break;
    }
    assert(intra_block_index != -1 && inter_block_index != -1);

    MPI_Comm_split(MPI_COMM_WORLD, ctx.is_on_edge_natural ? MPI_UNDEFINED : intra_block_index, world_rank, &ctx.v_comm_main);
    if (!ctx.is_on_edge_natural) {
        MPI_Comm_rank(ctx.v_comm_main, &ctx.v_main_rank);
        MPI_Comm_size(ctx.v_comm_main, &ctx.v_main_size);
    }
    
    MPI_Comm_split(MPI_COMM_WORLD, ctx.is_on_edge ? 1 : MPI_UNDEFINED, world_rank, &ctx.v_comm_edge);
    if (ctx.is_on_edge) {
        MPI_Comm_rank(ctx.v_comm_edge, &ctx.v_edge_rank);
        MPI_Comm_size(ctx.v_comm_edge, &ctx.v_edge_size);
    }

    MPI_Comm_split(MPI_COMM_WORLD, inter_block_index, world_rank, &ctx.u_comm);
    MPI_Comm_rank(ctx.u_comm, &ctx.u_rank);
    MPI_Comm_size(ctx.u_comm, &ctx.u_size);

    std::vector<int> cols_per_rank(ctx.state[0].size());
    int total_cols = n2;
    int base_cols = total_cols / ctx.state[0].size();
    int rem_cols = total_cols % ctx.state[0].size();
    for (size_t i = 0; i < ctx.state[0].size(); i++)
         cols_per_rank[i] = base_cols + (i < rem_cols ? 1 : 0);

    ctx.cols_owned_main = cols_per_rank[intra_block_index];
    if (ctx.is_on_edge) {
        if (ctx.is_on_edge_natural) {
            ctx.cols_owned_edge = cols_per_rank[intra_block_index];
            ctx.cols_owned_main = 0;
        } else {
            ctx.cols_owned_edge = cols_per_rank[intra_block_index + 1];
        }
    } else {
        ctx.cols_owned_edge = 0;
    }

    ctx.A_local_size = (size_t)ctx.m1 * (size_t)ctx.m2 * (size_t)(ctx.cols_owned_main + ctx.cols_owned_edge);
    ctx.v_recv_main_size = (size_t)ctx.m2 * (size_t)(ctx.cols_owned_main);
    ctx.v_recv_edge_size = (size_t)ctx.m2 * (size_t)(ctx.cols_owned_edge);
    ctx.u_recv_size = (size_t)ctx.m1;


    if (ctx.is_on_edge_natural) {
        ctx.v_send_main_size = 0;
    } else {
        int base_v_send_main_size = ctx.v_recv_main_size / ctx.v_main_size;
        int rem_v_send_main_size = ctx.v_recv_main_size % ctx.v_main_size;
        ctx.v_send_main_size = base_v_send_main_size + (ctx.v_main_rank < rem_v_send_main_size ? 1 : 0);
    }

    if (ctx.is_on_edge) {
        int base_v_send_edge_size = ctx.v_recv_edge_size / ctx.v_edge_size;
        int rem_v_send_edge_size = ctx.v_recv_edge_size % ctx.v_edge_size;
        ctx.v_send_edge_size = base_v_send_edge_size + (ctx.v_edge_rank < rem_v_send_edge_size ? 1 : 0);
    } else {
        ctx.v_send_edge_size = 0;
    }

    int base_u_send_size = ctx.u_recv_size / ctx.u_size;
    int rem_u_send_size = ctx.u_recv_size % ctx.u_size;
    ctx.u_send_size = base_u_send_size + (ctx.u_rank < rem_u_send_size ? 1 : 0);

    ctx.A_local = (double*)malloc(ctx.A_local_size * sizeof(double));
    ctx.v_send_main = (double*)malloc(ctx.v_send_main_size * sizeof(double));
    ctx.v_recv = (double*)malloc((size_t)(ctx.v_recv_main_size + ctx.v_recv_edge_size) * sizeof(double));
    if (ctx.is_on_edge) {
        ctx.v_send_edge = (double*)malloc(ctx.v_send_edge_size * sizeof(double));
    } else {
        ctx.v_send_edge = nullptr;
    }
    ctx.u_send = (double*)malloc(ctx.u_send_size * sizeof(double));
    ctx.u_recv = (double*)malloc(ctx.u_recv_size * sizeof(double));

    // for (size_t i = 0; i < ctx.A_local_size; i++)
    //     ctx.A_local[i] = (double)world_rank;
    // for (size_t i = 0; i < ctx.v_send_main_size; i++)
    //     ctx.v_send_main[i] = (double)world_rank;
    // for (size_t i = 0; i < ctx.v_send_edge_size; i++)
    //     ctx.v_send_edge[i] = (double)world_rank;
    // for (size_t i = 0; i < ctx.u_send_size; i++)
    //     ctx.u_send[i] = (double)world_rank;
    for (size_t i = 0; i < ctx.A_local_size; i++)
        ctx.A_local[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < ctx.v_send_main_size; i++)
        ctx.v_send_main[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < ctx.v_send_edge_size; i++)
        ctx.v_send_edge[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < ctx.u_send_size; i++)
        ctx.u_send[i] = (double)rand() / RAND_MAX;

    find_revcounts_displs(ctx.u_send_size, ctx.u_size, &ctx.recvcounts_u, &ctx.displs_u, ctx.u_comm);
    if (!ctx.is_on_edge_natural)
        find_revcounts_displs(ctx.v_send_main_size, ctx.v_main_size, &ctx.recvcounts_v_main, &ctx.displs_v_main, ctx.v_comm_main);
    if (ctx.is_on_edge)
        find_revcounts_displs(ctx.v_send_edge_size, ctx.v_edge_size, &ctx.recvcounts_v_edge, &ctx.displs_v_edge, ctx.v_comm_edge);

    return ctx;
}

void rrp_reconstruct_A_tilde(double** A_full, RRPContext* ctx) {
    if (ctx->simple_case) {
        wbp_reconstruct_A_tilde(A_full, &ctx->wbp_ctx);
        return;
    }

    if (ctx->world_rank == 0) {
        *A_full = (double*)malloc((size_t)ctx->m1 * (size_t)ctx->m2 * (size_t)ctx->n2 * ctx->n1 * sizeof(double));
    }

    int times_touched = 0;
    int blocks_consumed = 0;
    for (int i = 0; i < ctx->n1; i++) {
        for (int j = 0; j < ctx->m1; j++) {
            int A_full_offset = blocks_consumed * ctx->m2 * ctx->n2;
            for (int k = 0; k < (int)ctx->state[i].size(); k++) {
                int cur_rank = ctx->state[i][k];
                bool on_edge = k == (int)ctx->state[i].size() - 2 && ctx->state[i][k+1] == cur_rank;
                if (on_edge) continue;
                if (cur_rank == 0 && ctx->world_rank == 0) {
                    int data_to_send = (ctx->cols_owned_main + ctx->cols_owned_edge) * ctx->m2;
                    int A_local_offset = (size_t)times_touched * data_to_send;
                    times_touched++;
                    for (size_t idx = 0; idx < (size_t)data_to_send; idx++) {
                        (*A_full)[A_full_offset + idx] = ctx->A_local[A_local_offset + idx];
                    }
                    A_full_offset += data_to_send;
                } else {
                    if (cur_rank == ctx->world_rank) {
                        int data_to_send = (ctx->cols_owned_main + ctx->cols_owned_edge) * ctx->m2;
                        int A_local_offset = (size_t)times_touched * data_to_send;
                        times_touched++;
                        // send to rank 0 how much data I am sending so it can post the right recv
                        MPI_Send(&data_to_send, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                        MPI_Send(ctx->A_local + A_local_offset, data_to_send, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                    if (ctx->world_rank == 0) {
                        int data_to_recv;
                        MPI_Recv(&data_to_recv, 1, MPI_INT, cur_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(*A_full + A_full_offset, data_to_recv, MPI_DOUBLE, cur_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        A_full_offset += data_to_recv;
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            blocks_consumed++;
        }
    }
}

void rrp_reconstruct_v(double** v_full, RRPContext* ctx) {
    if (ctx->simple_case) {
        wbp_reconstruct_v(v_full, &ctx->wbp_ctx);
        return;
    }

    if (ctx->world_rank == 0) {
        *v_full = (double*)malloc((size_t)ctx->m2 * (size_t)ctx->n2 * sizeof(double));
    }

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

    // now iterate through state[0] to all gather the pieces in the right order to v_full
    int v_offset_a = 0;
    for (size_t j = 0; j < (size_t)ctx->state[0].size(); j++) {
        int cur_rank = ctx->state[0][j];
        if (cur_rank == 0 && ctx->world_rank == 0) {
            int data_to_send = ctx->v_recv_main_size + ctx->v_recv_edge_size;
            for (size_t idx = 0; idx < (size_t)data_to_send; idx++) {
                (*v_full)[v_offset_a + idx] = ctx->v_recv[idx];
            }
            v_offset_a += data_to_send;
        } else {
            if (cur_rank == ctx->world_rank) {
                int data_to_send = ctx->v_recv_main_size + ctx->v_recv_edge_size;
                MPI_Send(&data_to_send, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(ctx->v_recv, data_to_send, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
            if (ctx->world_rank == 0) {
                int data_to_recv;
                MPI_Recv(&data_to_recv, 1, MPI_INT, cur_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(*v_full + v_offset_a, data_to_recv, MPI_DOUBLE, cur_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                v_offset_a += data_to_recv;
            }
        }
    }
}

void rrp_reconstruct_u(double** u_full, RRPContext* ctx) {
    if (ctx->simple_case) {
        wbp_reconstruct_u(u_full, &ctx->wbp_ctx);
        return;
    }

    if (ctx->world_rank == 0) {
        *u_full = (double*)malloc((size_t)ctx->m1 * ctx->n1 * sizeof(double));
    }

    MPI_Allgatherv(ctx->u_send, ctx->u_send_size, MPI_DOUBLE,
                ctx->u_recv, ctx->recvcounts_u, ctx->displs_u, MPI_DOUBLE,
                ctx->u_comm);

    int u_offset_a = 0;
    for (size_t i = 0; i < ctx->n1; i++) {
        int cur_rank = ctx->state[i][0];
        if (cur_rank == 0 && ctx->world_rank == 0) {
            int data_to_send = ctx->u_recv_size;
            for (size_t idx = 0; idx < (size_t)data_to_send; idx++) {
                (*u_full)[u_offset_a + idx] = ctx->u_recv[idx];
            }
            u_offset_a += data_to_send;
        } else {
            if (cur_rank == ctx->world_rank) {
                int data_to_send = ctx->u_recv_size;
                MPI_Send(&data_to_send, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                MPI_Send(ctx->u_recv, data_to_send, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
            if (ctx->world_rank == 0) {
                int data_to_recv;
                MPI_Recv(&data_to_recv, 1, MPI_INT, cur_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(*u_full + u_offset_a, data_to_recv, MPI_DOUBLE, cur_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                u_offset_a += data_to_recv;
            }
        }
    }
}

void rrp_free_context(RRPContext* ctx) {
    if (ctx->simple_case) {
        wbp_free_context(&ctx->wbp_ctx);
        return;
    }
    free(ctx->A_local);
    free(ctx->v_send_main);
    free(ctx->v_send_edge);
    free(ctx->u_send);
    free(ctx->u_recv);
    if (!ctx->is_on_edge_natural) {
        free(ctx->recvcounts_v_main);
        free(ctx->displs_v_main);
    }
    if (ctx->is_on_edge) {
        free(ctx->recvcounts_v_edge);
        free(ctx->displs_v_edge);
    }
}

#endif // RRP_DISTR_H
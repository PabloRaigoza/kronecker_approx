#ifndef BCP_DISTR_H
#define BCP_DISTR_H

#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <cassert>
#include <algorithm>
#include "../common.h"

// If world_size <= n1 we essentially wrap WBP distribution
struct BCPContext {
    int world_rank, world_size;
    int m1, n1, m2, n2;
    int cols_owned_intra, cols_owned_inter;
    double *A_local;
    int A_local_size;
    
    double *v_send, *v_recv;
    int v_send_size, v_recv_size;

    double *u_recv, *u_send;
    int u_recv_size, u_send_size;
    
    MPI_Comm v_comm, u_comm;
    int *recvcounts_v, *displs_v;
    int *recvcounts_u, *displs_u;
    int v_rank, v_size;
    int u_rank, u_size;
    std::vector<std::vector<int>> state;    
    std::vector<int> cols_per_rank;
};

void bcp_split_each_rank(BCPContext* ctx, std::vector<std::vector<int>>& state, int& num_ranks) {
    std::vector<std::vector<int>> new_state(ctx->n1);
    for (int i = 0; i < ctx->n1; i++) {
        int states_to_add = std::max(ctx->n2 - (int)state[i].size(), 0);
        for (size_t j = 0; j < state[i].size(); j++) {
            new_state[i].push_back(state[i][j]);
            if (states_to_add > 0) {
                new_state[i].push_back(state[i][j] + num_ranks);
                states_to_add--;
            }
        }
    }
    num_ranks *= 2;
    state = new_state;
}

void sort_partition(std::vector<std::vector<int>>& state) {
    for (size_t i = 0; i < state.size(); i++)
        std::sort(state[i].begin(), state[i].end());
}

void bcp_split_columns(BCPContext* ctx, std::vector<std::vector<int>>& state, int& num_ranks) {
    bool is_last = num_ranks * 2 > ctx->n1;

    std::vector<std::vector<int>> new_state(ctx->n1);

    for (int rank = 0; rank < num_ranks; rank++) {
        std::vector<int> columns;
        for (int i = 0; i < ctx->n1; i++) {
            if (std::find(state[i].begin(), state[i].end(), rank) != state[i].end()) {
                columns.push_back(i);
            }
        }

        if (!is_last) {
            int split_point = columns.size() / 2;
            for (int i = 0; i < split_point; i++) {
                new_state[columns[i]].push_back(rank);
            }
            for (size_t i = split_point; i < columns.size(); i++) {
                new_state[columns[i]].push_back(rank + num_ranks);
            }
        } else {
            if (columns.size() == 1) {
                new_state[columns[0]].push_back(rank);
            } else {
                int split_point = columns.size() / 2;
                for (int i = 0; i < split_point; i++) {
                    new_state[columns[i]].push_back(rank);
                }
                for (size_t i = split_point; i < columns.size(); i++) {
                    new_state[columns[i]].push_back(rank + num_ranks);
                }
            }
        }
    }

    sort_partition(new_state);  // assuming in-place; adjust if it returns a new vector
    state = new_state;
    num_ranks *= 2;
}

BCPContext bcp_distribute(int world_rank, int world_size, int m1, int n1, int m2, int n2) {
    BCPContext ctx;
    ctx.world_rank = world_rank;
    ctx.world_size = world_size;
    ctx.m1 = m1;
    ctx.n1 = n1;
    ctx.m2 = m2;
    ctx.n2 = n2;

    ctx.state.resize(ctx.n1, std::vector<int>(1, 0));
    int num_ranks = 1;
    for (int i = 0; num_ranks < ctx.world_size; i++) {
        if (i % 2 == 1) bcp_split_each_rank(&ctx, ctx.state, num_ranks);
        else bcp_split_columns(&ctx, ctx.state, num_ranks);
    }

    int intra_block_index = -1, inter_block_index = -1;
    for (size_t i = 0; i < ctx.n1; i++) {
        for (size_t j = 0; j < ctx.state[i].size(); j++) {
            if (ctx.state[i][j] == world_rank) {
                intra_block_index = j;
                inter_block_index = i;
                break;
            }
        }
        if (intra_block_index != -1) break;
    }
    assert(intra_block_index != -1 && inter_block_index != -1);

    MPI_Comm_split(MPI_COMM_WORLD, intra_block_index, world_rank, &ctx.v_comm);
    MPI_Comm_rank(ctx.v_comm, &ctx.v_rank);
    MPI_Comm_size(ctx.v_comm, &ctx.v_size);

    MPI_Comm_split(MPI_COMM_WORLD, inter_block_index, world_rank, &ctx.u_comm);
    MPI_Comm_rank(ctx.u_comm, &ctx.u_rank);
    MPI_Comm_size(ctx.u_comm, &ctx.u_size);

    int total_cols = ctx.n2;
    int base_cols = total_cols / ctx.state[0].size();
    int rem_cols = total_cols % ctx.state[0].size();
    ctx.cols_per_rank.resize(ctx.state[0].size());
    for (int i = 0; i < ctx.state[0].size(); i++)
        ctx.cols_per_rank[i] = base_cols + (i < rem_cols ? 1 : 0);
    
    ctx.cols_owned_intra = ctx.cols_per_rank[intra_block_index];
    ctx.cols_owned_inter = 0;
    for (size_t i = 0; i < ctx.n1; i++)
        if (ctx.state[i][intra_block_index] == world_rank)
            ctx.cols_owned_inter++;

    ctx.A_local_size = (size_t)ctx.m1 * (size_t)ctx.m2 * (size_t)ctx.cols_owned_intra * (size_t)ctx.cols_owned_inter;
    ctx.v_recv_size = (size_t)ctx.m2 * (size_t)ctx.cols_owned_intra;
    ctx.u_recv_size = (size_t)ctx.m1 * (size_t)ctx.cols_owned_inter;

    int total_v_send = ctx.v_recv_size;
    int base_v_send = total_v_send / ctx.v_size;
    int rem_v_send = total_v_send % ctx.v_size;
    ctx.v_send_size = base_v_send + (ctx.v_rank < rem_v_send ? 1 : 0);

    int total_u_send = ctx.u_recv_size;
    int base_u_send = total_u_send / ctx.u_size;
    int rem_u_send = total_u_send % ctx.u_size;
    ctx.u_send_size = base_u_send + (ctx.u_rank < rem_u_send ? 1 : 0);

    ctx.A_local = (double*)malloc(ctx.A_local_size * sizeof(double));
    ctx.v_recv = (double*)malloc(ctx.v_recv_size * sizeof(double));
    ctx.u_recv = (double*)malloc(ctx.u_recv_size * sizeof(double));
    ctx.v_send = (double*)malloc(ctx.v_send_size * sizeof(double));
    ctx.u_send = (double*)malloc(ctx.u_send_size * sizeof(double));

    for (size_t i = 0; i < ctx.A_local_size; i++) ctx.A_local[i] = (double)world_rank;
    for (size_t i = 0; i < (size_t)ctx.v_recv_size; i++) ctx.v_recv[i] = (double)world_rank;
    for (size_t i = 0; i < (size_t)ctx.u_recv_size; i++) ctx.u_recv[i] = (double)world_rank;
    for (size_t i = 0; i < (size_t)ctx.v_send_size; i++) ctx.v_send[i] = (double)world_rank;
    for (size_t i = 0; i < (size_t)ctx.u_send_size; i++) ctx.u_send[i] = (double)world_rank;

    for (size_t i = 0; i < ctx.A_local_size; i++) ctx.A_local[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)ctx.v_recv_size; i++) ctx.v_recv[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)ctx.u_recv_size; i++) ctx.u_recv[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)ctx.v_send_size; i++) ctx.v_send[i] = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)ctx.u_send_size; i++) ctx.u_send[i] = (double)rand() / RAND_MAX;

    find_revcounts_displs(ctx.v_send_size, ctx.v_size, &ctx.recvcounts_v, &ctx.displs_v, ctx.v_comm);
    find_revcounts_displs(ctx.u_send_size, ctx.u_size, &ctx.recvcounts_u, &ctx.displs_u, ctx.u_comm);

    return ctx;
}

void bcp_reconstruct_A_tilde(double** A_full, BCPContext* ctx) {
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
                if (cur_rank == 0 && ctx->world_rank == 0) {
                    int data_to_send = ctx->cols_owned_intra * ctx->m2;
                    int A_local_offset = (size_t)times_touched * data_to_send;
                    times_touched++;
                    for (size_t idx = 0; idx < (size_t)data_to_send; idx++) {
                        (*A_full)[A_full_offset + idx] = ctx->A_local[A_local_offset + idx];
                    }
                    A_full_offset += data_to_send;
                } else {
                    if (cur_rank == ctx->world_rank) {
                        int data_to_send = ctx->cols_owned_intra * ctx->m2;
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

void bcp_reconstruct_v(double** v_full, BCPContext* ctx) {
    if (ctx->world_rank == 0) {
        *v_full = (double*)malloc((size_t)ctx->m2 * (size_t)ctx->n2 * sizeof(double));
    }

    MPI_Allgatherv(ctx->v_send, ctx->v_send_size, MPI_DOUBLE,
                    ctx->v_recv, ctx->recvcounts_v, ctx->displs_v, MPI_DOUBLE,
                    ctx->v_comm);

    int v_offset_a = 0;
    for (size_t j = 0; j < (size_t)ctx->state[0].size(); j++) {
        int cur_rank = ctx->state[0][j];
        if (cur_rank == 0 && ctx->world_rank == 0) {
            int data_to_send = ctx->v_recv_size;
            for (size_t idx = 0; idx < (size_t)data_to_send; idx++) {
                (*v_full)[v_offset_a + idx] = ctx->v_recv[idx];
            }
            v_offset_a += data_to_send;
        } else {
            if (cur_rank == ctx->world_rank) {
                int data_to_send = ctx->v_recv_size;
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

void bcp_reconstruct_u(double** u_full, BCPContext* ctx) {
    if (ctx->world_rank == 0) {
        *u_full = (double*)malloc((size_t)ctx->m1 * (size_t)ctx->n1 * sizeof(double));
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

void bcp_free_context(BCPContext* ctx) {
    free(ctx->A_local);
    free(ctx->v_send);
    free(ctx->v_recv);
    free(ctx->u_send);
    free(ctx->u_recv);
    free(ctx->recvcounts_v);
    free(ctx->displs_v);
    free(ctx->recvcounts_u);
    free(ctx->displs_u);
}

#endif // BCP_DISTR_H
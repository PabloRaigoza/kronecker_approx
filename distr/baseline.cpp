// baseline.cpp
//
// A "naive" distributed baseline for the A~ operator, to compare against the
// Kronecker-aware WBP/RRP/BCP partitioning schemes elsewhere in this repo.
// Those schemes generate their local data directly in A~'s block layout and
// so never actually pay to materialize A~. This program does the opposite:
// it starts from A's m1*n1 blocks handed out in a Kronecker-oblivious
// "raster" order (block (i,j) -> index i*n1+j, as if just streamed off disk
// in row-major block order), explicitly permutes that data into A~'s layout
// (row j*m1+i, per CLAUDE.md's A-tilde reformulation) with MPI_Alltoallv,
// and then hands the materialized A~ to SLATE as a standard 2-D
// block-cyclic distributed matrix to compute A~*x and A~^T*x.
//
// SLATE targets BLAS-3 (it has no distributed gemv), so the mat-vec is done
// via slate::gemm against the vector treated as an (n x 1) "skinny" matrix,
// and A~^T*x reuses A~'s tiles through slate::transpose (an algebraic view,
// no extra copy).
//
// This isolates two costs the specialized algorithms avoid paying at all:
//   1. the one-time re-permutation communication, timed around
//      MPI_Alltoallv (communication only)
//   2. the recurring cost of each A~/A~^T mat-vec, timed around
//      slate::gemm (communication + computation together, as SLATE mixes
//      the two internally and does not expose them separately)
//
// NOTE: SLATE (github.com/icl-utk-edu/slate) is not installed in the
// sandbox this file was written in, so it could not be compiled or run
// here. The SLATE calls below (Matrix, insertLocalTiles, tileRank,
// tileIsLocal, at, gemm, transpose) follow SLATE's documented public
// interface; verify them against the SLATE version on the target cluster
// (module load / site build) before relying on this.
//
// Usage:
//   mpirun -np <P> ./baseline <m1> <n1> <m2> <n2> [nb] [trials]
//
// nb     - SLATE tile size for A~ (and for the vector operands); default 256
// trials - number of timed repetitions to average, default 10

#include <mpi.h>
#include <slate/slate.hh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Ownership of the m1*n1 A-blocks in natural "raster" (i-major) order:
// block (i,j) has flat index i*n1+j, and consecutive ranks own contiguous
// ranges of that index. This is independent of A~'s own row order
// (j*m1+i, j-major) -- that mismatch is exactly what forces the
// redistribution below to be real communication rather than a relabeling.
static void raster_block_range(int rank, int world_size, int total_blocks,
                                int *start, int *count) {
    int base = total_blocks / world_size;
    int rem = total_blocks % world_size;
    *count = base + (rank < rem ? 1 : 0);
    *start = rank * base + std::min(rank, rem);
}

// A roughly-square p x q process grid with p*q == world_size, matching the
// grid SLATE's Matrix constructor expects.
static int grid_dim(int world_size) {
    int p = (int)std::sqrt((double)world_size);
    while (p > 1 && world_size % p != 0) p--;
    return p;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 5) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: %s <m1> <n1> <m2> <n2> [nb] [trials]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int m1 = atoi(argv[1]), n1 = atoi(argv[2]);
    int m2 = atoi(argv[3]), n2 = atoi(argv[4]);
    int nb = argc > 5 ? atoi(argv[5]) : 256;
    int trials = argc > 6 ? atoi(argv[6]) : 10;

    const int64_t M_tilde = (int64_t)m1 * n1;  // rows of A~
    const int64_t N_tilde = (int64_t)m2 * n2;  // cols of A~
    const int block_elems = m2 * n2;           // elements per A-block == one A~ row

    int p = grid_dim(world_size);
    int q = world_size / p;

    // --- Step 1: A's Kronecker-oblivious starting layout --------------------
    // Each rank owns a contiguous range of the m1*n1 blocks in raster order,
    // filled with placeholder values (matches the wbp/rrp/bcp convention of
    // using rank id / random data rather than a specific matrix, since this
    // is purely a timing benchmark, not a correctness check).
    int total_blocks = m1 * n1;
    int local_start, local_count;
    raster_block_range(world_rank, world_size, total_blocks, &local_start, &local_count);

    std::vector<double> A_local((size_t)local_count * block_elems);
    for (size_t k = 0; k < A_local.size(); k++) {
        A_local[k] = (double)world_rank;
    }

    // --- Step 2: A~ as a real SLATE 2-D block-cyclic distributed matrix -----
    slate::Matrix<double> A_tilde(M_tilde, N_tilde, nb, p, q, MPI_COMM_WORLD);
    A_tilde.insertLocalTiles();
    const int64_t nt = A_tilde.nt();  // tile columns

    // --- Step 3: work out where each local A-block's data must go -----------
    // Block (i,j) [raster index k = i*n1+j] becomes row r = j*m1+i of A~,
    // spanning ALL N_tilde columns. It lands in exactly one A~ tile-ROW
    // (R = r/nb) but is split across every tile-COLUMN in that row, each
    // potentially owned by a different rank. We ask SLATE itself
    // (tileRank) who owns each destination tile rather than re-deriving its
    // 2-D block-cyclic formula, so this stays correct regardless of the
    // exact process-grid convention the installed SLATE uses internally.
    // tileRank is a pure ownership query -- it is safe (and, since A~'s
    // distribution is a shared deterministic function, gives a globally
    // consistent answer) to call it for tiles this rank does not own.
    std::vector<int> send_counts(world_size, 0);
    for (int k = 0; k < local_count; k++) {
        int blk = local_start + k;
        int i = blk / n1, j = blk % n1;
        int64_t r = (int64_t)j * m1 + i;
        int64_t R = r / nb;
        for (int64_t J = 0; J < nt; J++) {
            int owner = A_tilde.tileRank(R, J);
            int64_t col_start = J * nb;
            int64_t col_end = std::min(col_start + nb, N_tilde);
            send_counts[owner] += (int)(col_end - col_start);
        }
    }

    std::vector<int> recv_counts(world_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> send_displs(world_size), recv_displs(world_size);
    send_displs[0] = recv_displs[0] = 0;
    for (int r = 1; r < world_size; r++) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }
    int send_total = send_displs[world_size - 1] + send_counts[world_size - 1];
    int recv_total = recv_displs[world_size - 1] + recv_counts[world_size - 1];

    std::vector<double> send_buf(send_total);
    {
        std::vector<int> cursor = send_displs;  // per-destination write cursor
        for (int k = 0; k < local_count; k++) {
            int blk = local_start + k;
            int i = blk / n1, j = blk % n1;
            int64_t r = (int64_t)j * m1 + i;
            int64_t R = r / nb;
            const double *block_data = &A_local[(size_t)k * block_elems];
            for (int64_t J = 0; J < nt; J++) {
                int owner = A_tilde.tileRank(R, J);
                int64_t col_start = J * nb;
                int64_t col_end = std::min(col_start + nb, N_tilde);
                int len = (int)(col_end - col_start);
                std::copy(block_data + col_start, block_data + col_end,
                           send_buf.begin() + cursor[owner]);
                cursor[owner] += len;
            }
        }
    }
    std::vector<double> recv_buf(recv_total);

    // --- Timed: the pure re-permutation communication ------------------------
    double redistribute_time = 0.0;
    for (int t = 0; t < trials; t++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                      recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                      MPI_COMM_WORLD);
        redistribute_time += MPI_Wtime() - t0;
    }
    redistribute_time /= trials;

    // --- Step 4: unpack recv_buf into A~'s local SLATE tiles -----------------
    // recv_buf's segment for source rank `src` (recv_displs[src] .. +
    // recv_counts[src]) holds exactly the chunks `src` classified as
    // owner==world_rank while packing, in that same iteration order. Every
    // rank can recompute `src`'s block ownership locally (raster_block_range
    // is a pure function, no communication needed), so we replay that same
    // iteration here to know how to walk each source's segment.
    for (int src = 0; src < world_size; src++) {
        int s_start, s_count;
        raster_block_range(src, world_size, total_blocks, &s_start, &s_count);
        int cursor = recv_displs[src];
        for (int k = 0; k < s_count; k++) {
            int blk = s_start + k;
            int i = blk / n1, j = blk % n1;
            int64_t r = (int64_t)j * m1 + i;
            int64_t R = r / nb;
            int64_t local_row = r - R * nb;
            for (int64_t J = 0; J < nt; J++) {
                int owner = A_tilde.tileRank(R, J);
                int64_t col_start = J * nb;
                int64_t col_end = std::min(col_start + nb, N_tilde);
                int len = (int)(col_end - col_start);
                if (owner == world_rank) {
                    auto T = A_tilde.at(R, J);
                    // SLATE tiles are column-major (BLAS/LAPACK convention),
                    // unlike this repo's row-major A~ layout elsewhere.
                    double *dst = T.data();
                    int64_t ld = T.stride();
                    for (int c = 0; c < len; c++) {
                        dst[local_row + (int64_t)c * ld] = recv_buf[cursor + c];
                    }
                    cursor += len;
                }
                // else: this chunk belongs to a different rank and never
                // appears in our recv_buf -- don't touch cursor for it.
            }
        }
    }

    // --- Step 5: vectors as skinny (n x 1) SLATE matrices ---------------------
    slate::Matrix<double> V(N_tilde, 1, nb, p, q, MPI_COMM_WORLD);  // input to A~*x
    slate::Matrix<double> U(M_tilde, 1, nb, p, q, MPI_COMM_WORLD);  // output of A~*x
    V.insertLocalTiles();
    U.insertLocalTiles();
    for (int64_t I = 0; I < V.mt(); I++) {
        if (V.tileIsLocal(I, 0)) {
            auto T = V.at(I, 0);
            for (int64_t r = 0; r < T.mb(); r++) T.data()[r] = (double)world_rank;
        }
    }

    // --- Timed: A~ * x  (communication + computation) -------------------------
    double ax_time = 0.0;
    for (int t = 0; t < trials; t++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        slate::gemm(1.0, A_tilde, V, 0.0, U);
        ax_time += MPI_Wtime() - t0;
    }
    ax_time /= trials;

    // --- Timed: A~^T * u  (communication + computation) ------------------------
    slate::Matrix<double> U2(M_tilde, 1, nb, p, q, MPI_COMM_WORLD);  // input to A~^T*x
    slate::Matrix<double> V2(N_tilde, 1, nb, p, q, MPI_COMM_WORLD);  // output of A~^T*x
    U2.insertLocalTiles();
    V2.insertLocalTiles();
    for (int64_t I = 0; I < U2.mt(); I++) {
        if (U2.tileIsLocal(I, 0)) {
            auto T = U2.at(I, 0);
            for (int64_t r = 0; r < T.mb(); r++) T.data()[r] = (double)world_rank;
        }
    }
    auto A_tilde_T = slate::transpose(A_tilde);  // algebraic view, shares tiles with A_tilde

    double atx_time = 0.0;
    for (int t = 0; t < trials; t++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        slate::gemm(1.0, A_tilde_T, U2, 0.0, V2);
        atx_time += MPI_Wtime() - t0;
    }
    atx_time /= trials;

    // --- Report ----------------------------------------------------------------
    double total_redistribute = 0.0, total_ax = 0.0, total_atx = 0.0;
    MPI_Reduce(&redistribute_time, &total_redistribute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ax_time, &total_ax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&atx_time, &total_atx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        total_redistribute /= world_size;
        total_ax /= world_size;
        total_atx /= world_size;
        printf("Experiment: %dx%dx%dx%d %d ranks baseline (nb=%d, grid=%dx%d)\n",
               m1, n1, m2, n2, world_size, nb, p, q);
        printf("Mean Redistribution (comm only): %.6f | Mean A~x (comm+comp): %.6f | "
               "Mean A~^Tx (comm+comp): %.6f\n",
               total_redistribute, total_ax, total_atx);
    }

    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <math.h>
#include "common.h"
#include "wbp/wbp_distr.h"
#include "wbp/wbp_kernel.h"
#include "rrp/rrp_distr.h"
#include "rrp/rrp_kernel.h"
#include "bcp/bcp_distr.h"
#include "bcp/bcp_kernel.h"
#include "bidiag.h"
#include "verify.h"

#define NUM_TRIALS 25
#define NUM_BIDIAG_TRIALS 1  // each trial already averages over 2k matvecs (k steps)

// Usage: main <m1> <n1> <m2> <n2> <op> <alg> [seed] [k] [reorth]
//   op:     Ax | ATx | bidiag | verify
//   alg:    wbp | rrp | bcp
//   seed:   seeds Ã and the starting vectors (default 42); the same seed
//           gives the same Ã for every scheme and rank count
//   k:      Golub-Kahan steps for bidiag / verify (default 50)
//   reorth: 1 = full reorthogonalization for bidiag (default 0); verify
//           always checks both
template <class Ctx>
static int run_op(Ctx* ctx, const char* op, int m1, int n1, int m2, int n2, uint64_t seed, int k, bool reorth) {
    if (strcmp(op, "bidiag") == 0) {
        BidiagResult r;
        BidiagTimers acc;
        for (int trial = 0; trial < NUM_BIDIAG_TRIALS; trial++) {
            r = bidiagonalize(ctx, k, seed, reorth);
            bidiag_timers_add(&acc, r.timers);
        }
        bidiag_print_report(r, acc, NUM_BIDIAG_TRIALS);
        return 0;
    }
    if (strcmp(op, "verify") == 0) {
        bool ok = verify_kernels(ctx, m1, n1, m2, n2, seed);
        ok = verify_bidiag(ctx, m1, n1, m2, n2, seed, k) && ok;
        return ok ? 0 : 1;
    }
    return -1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 7) {
        if (world_rank == 0)
            fprintf(stderr, "Usage: %s <m1> <n1> <m2> <n2> <Ax|ATx|bidiag|verify> <wbp|rrp|bcp> [seed] [k] [reorth]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int m1 = atoi(argv[1]);
    int n1 = atoi(argv[2]);
    int m2 = atoi(argv[3]);
    int n2 = atoi(argv[4]);
    const char *op = argv[5];
    const char *alg_str = argv[6];
    uint64_t seed = argc > 7 ? strtoull(argv[7], NULL, 10) : 42;
    int k = argc > 8 ? atoi(argv[8]) : 50;
    bool reorth = argc > 9 ? atoi(argv[9]) != 0 : false;

    int status = 0;
    if (strcmp(alg_str, "wbp") == 0) {
        WBPContext ctx = wbp_distribute(world_rank, world_size, m1, n1, m2, n2, seed);
        if (strcmp(op, "Ax") == 0) wbp_ax(&ctx, NUM_TRIALS, false);
        else if (strcmp(op, "ATx") == 0) wbp_atx(&ctx, NUM_TRIALS, false);
        else status = run_op(&ctx, op, m1, n1, m2, n2, seed, k, reorth);
        wbp_free_context(&ctx);
    } else if (strcmp(alg_str, "rrp") == 0) {
        RRPContext ctx = rrp_distribute(world_rank, world_size, m1, n1, m2, n2, seed);
        if (strcmp(op, "Ax") == 0) rrp_ax(&ctx, NUM_TRIALS, false);
        else if (strcmp(op, "ATx") == 0) rrp_atx(&ctx, NUM_TRIALS, false);
        else status = run_op(&ctx, op, m1, n1, m2, n2, seed, k, reorth);
        rrp_free_context(&ctx);
    } else if (strcmp(alg_str, "bcp") == 0) {
        BCPContext ctx = bcp_distribute(world_rank, world_size, m1, n1, m2, n2, seed);
        if (strcmp(op, "Ax") == 0) bcp_ax(&ctx, NUM_TRIALS, false);
        else if (strcmp(op, "ATx") == 0) bcp_atx(&ctx, NUM_TRIALS, false);
        else status = run_op(&ctx, op, m1, n1, m2, n2, seed, k, reorth);
        bcp_free_context(&ctx);
    } else {
        status = -1;
    }

    if (status == -1 && world_rank == 0)
        fprintf(stderr, "Unknown op '%s' or alg '%s'\n", op, alg_str);

    MPI_Finalize();
    return status == 0 ? 0 : 1;
}

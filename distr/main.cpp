#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <math.h>
#include <time.h>
#include "common.h"
#include "wbp/wbp_distr.h"
#include "wbp/wbp_kernel.h"
#include "rrp/rrp_distr.h"
#include "rrp/rrp_kernel.h"
#include "bcp/bcp_distr.h"
#include "bcp/bcp_kernel.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    srand((unsigned)time(NULL));

    int world_rank, world_size;
    int m1, n1, m2, n2;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    m1 = atoi(argv[1]);
    n1 = atoi(argv[2]);
    m2 = atoi(argv[3]);
    n2 = atoi(argv[4]);
    const char *op = argv[5];

    // WBPContext ctx = wbp_distribute(world_rank, world_size, m1, n1, m2, n2);
    // if (strcmp(op, "Ax") == 0) wbp_ax(&ctx, 10, false);
    // else if (strcmp(op, "ATx") == 0) wbp_atx(&ctx, 10, false);
    // wbp_verify(&ctx);
    // free_wbp_context(&ctx);

    // RRPContext ctx = rrp_distribute(world_rank, world_size, m1, n1, m2, n2);
    // if (strcmp(op, "Ax") == 0) rrp_ax(&ctx, 1, false);
    // else if (strcmp(op, "ATx") == 0) rrp_atx(&ctx, 1, false);
    // rrp_verify(&ctx);
    // rrp_free_context(&ctx);

    BCPContext ctx = bcp_distribute(world_rank, world_size, m1, n1, m2, n2);
    if (strcmp(op, "Ax") == 0) bcp_ax(&ctx, 1, false);
    else if (strcmp(op, "ATx") == 0) bcp_atx(&ctx, 1, false);
    bcp_verify(&ctx);
    bcp_free_context(&ctx);

    MPI_Finalize();
    return 0;
}
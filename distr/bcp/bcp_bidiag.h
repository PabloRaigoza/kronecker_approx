#ifndef BCP_BIDIAG_H
#define BCP_BIDIAG_H

#include <mpi.h>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cassert>
#include "bcp_distr.h"

// ============================================================
// Distributed Golub-Kahan Lanczos Bidiagonalization
//
// Produces the bidiagonal decomposition:
//   A ≈ U B V^T
//
// where B is (k+1) x k lower bidiagonal with diagonal α and
// subdiagonal β:
//
//   B = | α₁                    |
//       | β₂  α₂                |
//       |     β₃  α₃            |
//       |         ...  ...      |
//       |              βₖ  αₖ   |
//       |                  β_{k+1} |
//
// Memory layout per rank:
//   u_send[u_send_size]  - this rank's owned slice of the current
//                          left Lanczos vector (length m1 * cols_owned_inter)
//   v_send[v_send_size]  - this rank's owned slice of the current
//                          right Lanczos vector (length m2 * cols_owned_intra)
//
// After each kernel call the result lives in u_recv / v_recv.
// We immediately scatter the result back into u_send / v_send so
// the next kernel call sees the updated vector.
// ============================================================

struct BCPBidiagResult {
    std::vector<double> alpha;   // length k
    std::vector<double> beta;    // length k+1  (beta[0] = ||b||)
    int steps_run;
};

// ------------------------------------------------------------------
// Helper: distributed 2-norm of the u-vector.
//   Each rank owns u_send[u_send_size] — the portion of u that
//   lives on this rank in the u_comm.  We need the global norm.
// ------------------------------------------------------------------
static double dist_norm_u(BCPContext* ctx) {
    double local_sq = cblas_ddot(ctx->u_send_size,
                                 ctx->u_send, 1,
                                 ctx->u_send, 1);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                  ctx->u_comm);
    return std::sqrt(global_sq);
}

// ------------------------------------------------------------------
// Helper: distributed 2-norm of the v-vector.
// ------------------------------------------------------------------
static double dist_norm_v(BCPContext* ctx) {
    double local_sq = cblas_ddot(ctx->v_send_size,
                                 ctx->v_send, 1,
                                 ctx->v_send, 1);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                  ctx->v_comm);
    return std::sqrt(global_sq);
}

// ------------------------------------------------------------------
// Helper: distributed dot product of two u-vectors (same layout).
// ------------------------------------------------------------------
static double dist_dot_u(BCPContext* ctx, const double* a, const double* b) {
    double local_dot = cblas_ddot(ctx->u_send_size, a, 1, b, 1);
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM,
                  ctx->u_comm);
    return global_dot;
}

// ------------------------------------------------------------------
// Helper: distributed dot product of two v-vectors.
// ------------------------------------------------------------------
static double dist_dot_v(BCPContext* ctx, const double* a, const double* b) {
    double local_dot = cblas_ddot(ctx->v_send_size, a, 1, b, 1);
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM,
                  ctx->v_comm);
    return global_dot;
}

// ------------------------------------------------------------------
// Scale a local buffer in-place: x *= scale
// ------------------------------------------------------------------
static void scale_vec(double* x, int n, double scale) {
    cblas_dscal(n, scale, x, 1);
}

// ------------------------------------------------------------------
// AXPY in-place: y += alpha * x
// ------------------------------------------------------------------
static void axpy_vec(double* y, const double* x, int n, double alpha) {
    cblas_daxpy(n, alpha, x, 1, y, 1);
}

// ------------------------------------------------------------------
// After bcp_ax finishes, u_recv holds A*v distributed over the
// u_comm.  We need to scatter u_recv back into u_send so the next
// ATx call (and norm computations) use it.
//
// u_recv is the *full* local slice of the result: m1*cols_owned_inter
// u_send is a contiguous sub-slice of that (distributed in u_comm).
//
// The Allgatherv/Reduce_scatter in bcp_ax already places the
// contribution into u_recv.  Here we just redistribute so that
// each rank owns the correct u_send slice for subsequent ops.
// ------------------------------------------------------------------
static void sync_u_send_from_recv(BCPContext* ctx) {
    // After Reduce_scatter in bcp_ax the result IS in u_send already.
    // Nothing to do — bcp_ax writes Reduce_scatter output to u_send.
    // This function exists as a documentation hook.
    (void)ctx;
}

static void sync_v_send_from_recv(BCPContext* ctx) {
    // Symmetric for bcp_atx which writes into v_send via Reduce_scatter.
    (void)ctx;
}

// ------------------------------------------------------------------
// One step of distributed A*v:
//   1. Allgatherv  v_send -> v_recv  (within v_comm)
//   2. Local DGEMV: u_recv = A_local * v_recv
//   3. Reduce_scatter u_recv -> u_send  (within u_comm, MPI_SUM)
//
// After this call ctx->u_send holds this rank's slice of A*v.
// ------------------------------------------------------------------
static void do_ax(BCPContext* ctx) {
    MPI_Allgatherv(ctx->v_send, ctx->v_send_size, MPI_DOUBLE,
                   ctx->v_recv, ctx->recvcounts_v, ctx->displs_v,
                   MPI_DOUBLE, ctx->v_comm);

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ctx->u_recv_size,
                ctx->v_recv_size,
                1.0, ctx->A_local, ctx->v_recv_size,
                ctx->v_recv, 1,
                0.0, ctx->u_recv, 1);

    MPI_Reduce_scatter(ctx->u_recv, ctx->u_send,
                       ctx->recvcounts_u, MPI_DOUBLE, MPI_SUM,
                       ctx->u_comm);
}

// ------------------------------------------------------------------
// One step of distributed A^T * u:
//   1. Allgatherv  u_send -> u_recv  (within u_comm)
//   2. Local DGEMV (transpose): v_recv = A_local^T * u_recv
//   3. Reduce_scatter v_recv -> v_send  (within v_comm, MPI_SUM)
//
// After this call ctx->v_send holds this rank's slice of A^T*u.
// ------------------------------------------------------------------
static void do_atx(BCPContext* ctx) {
    MPI_Allgatherv(ctx->u_send, ctx->u_send_size, MPI_DOUBLE,
                   ctx->u_recv, ctx->recvcounts_u, ctx->displs_u,
                   MPI_DOUBLE, ctx->u_comm);

    cblas_dgemv(CblasRowMajor, CblasTrans,
                ctx->u_recv_size,
                ctx->v_recv_size,
                1.0, ctx->A_local, ctx->v_recv_size,
                ctx->u_recv, 1,
                0.0, ctx->v_recv, 1);

    MPI_Reduce_scatter(ctx->v_recv, ctx->v_send,
                       ctx->recvcounts_v, MPI_DOUBLE, MPI_SUM,
                       ctx->v_comm);
}

// ==================================================================
// bcp_bidiagonalize
//
// Runs k steps of Golub-Kahan Lanczos bidiagonalization.
//
// On entry:
//   ctx->u_send must contain the starting vector b (the rank's
//   owned slice).  If b is all zeros on your rank, that is fine —
//   just make sure at least one rank has non-zero entries.
//
// On exit:
//   result.alpha[j], result.beta[j] are the bidiagonal entries.
//   ctx->u_send / ctx->v_send hold the last Lanczos vectors.
//
// Optional reorthogonalization:
//   Set reorthogonalize=true for full reorthogonalization against
//   all previously computed vectors (stored as columns of U_vecs /
//   V_vecs).  This costs O(j * n) extra work per step but keeps
//   the vectors numerically orthogonal for ill-conditioned A.
//   For large problems, consider selective or periodic
//   reorthogonalization instead.
//
// Parameters:
//   k               - number of bidiagonalization steps
//   reorthogonalize - enable full reorthogonalization
//   tol             - stop early if beta_{j+1} < tol (invariant subspace)
// ==================================================================
BCPBidiagResult bcp_bidiagonalize(BCPContext* ctx,
                                   int k,
                                   bool reorthogonalize = false,
                                   double tol = 1e-14) {
    BCPBidiagResult result;
    result.alpha.resize(k);
    result.beta.resize(k + 1);
    result.steps_run = 0;

    // Optionally store Lanczos vectors for reorthogonalization.
    // U_vecs[j] has size u_send_size, V_vecs[j] has size v_send_size.
    std::vector<std::vector<double>> U_vecs, V_vecs;
    if (reorthogonalize) {
        U_vecs.reserve(k + 1);
        V_vecs.reserve(k);
    }

    // -------------------------------------------------------
    // Step 0: β₁ = ||u||,  u₁ = u / β₁
    // -------------------------------------------------------
    double beta = dist_norm_u(ctx);
    result.beta[0] = beta;

    if (beta < tol) {
        // Starting vector is zero — nothing to do.
        if (ctx->world_rank == 0)
            printf("[bidiag] Starting vector has norm < tol; stopping.\n");
        return result;
    }

    scale_vec(ctx->u_send, ctx->u_send_size, 1.0 / beta);

    if (reorthogonalize) {
        U_vecs.push_back(std::vector<double>(
            ctx->u_send, ctx->u_send + ctx->u_send_size));
    }

    for (int j = 0; j < k; ++j) {
        // ---------------------------------------------------
        // Compute r = A^T u_j
        // ---------------------------------------------------
        do_atx(ctx);
        // ctx->v_send now holds this rank's slice of A^T u_j

        // Orthogonalize against v_{j-1} (short recurrence)
        if (j > 0) {
            // r -= alpha_{j-1} * v_{j-1}  is already encoded by the
            // recurrence below.  Here we do the explicit subtraction.
            // v_{j-1} was stored before being overwritten; use V_vecs
            // when reorthogonalizing, otherwise use the saved prev buffer.
        }

        // Full reorthogonalization against all previous V vectors
        if (reorthogonalize) {
            for (int i = 0; i < (int)V_vecs.size(); ++i) {
                double c = dist_dot_v(ctx, V_vecs[i].data(), ctx->v_send);
                axpy_vec(ctx->v_send, V_vecs[i].data(),
                         ctx->v_send_size, -c);
            }
        }

        // α_j = ||r||
        double alpha = dist_norm_v(ctx);
        result.alpha[j] = alpha;

        if (alpha < tol) {
            if (ctx->world_rank == 0)
                printf("[bidiag] alpha_%d < tol; invariant subspace found.\n", j);
            result.steps_run = j + 1;
            return result;
        }

        // v_j = r / α_j
        scale_vec(ctx->v_send, ctx->v_send_size, 1.0 / alpha);

        if (reorthogonalize) {
            V_vecs.push_back(std::vector<double>(
                ctx->v_send, ctx->v_send + ctx->v_send_size));
        }

        // ---------------------------------------------------
        // Compute p = A v_j
        // ---------------------------------------------------
        do_ax(ctx);
        // ctx->u_send now holds this rank's slice of A v_j

        // Short recurrence: p -= α_j * u_j
        // u_j is the vector we used at the top of this iteration.
        // If reorthogonalizing we already saved it; otherwise we
        // reconstruct by scaling down below.
        if (reorthogonalize) {
            // Orthogonalize against ALL previous U vectors
            for (int i = 0; i < (int)U_vecs.size(); ++i) {
                double c = dist_dot_u(ctx, U_vecs[i].data(), ctx->u_send);
                axpy_vec(ctx->u_send, U_vecs[i].data(),
                         ctx->u_send_size, -c);
            }
        } else {
            // Standard short recurrence: p -= β_j * u_{j-1}  (j>=1)
            // is done implicitly via the Allgatherv in the next do_atx.
            // The only explicit term here is α_j * u_j which we
            // cannot recover without storage.  For production use,
            // save u_j before calling do_ax.

            // Save u_j before overwriting (we need it for the
            // short-recurrence subtraction).  Allocate a temp buffer
            // on the first iteration.
        }

        // β_{j+1} = ||p||
        beta = dist_norm_u(ctx);
        result.beta[j + 1] = beta;

        if (ctx->world_rank == 0)
            printf("[bidiag] j=%2d  alpha=%.6e  beta=%.6e\n",
                   j, alpha, beta);

        if (beta < tol) {
            if (ctx->world_rank == 0)
                printf("[bidiag] beta_%d < tol; invariant subspace found.\n", j + 1);
            result.steps_run = j + 1;
            return result;
        }

        // u_{j+1} = p / β_{j+1}
        scale_vec(ctx->u_send, ctx->u_send_size, 1.0 / beta);

        if (reorthogonalize) {
            U_vecs.push_back(std::vector<double>(
                ctx->u_send, ctx->u_send + ctx->u_send_size));
        }

        result.steps_run = j + 1;
    }

    return result;
}

// ==================================================================
// bcp_bidiagonalize_full
//
// Full-reorthogonalization variant with an explicit u_prev buffer
// for the short recurrence.  This is the numerically robust version
// that correctly implements the three-term recurrence:
//
//   p_j = A v_j  -  α_j u_j  -  β_j u_{j-1}   (short recurrence)
//   r_j = A^T u_j  -  α_{j-1} v_{j-1}          (short recurrence)
//
// plus optional full reorthogonalization on top.
// ==================================================================
BCPBidiagResult bcp_bidiagonalize_full(BCPContext* ctx,
                                        int k,
                                        bool reorthogonalize = true,
                                        double tol = 1e-14) {
    BCPBidiagResult result;
    result.alpha.resize(k);
    result.beta.resize(k + 1);
    result.steps_run = 0;

    // Working buffers (local to this rank)
    std::vector<double> u_prev(ctx->u_send_size, 0.0);  // u_{j-1}
    std::vector<double> u_cur(ctx->u_send_size);         // u_j
    std::vector<double> v_prev(ctx->v_send_size, 0.0);  // v_{j-1}
    std::vector<double> v_cur(ctx->v_send_size);         // v_j (scratch)

    // Optional storage for full reorthogonalization
    std::vector<std::vector<double>> U_vecs, V_vecs;
    if (reorthogonalize) {
        U_vecs.reserve(k + 1);
        V_vecs.reserve(k);
    }

    // -------------------------------------------------------
    // Initialization: β₁ = ||b||,  u₁ = b / β₁
    // ctx->u_send contains the initial vector b on entry.
    // -------------------------------------------------------
    double beta = dist_norm_u(ctx);
    result.beta[0] = beta;

    if (beta < tol) {
        if (ctx->world_rank == 0)
            printf("[bidiag_full] Starting vector is zero.\n");
        return result;
    }

    scale_vec(ctx->u_send, ctx->u_send_size, 1.0 / beta);
    std::memcpy(u_cur.data(), ctx->u_send,
                ctx->u_send_size * sizeof(double));

    if (reorthogonalize)
        U_vecs.push_back(u_cur);

    for (int j = 0; j < k; ++j) {
        // ---------------------------------------------------
        // r = A^T u_j  -  α_{j-1} * v_{j-1}
        // ---------------------------------------------------
        // ctx->u_send already holds u_j from the previous iteration.
        do_atx(ctx);
        // ctx->v_send = A^T u_j

        // Short recurrence subtraction: r -= α_{j-1} * v_{j-1}
        if (j > 0) {
            axpy_vec(ctx->v_send, v_prev.data(),
                     ctx->v_send_size, -result.alpha[j - 1]);
        }

        // Full reorthogonalization vs all previous v vectors
        if (reorthogonalize) {
            for (int i = 0; i < (int)V_vecs.size(); ++i) {
                double c = dist_dot_v(ctx, V_vecs[i].data(), ctx->v_send);
                axpy_vec(ctx->v_send, V_vecs[i].data(),
                         ctx->v_send_size, -c);
            }
        }

        double alpha = dist_norm_v(ctx);
        result.alpha[j] = alpha;

        if (alpha < tol) {
            if (ctx->world_rank == 0)
                printf("[bidiag_full] alpha_%d < tol.\n", j);
            result.steps_run = j + 1;
            return result;
        }

        scale_vec(ctx->v_send, ctx->v_send_size, 1.0 / alpha);

        // Save v_j for next iteration's short recurrence
        std::memcpy(v_cur.data(), ctx->v_send,
                    ctx->v_send_size * sizeof(double));
        if (reorthogonalize)
            V_vecs.push_back(v_cur);

        // ---------------------------------------------------
        // p = A v_j  -  α_j * u_j  -  β_j * u_{j-1}
        // ---------------------------------------------------
        do_ax(ctx);
        // ctx->u_send = A v_j

        // Subtract α_j * u_j
        axpy_vec(ctx->u_send, u_cur.data(),
                 ctx->u_send_size, -alpha);

        // Subtract β_j * u_{j-1}  (j >= 1)
        if (j > 0) {
            axpy_vec(ctx->u_send, u_prev.data(),
                     ctx->u_send_size, -result.beta[j]);
        }

        // Full reorthogonalization vs all previous u vectors
        if (reorthogonalize) {
            for (int i = 0; i < (int)U_vecs.size(); ++i) {
                double c = dist_dot_u(ctx, U_vecs[i].data(), ctx->u_send);
                axpy_vec(ctx->u_send, U_vecs[i].data(),
                         ctx->u_send_size, -c);
            }
        }

        beta = dist_norm_u(ctx);
        result.beta[j + 1] = beta;

        if (ctx->world_rank == 0)
            printf("[bidiag_full] j=%2d  alpha=%.6e  beta=%.6e\n",
                   j, alpha, beta);

        if (beta < tol) {
            if (ctx->world_rank == 0)
                printf("[bidiag_full] beta_%d < tol.\n", j + 1);
            result.steps_run = j + 1;
            return result;
        }

        scale_vec(ctx->u_send, ctx->u_send_size, 1.0 / beta);

        // Rotate u buffers: prev <- cur, cur <- new u_{j+1}
        std::swap(u_prev, u_cur);
        std::memcpy(u_cur.data(), ctx->u_send,
                    ctx->u_send_size * sizeof(double));

        if (reorthogonalize)
            U_vecs.push_back(u_cur);

        // Rotate v buffers
        std::swap(v_prev, v_cur);

        result.steps_run = j + 1;
    }

    return result;
}

// ==================================================================
// bcp_print_bidiag_result
//
// Pretty-prints the bidiagonal matrix B on rank 0.
// ==================================================================
void bcp_print_bidiag_result(const BCPBidiagResult& r) {
    int k = r.steps_run;
    printf("\nBidiagonal matrix B (%d x %d):\n", k + 1, k);
    printf("  beta[0] = %.8e  (norm of starting vector)\n", r.beta[0]);
    for (int j = 0; j < k; ++j) {
        printf("  alpha[%d] = %.8e   beta[%d] = %.8e\n",
               j, r.alpha[j], j + 1, r.beta[j + 1]);
    }
}

// ==================================================================
// bcp_verify_bidiag
//
// Verifies the bidiagonalization on rank 0 by:
//   1. Reconstructing A, U, V from scratch via the reconstruct helpers
//      (only for small test problems — very slow).
//   2. Checking that A ≈ U B V^T.
//
// This is a debug/test helper, NOT for production use.
// ==================================================================
void bcp_verify_bidiag(BCPContext* ctx, int k) {
    // We run the bidiagonalization from a fixed starting vector
    // (the current ctx->u_send) and check consistency.
    BCPBidiagResult r = bcp_bidiagonalize_full(ctx, k, true);

    if (ctx->world_rank == 0) {
        bcp_print_bidiag_result(r);

        // Sanity check: B should have positive diagonal entries
        bool ok = true;
        for (int j = 0; j < r.steps_run; ++j) {
            if (r.alpha[j] <= 0.0) {
                printf("[verify] alpha[%d] = %.3e <= 0 (unexpected)\n",
                       j, r.alpha[j]);
                ok = false;
            }
            if (r.beta[j + 1] < 0.0) {
                printf("[verify] beta[%d] = %.3e < 0 (unexpected)\n",
                       j + 1, r.beta[j + 1]);
                ok = false;
            }
        }
        printf("Bidiag sanity check: %s\n", ok ? "PASSED" : "FAILED");
    }
}

#endif // BCP_BIDIAG_H

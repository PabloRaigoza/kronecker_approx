#ifndef BIDIAG_H
#define BIDIAG_H

#include <mpi.h>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "common.h"

// ============================================================
// Distributed Golub-Kahan (Lanczos) bidiagonalization of Ã,
// generic over the partitioning scheme (Golub & Van Loan, 4th ed.,
// Alg. 10.4.1 -- same recurrence as serial/serial.py):
//
//   v_1 = random unit vector (seeded)
//   for j = 1..k:
//       u_j     = (Ã v_j   - beta_{j-1} u_{j-1}) / alpha_j
//       v_{j+1} = (Ã^T u_j - alpha_j    v_j    ) / beta_j
//
// giving Ã V_k = U_k B_k with B_k upper bidiagonal
// (diag alpha_1..alpha_k, superdiag beta_1..beta_{k-1}).
//
// A scheme plugs in by providing, for its context type Ctx:
//   void      do_ax (Ctx*, KernelTimers*, bool sync)  // v slice -> u slice
//   void      do_atx(Ctx*, KernelTimers*, bool sync)  // u slice -> v slice
//   OwnedVecs owned_vecs(Ctx*)
// The u / v slices are the buffers do_ax / do_atx read and write in
// place, so no data is moved between kernel calls.
// ============================================================

struct BidiagTimers {
    KernelTimers ax, atx;     // time inside the matvec kernels, by phase
    double reductions = 0.0;  // MPI_Allreduce for norms / reorth coefficients
    double vector_ops = 0.0;  // local BLAS-1/2 work and copies
    double barrier = 0.0;     // sync barriers before the Allreduces (kernel barriers are in ax / atx)
    double total = 0.0;       // wall time of the whole bidiagonalization
    double kernel() const { return ax.total() + atx.total(); }
    // All time spent waiting in sync barriers, i.e. for slower ranks
    double barrier_total() const { return barrier + ax.barrier + atx.barrier; }
};

struct BidiagResult {
    std::vector<double> alpha;  // alpha_1..alpha_steps
    std::vector<double> beta;   // beta_1..beta_steps (beta_steps is the residual coupling)
    int steps = 0;
    BidiagTimers timers;
};

static double gk_dot(const double* x, const double* y, int n, MPI_Comm comm, BidiagTimers* T, bool sync) {
    double t0 = MPI_Wtime();
    double local = cblas_ddot(n, x, 1, y, 1);
    T->vector_ops += MPI_Wtime() - t0;
    double t1 = phase_start(&T->barrier, sync, comm);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    T->reductions += MPI_Wtime() - t1;
    return global;
}

// Classical Gram-Schmidt applied twice (CGS2): orthogonalize x against
// the m columns of Q (n x m, column-major). One Allreduce of length m
// per pass instead of m separate dot products.
static void gk_reorthogonalize(double* x, const double* Q, int n, int m, double* h,
                               MPI_Comm comm, BidiagTimers* T, bool sync) {
    if (m == 0) return;
    for (int pass = 0; pass < 2; pass++) {
        double t0 = MPI_Wtime();
        // BLAS quick-returns without touching h when n == 0
        std::fill(h, h + m, 0.0);
        cblas_dgemv(CblasColMajor, CblasTrans, n, m, 1.0, Q, std::max(n, 1), x, 1, 0.0, h, 1);
        T->vector_ops += MPI_Wtime() - t0;
        double t1 = phase_start(&T->barrier, sync, comm);
        MPI_Allreduce(MPI_IN_PLACE, h, m, MPI_DOUBLE, MPI_SUM, comm);
        double t2 = MPI_Wtime();
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, m, -1.0, Q, std::max(n, 1), h, 1, 1.0, x, 1);
        T->vector_ops += MPI_Wtime() - t2;
        T->reductions += t2 - t1;
    }
}

// k            - number of steps (clamped to min(m1*n1, m2*n2))
// seed         - seeds the starting vector v_1 (by global index, so it is
//                identical across partitioning schemes)
// reorth       - full reorthogonalization (stores all U_k, V_k slices)
// sync         - barrier before every communication phase (kernel AllGather /
//                dgemv / ReduceScatter and each Allreduce), like the Ax/ATx
//                benchmarks. Phase timings then exclude waiting for slower
//                ranks; that wait is reported separately (barrier_total).
//                Without sync, the wait lands in whichever collective comes
//                next (often the norm Allreduce) and total is the
//                barrier-free wall time a production solver would see.
// tol          - stop early when alpha_j or beta_j <= tol * alpha_1
template <class Ctx>
BidiagResult bidiagonalize(Ctx* ctx, int k, uint64_t seed, bool reorth, bool sync = true, double tol = 1e-14) {
    OwnedVecs o = owned_vecs(ctx);
    BidiagResult r;
    BidiagTimers* T = &r.timers;

    int nu_global = 0, nv_global = 0;
    MPI_Allreduce(&o.nu, &nu_global, 1, MPI_INT, MPI_SUM, o.comm);
    MPI_Allreduce(&o.nv, &nv_global, 1, MPI_INT, MPI_SUM, o.comm);
    k = std::min(k, std::min(nu_global, nv_global));

    // Without reorth only u_{j-1} and v_j are kept (one slot each)
    int slots = reorth ? k : 1;
    std::vector<double> U((size_t)o.nu * slots), V((size_t)o.nv * slots), h(k);

    MPI_Barrier(o.comm);
    double t_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    fill_seeded_vec(o.v, o.v_gidx, o.nv, seed, STREAM_V_START);
    T->vector_ops += MPI_Wtime() - t0;
    double nrm = std::sqrt(gk_dot(o.v, o.v, o.nv, o.comm, T, sync));
    t0 = MPI_Wtime();
    cblas_dscal(o.nv, 1.0 / nrm, o.v, 1);
    T->vector_ops += MPI_Wtime() - t0;

    double beta_prev = 0.0;
    for (int j = 0; j < k; j++) {
        double* u_prev = reorth ? (j > 0 ? &U[(size_t)(j - 1) * o.nu] : nullptr) : U.data();
        double* u_cur  = reorth ? &U[(size_t)j * o.nu] : U.data();
        double* v_cur  = reorth ? &V[(size_t)j * o.nv] : V.data();

        // Keep v_j: do_atx below overwrites o.v
        t0 = MPI_Wtime();
        if (o.nv) memcpy(v_cur, o.v, (size_t)o.nv * sizeof(double));
        T->vector_ops += MPI_Wtime() - t0;

        // u_j = Ã v_j - beta_{j-1} u_{j-1}
        do_ax(ctx, &T->ax, sync);
        t0 = MPI_Wtime();
        if (j > 0) cblas_daxpy(o.nu, -beta_prev, u_prev, 1, o.u, 1);
        T->vector_ops += MPI_Wtime() - t0;
        if (reorth) gk_reorthogonalize(o.u, U.data(), o.nu, j, h.data(), o.comm, T, sync);

        double alpha = std::sqrt(gk_dot(o.u, o.u, o.nu, o.comm, T, sync));
        r.alpha.push_back(alpha);
        r.steps = j + 1;
        if (alpha <= tol * r.alpha[0] || alpha == 0.0) break;

        t0 = MPI_Wtime();
        cblas_dscal(o.nu, 1.0 / alpha, o.u, 1);
        if (o.nu) memcpy(u_cur, o.u, (size_t)o.nu * sizeof(double));
        T->vector_ops += MPI_Wtime() - t0;

        // v_{j+1} = Ã^T u_j - alpha_j v_j
        do_atx(ctx, &T->atx, sync);
        t0 = MPI_Wtime();
        cblas_daxpy(o.nv, -alpha, v_cur, 1, o.v, 1);
        T->vector_ops += MPI_Wtime() - t0;
        if (reorth) gk_reorthogonalize(o.v, V.data(), o.nv, j + 1, h.data(), o.comm, T, sync);

        double beta = std::sqrt(gk_dot(o.v, o.v, o.nv, o.comm, T, sync));
        r.beta.push_back(beta);
        if (beta <= tol * r.alpha[0]) break;

        t0 = MPI_Wtime();
        cblas_dscal(o.nv, 1.0 / beta, o.v, 1);
        T->vector_ops += MPI_Wtime() - t0;
        beta_prev = beta;
    }

    T->total = MPI_Wtime() - t_start;
    return r;
}

// Largest singular value of the k x k upper bidiagonal B_k (the Ritz
// approximation of sigma_1(Ã)), via power iteration on B^T B. Serial
// and O(k) per iteration, so it is negligible.
static double bidiag_sigma_max(const std::vector<double>& alpha, const std::vector<double>& beta) {
    int k = (int)alpha.size();
    if (k == 0) return 0.0;
    std::vector<double> x(k, 1.0 / std::sqrt((double)k)), y(k), z(k);
    double sigma = 0.0;
    for (int it = 0; it < 100000; it++) {
        for (int i = 0; i < k; i++)   // y = B x
            y[i] = alpha[i] * x[i] + (i + 1 < k ? beta[i] * x[i + 1] : 0.0);
        for (int i = 0; i < k; i++)   // z = B^T y
            z[i] = alpha[i] * y[i] + (i > 0 ? beta[i - 1] * y[i - 1] : 0.0);
        double nz = std::sqrt(cblas_ddot(k, z.data(), 1, z.data(), 1));
        double new_sigma = std::sqrt(nz);
        for (int i = 0; i < k; i++) x[i] = z[i] / nz;
        if (std::fabs(new_sigma - sigma) <= 1e-15 * new_sigma) { sigma = new_sigma; break; }
        sigma = new_sigma;
    }
    return sigma;
}

// Accumulate timers across trials (results are identical across trials)
static void bidiag_timers_add(BidiagTimers* acc, const BidiagTimers& t) {
    acc->ax.all_gather += t.ax.all_gather;
    acc->ax.computation += t.ax.computation;
    acc->ax.reduce_scatter += t.ax.reduce_scatter;
    acc->atx.all_gather += t.atx.all_gather;
    acc->atx.computation += t.atx.computation;
    acc->atx.reduce_scatter += t.atx.reduce_scatter;
    acc->ax.barrier += t.ax.barrier;
    acc->atx.barrier += t.atx.barrier;
    acc->barrier += t.barrier;
    acc->reductions += t.reductions;
    acc->vector_ops += t.vector_ops;
    acc->total += t.total;
}

// Prints (rank 0) timings averaged over num_trials, reduced across ranks.
// The kernel fraction is computed per rank, then reported min/mean/max.
static void bidiag_print_report(const BidiagResult& r, const BidiagTimers& acc, int num_trials, bool sync) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    enum { TOTAL, KERNEL, AX, ATX, AX_AG, AX_COMP, AX_RS, ATX_AG, ATX_COMP, ATX_RS, RED, VEC, BAR, OTHER, FRAC, N };
    double local[N];
    local[TOTAL]    = acc.total / num_trials;
    local[KERNEL]   = acc.kernel() / num_trials;
    local[AX]       = acc.ax.total() / num_trials;
    local[ATX]      = acc.atx.total() / num_trials;
    local[AX_AG]    = acc.ax.all_gather / num_trials;
    local[AX_COMP]  = acc.ax.computation / num_trials;
    local[AX_RS]    = acc.ax.reduce_scatter / num_trials;
    local[ATX_AG]   = acc.atx.all_gather / num_trials;
    local[ATX_COMP] = acc.atx.computation / num_trials;
    local[ATX_RS]   = acc.atx.reduce_scatter / num_trials;
    local[RED]      = acc.reductions / num_trials;
    local[VEC]      = acc.vector_ops / num_trials;
    local[BAR]      = acc.barrier_total() / num_trials;
    local[OTHER]    = local[TOTAL] - local[KERNEL];
    local[FRAC]     = local[TOTAL] > 0 ? local[KERNEL] / local[TOTAL] : 0.0;

    double mx[N], mn[N], sum[N];
    MPI_Reduce(local, mx, N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, mn, N, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, sum, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank != 0) return;

    double alpha_sum = 0.0, beta_sum = 0.0;
    for (double a : r.alpha) alpha_sum += a;
    for (double b : r.beta) beta_sum += b;
    printf("Bidiag Steps: %d | Sigma1: %.15e | Alpha Sum: %.15e | Beta Sum: %.15e | Sync: %d\n",
           r.steps, bidiag_sigma_max(r.alpha, r.beta), alpha_sum, beta_sum, (int)sync);
    // Max Other = Max of (total - kernel) and includes barrier waits
    printf("Bidiag Max Total: %.6f | Max Kernel: %.6f | Max Other: %.6f | Max Reductions: %.6f | Max Vector Ops: %.6f"
           " | Max Barrier: %.6f\n",
           mx[TOTAL], mx[KERNEL], mx[OTHER], mx[RED], mx[VEC], mx[BAR]);
    printf("Bidiag Kernel Fraction: Mean %.4f | Min %.4f | Max %.4f\n",
           sum[FRAC] / world_size, mn[FRAC], mx[FRAC]);
    printf("Bidiag Ax (max): All Gather: %.6f | Computation: %.6f | Reduce Scatter: %.6f | Total: %.6f\n",
           mx[AX_AG], mx[AX_COMP], mx[AX_RS], mx[AX]);
    printf("Bidiag ATx (max): All Gather: %.6f | Computation: %.6f | Reduce Scatter: %.6f | Total: %.6f\n",
           mx[ATX_AG], mx[ATX_COMP], mx[ATX_RS], mx[ATX]);
    // Means over ranks of each component. Unlike the max lines (each max can
    // come from a different rank), these add up to the mean total, so they
    // are the ones to stack in a breakdown. Barrier is the time ranks spent
    // waiting for slower ranks (0 without sync, where that wait is instead
    // folded into the next collective).
    printf("Bidiag Mean Breakdown: Total: %.6f | Ax All Gather: %.6f | Ax Computation: %.6f | Ax Reduce Scatter: %.6f"
           " | ATx All Gather: %.6f | ATx Computation: %.6f | ATx Reduce Scatter: %.6f | Reductions: %.6f | Vector Ops: %.6f"
           " | Barrier: %.6f\n",
           sum[TOTAL] / world_size, sum[AX_AG] / world_size, sum[AX_COMP] / world_size, sum[AX_RS] / world_size,
           sum[ATX_AG] / world_size, sum[ATX_COMP] / world_size, sum[ATX_RS] / world_size,
           sum[RED] / world_size, sum[VEC] / world_size, sum[BAR] / world_size);
}

#endif // BIDIAG_H

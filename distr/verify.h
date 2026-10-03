#ifndef VERIFY_H
#define VERIFY_H

#include <mpi.h>
#include <cblas.h>
#include <cmath>
#include <vector>
#include "common.h"
#include "bidiag.h"

// ============================================================
// Scheme-independent verification. Uses only a scheme's
// do_ax / do_atx / owned_vecs and the seeded global-index
// generator, so it checks each partitioning against the same
// explicitly-formed Ã on rank 0 (small problems only).
// ============================================================

// Serial "scheme": rank-local, full Ã. Lets bidiagonalize() run as
// the serial reference on rank 0 (MPI_COMM_SELF).
struct SerialContext {
    int rows, cols;
    std::vector<double> A, u, v;
    std::vector<long> u_gidx, v_gidx;
};

SerialContext serial_build(int m1, int n1, int m2, int n2, uint64_t seed) {
    SerialContext s;
    s.rows = m1 * n1;
    s.cols = m2 * n2;
    s.A.resize((size_t)s.rows * s.cols);
    s.u.resize(s.rows);
    s.v.resize(s.cols);
    s.u_gidx.resize(s.rows);
    s.v_gidx.resize(s.cols);
    for (int i = 0; i < s.rows; i++) s.u_gidx[i] = i;
    for (int i = 0; i < s.cols; i++) s.v_gidx[i] = i;
    fill_a_local(s.A.data(), seed, s.u_gidx.data(), s.rows, s.v_gidx.data(), s.cols, s.cols);
    return s;
}

void do_ax(SerialContext* s, KernelTimers* t, bool) {
    double t0 = MPI_Wtime();
    cblas_dgemv(CblasRowMajor, CblasNoTrans, s->rows, s->cols, 1.0, s->A.data(), s->cols,
                s->v.data(), 1, 0.0, s->u.data(), 1);
    t->computation += MPI_Wtime() - t0;
}

void do_atx(SerialContext* s, KernelTimers* t, bool) {
    double t0 = MPI_Wtime();
    cblas_dgemv(CblasRowMajor, CblasTrans, s->rows, s->cols, 1.0, s->A.data(), s->cols,
                s->u.data(), 1, 0.0, s->v.data(), 1);
    t->computation += MPI_Wtime() - t0;
}

OwnedVecs owned_vecs(SerialContext* s) {
    return { s->u.data(), s->rows, s->u_gidx.data(),
             s->v.data(), s->cols, s->v_gidx.data(), MPI_COMM_SELF };
}

// Gather a distributed owned slice into its global order on rank 0.
// Also checks that the slices cover every global index exactly once.
static bool gather_global(const double* x, const long* gidx, int n, int global_n,
                          std::vector<double>* out) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> counts(size), displs(size, 0);
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int total = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) { displs[i] = total; total += counts[i]; }
    }
    std::vector<double> vals(rank == 0 ? total : 0);
    std::vector<long> idx(rank == 0 ? total : 0);
    MPI_Gatherv(x, n, MPI_DOUBLE, vals.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(gidx, n, MPI_LONG, idx.data(), counts.data(), displs.data(), MPI_LONG, 0, MPI_COMM_WORLD);

    int ok = 1;
    if (rank == 0) {
        out->assign(global_n, 0.0);
        std::vector<int> hits(global_n, 0);
        if (total != global_n) ok = 0;
        for (int i = 0; i < total; i++) {
            if (idx[i] < 0 || idx[i] >= global_n) { ok = 0; continue; }
            hits[idx[i]]++;
            (*out)[idx[i]] = vals[i];
        }
        for (int i = 0; i < global_n; i++) if (hits[i] != 1) ok = 0;
        if (!ok) printf("Ownership check FAILED: owned slices do not partition the vector (%d owned, %d expected)\n", total, global_n);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok;
}

static double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        num = std::max(num, std::fabs(a[i] - b[i]));
        den = std::max(den, std::fabs(b[i]));
    }
    return den > 0 ? num / den : num;
}

// Checks one Ax and one ATx against serial dgemv on the explicit Ã.
template <class Ctx>
bool verify_kernels(Ctx* ctx, int m1, int n1, int m2, int n2, uint64_t seed, double tol = 1e-12) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int R = m1 * n1, C = m2 * n2;
    OwnedVecs o = owned_vecs(ctx);
    KernelTimers t;

    fill_seeded_vec(o.v, o.v_gidx, o.nv, seed, STREAM_VERIFY);
    std::vector<double> v_in, u_out, v_out;
    bool ok = gather_global(o.v, o.v_gidx, o.nv, C, &v_in);
    do_ax(ctx, &t, false);
    ok = gather_global(o.u, o.u_gidx, o.nu, R, &u_out) && ok;
    do_atx(ctx, &t, false);
    ok = gather_global(o.v, o.v_gidx, o.nv, C, &v_out) && ok;

    int pass = ok;
    if (rank == 0 && ok) {
        SerialContext s = serial_build(m1, n1, m2, n2, seed);
        s.v = v_in;
        do_ax(&s, &t, false);
        double err_ax = max_rel_diff(u_out, s.u);
        s.u = u_out;  // same input the distributed ATx saw
        do_atx(&s, &t, false);
        double err_atx = max_rel_diff(v_out, s.v);
        pass = err_ax <= tol && err_atx <= tol;
        printf("Verify Ax:  rel err %.3e %s\n", err_ax, err_ax <= tol ? "PASSED" : "FAILED");
        printf("Verify ATx: rel err %.3e %s\n", err_atx, err_atx <= tol ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&pass, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return pass;
}

// Bidiagonalization check:
//  1. With full reorthogonalization, alpha / beta must match the serial
//     run (same seed, same starting vector) to rounding. Without reorth
//     they cannot: once a Ritz value converges, plain Lanczos loses
//     orthogonality and amplifies reduction-order differences, so alpha /
//     beta diverge across schemes after a few steps.
//  2. sigma_1 of B_k must match power iteration on Ã^T Ã (an independent
//     check of the whole pipeline): to rounding with reorth, and to
//     plain_tol without it (loss of orthogonality limits its accuracy).
template <class Ctx>
bool verify_bidiag(Ctx* ctx, int m1, int n1, int m2, int n2, uint64_t seed, int k,
                   double tol = 1e-8, double plain_tol = 1e-6) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    BidiagResult dist_reorth = bidiagonalize(ctx, k, seed, true);
    BidiagResult dist_plain = bidiagonalize(ctx, k, seed, false);

    int pass = 1;
    if (rank == 0) {
        SerialContext s = serial_build(m1, n1, m2, n2, seed);
        BidiagResult ref = bidiagonalize(&s, k, seed, true);

        bool same_steps = dist_reorth.steps == ref.steps && dist_reorth.beta.size() == ref.beta.size();
        double err_alpha = same_steps ? max_rel_diff(dist_reorth.alpha, ref.alpha) : INFINITY;
        double err_beta = same_steps ? max_rel_diff(dist_reorth.beta, ref.beta) : INFINITY;
        bool ab_ok = err_alpha <= tol && err_beta <= tol;
        printf("Verify bidiag (k=%d, reorth, steps %d vs serial %d): alpha rel err %.3e | beta rel err %.3e %s\n",
               k, dist_reorth.steps, ref.steps, err_alpha, err_beta, ab_ok ? "PASSED" : "FAILED");

        // Independent check of sigma_1 via power iteration on Ã^T Ã
        std::vector<double> x(s.cols, 1.0);
        double sigma_pi = 0.0;
        KernelTimers t;
        for (int it = 0; it < 10000; it++) {
            double nx = std::sqrt(cblas_ddot(s.cols, x.data(), 1, x.data(), 1));
            for (double& xi : x) xi /= nx;
            s.v = x;
            do_ax(&s, &t, false);
            do_atx(&s, &t, false);
            x = s.v;
            double new_sigma = std::sqrt(std::sqrt(cblas_ddot(s.cols, x.data(), 1, x.data(), 1)));
            if (std::fabs(new_sigma - sigma_pi) <= 1e-15 * new_sigma) { sigma_pi = new_sigma; break; }
            sigma_pi = new_sigma;
        }
        double sigma_reorth = bidiag_sigma_max(dist_reorth.alpha, dist_reorth.beta);
        double sigma_plain = bidiag_sigma_max(dist_plain.alpha, dist_plain.beta);
        double err_reorth = std::fabs(sigma_reorth - sigma_pi) / sigma_pi;
        double err_plain = std::fabs(sigma_plain - sigma_pi) / sigma_pi;
        bool sigma_ok = err_reorth <= tol && err_plain <= plain_tol;
        printf("Verify sigma1: power iteration %.15e | GK reorth rel err %.3e | GK no reorth rel err %.3e (tol %.0e) %s\n",
               sigma_pi, err_reorth, err_plain, plain_tol, sigma_ok ? "PASSED" : "FAILED");

        pass = ab_ok && sigma_ok;
    }
    MPI_Bcast(&pass, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return pass;
}

#endif // VERIFY_H

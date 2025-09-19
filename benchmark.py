import numpy as np
from serial.serial import construct_A_tilde, unVec, svd_decomp
from multiprocessing import Pool, cpu_count

def gen_A_condition(m1, n1, m2, n2, cond_num):
    """ 
    Generate a matrix A of size (m1*m2, n1*n2) with a specified condition number.
    """
    m = m1 * m2
    n = n1 * n2
    min_dim = min(m, n)

    # Generate orthogonal matrices using SVD
    U_rand = np.random.randn(m, m)
    V_rand = np.random.randn(n, n)
    U, _ = np.linalg.qr(U_rand)
    V, _ = np.linalg.qr(V_rand)

    # Generate singular values between 1 and 1/cond_num
    # singular_values = np.linspace(1, 1 / cond_num, min_dim)
    # fix it so s[0]/s[1] = cond_num
    singular_values = np.linspace(1, 1 / cond_num, min_dim)
    singular_values[0] = cond_num * singular_values[1]
    # fix it so the later singular values decay are similar to s[1]
    for i in range(2, min_dim):
        singular_values[i] = singular_values[i-1] * 1
    Sig = np.zeros((m, n))
    for i in range(min_dim):
        Sig[i, i] = singular_values[i]

    # Construct A with the desired singular values
    A = U @ Sig @ V.T

    # Check: recompute SVD and validate
    U_check, s_check, Vt_check = np.linalg.svd(A, full_matrices=False)
    x = np.random.rand(n)
    Ax = A @ x
    Ax_check = U_check @ np.diag(s_check) @ (Vt_check @ x)
    assert np.allclose(Ax, Ax_check, atol=1e-8)
    assert A.shape == (m, n)
    
    return A

def gen_A_tilde_condition(m1, n1, m2, n2, cond_num):
    '''
    A_tilde is (m1*n1, m2*n2)
    '''
    A_tilde = gen_A_condition(m1=m1, m2=n1, n1=m2, n2=n2, cond_num=cond_num)
    
    A = np.zeros((m1*m2, n1*n2))
    for i in range(n1):
        for j in range(m1):
            A[j*m2:(j+1)*m2, i*n2:(i+1)*n2] = unVec(A_tilde[i*m1 + j], m2, n2)

    assert np.all(A_tilde == construct_A_tilde(A, m1, n1, m2, n2))
    return A

# gen_A_tilde_condition(4,5,6,7,10)

def benchmark_svd_approach(cond_num):
    l, u = 10, 50
    m1, n1 = np.random.randint(l, u), np.random.randint(l, u)
    m2, n2 = np.random.randint(l, u), np.random.randint(l, u)
    print(f"Benchmarking SVD approach for condition number {cond_num} with matrix size ({m1*m2}, {n1*n2})")
    _A = gen_A_tilde_condition(m1, n1, m2, n2, cond_num)
    
    stats = svd_decomp(_A, m1, n1, m2, n2)

    with open('svd_benchmark_results.txt', 'a') as f:
        f.write(f"{m1},{n1},{m2},{n2},{cond_num},{stats['num_iters']},{stats['converged']},{stats['optimal']},{stats['iters_for_ritz_conv']},{stats['iters_for_res_conv']},{stats['condition_number_A_local']},{stats['condition_number_A_global']},{stats['condition_number_A_tilde_local']},{stats['condition_number_A_tilde_global']},{stats['rel_error']}\n")

    # Header:
    # m1,n1,m2,n2,cond_num,num_iters,converged,optimal,iters_for_ritz_conv,iters_for_res_conv,condition_number_A_local,condition_number_A_global,condition_number_A_tilde_local,condition_number_A_tilde_global,rel_error

if __name__ == "__main__":
    condition_numbers = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # with Pool(processes=cpu_count()) as pool:
    with Pool(processes=min(len(condition_numbers), cpu_count())) as pool:
        pool.map(benchmark_svd_approach, condition_numbers)

    # for cond_num in condition_numbers:
    #     benchmark_svd_approach(cond_num)
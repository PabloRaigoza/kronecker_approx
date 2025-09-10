def kron_decomp(A, m1, n1, m2, n2):
    '''
    A: matrix of size (m1*m2, n1*n2)
    return: list of tuples (U, V) where U is of size (m1, n1) and V is of size (m2, n2)
    '''
    A_tilde = np.zeros((m1*n1, m2*n2))
    for j in range(n1):
        for i in range(m1):
            A_tilde[j*m1 + i] = vec(Aij(A, i, j))
    
    U, S, VT = np.linalg.svd(A_tilde)
    rank = np.sum(S > 1e-10)
    U = U[:, :rank]
    S = S[:rank]
    VT = VT[:rank, :]

    kron_factors = []
    for k in range(rank):
        u_k = U[:, k].reshape((m1, n1))
        v_k = VT[k, :].reshape((m2, n2))
        kron_factors.append((u_k * np.sqrt(S[k]), v_k * np.sqrt(S[k])))
    
    return kron_factors
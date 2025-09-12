import numpy as np

def Aij(_A, i, j, m1, n1, m2, n2, is_tilde=False):
    '''
    i: integer from [0, m1]
    j: integer from [0, n1]
    '''
    if is_tilde: 
        return _A[i*m1:(i+1)*m1, j*m2:(j+1)*m2]
    return _A[i*m2:(i+1)*m2, j*n2:(j+1)*n2]

def hatAij(_A, i, j, m1, n1, m2, n2):
    '''
    i: integer from [0, m2]
    j: integer from [0, n2]
    '''
    m, n = _A.shape 
    return _A[i:m:m2, j:n:n2]

def vec(X):
    '''
    Stacks the columns of X into a single vector.
    '''
    return X.T.flatten()

def unVec(x, m, n):
    '''
    x: vector of size m*n
    return matrix of size m*n
    '''
    return x.reshape((n, m)).T

def Ax(_A, x, m1, n1, m2, n2):
    '''
    x: vector of size m2*n2
    return vector of size m1*n1
    '''
    y = np.zeros(m1*n1)
    for j in range(n1):
        for i in range(m1):
            y[j*m1 + i] = np.dot(vec(Aij(_A, i, j, m1, n1, m2, n2)), x)
    return y

def ATx(_A, x, m1, n1, m2, n2):
    '''
    x: vector of size m1*n1
    return vector of size m2*n2
    '''
    y = np.zeros(m2*n2)
    for j in range(n1):
        for i in range(m1):
            y += x[j*m1 + i] * vec(Aij(_A, i, j, m1, n1, m2, n2))
    return y

def construct_A_tilde(_A, m1, n1, m2, n2):
    A_tilde = np.zeros((m1*n1, m2*n2))
    for j in range(n1):
        for i in range(m1):
            A_tilde[j*m1 + i] = vec(Aij(_A, i, j, m1, n1, m2, n2))
    return A_tilde

def kron_decomp(_A, m1, n1, m2, n2):
    '''
    A: matrix of size (m1*m2, n1*n2)
    return B, C where B is of size (m1, n1) and C is of size (m2, n2)
    maxit looks like it should be n1*n2
    https://math.ecnu.edu.cn/~jypan/Teaching/books/2013%20Matrix%20Computations%204th.pdf
    p597
    '''
    # maxit = n1*n2
    maxit = max(2000, n1*n2)
    V = np.zeros((m2*n2, maxit))
    U = np.zeros((m1*n1, maxit))
    Beta = np.zeros(maxit)
    Alpha = np.zeros(maxit)
    P = np.zeros((m2*n2, maxit))
    R = np.zeros((m1*n1, maxit))
    
    V[:, 1] = np.random.rand(m2*n2)
    V[:, 1] = V[:, 1] / np.linalg.norm(V[:, 1])
    P[:, 0] = V[:, 1]
    Beta[0] = 1
    j = 0

    while not np.isclose(Beta[j], 0, atol=1e-2):
        V[:, j+1] = P[:, j] / Beta[j]
        j += 1
        R[:, j] = Ax(_A, V[:, j], m1, n1, m2, n2) - Beta[j-1] * U[:, j-1]
        # print(f'Ax {Ax(V[:, j])}')
        Alpha[j] = np.linalg.norm(R[:, j])
        if Alpha[j] == 0:
            print('Alpha is zero')
            break
        U[:, j] = R[:, j] / Alpha[j]
        P[:, j] = ATx(_A, U[:, j], m1, n1, m2, n2) - Alpha[j] * V[:, j]
        Beta[j] = np.linalg.norm(P[:, j])

        if j >= maxit - 1:
            if j > n1*n2:
                print('Exceeding n1*n2 iterations')
            print('Breaking early')
            break
    print(f'Beta[{j}]: {Beta[j]} | min(Beta): {np.min(Beta[1:j+1])} @ {np.argmin(Beta[1:j+1])+1}')
    j = np.argmin(Beta[1:j+1])+1

    U = U[:, 1:j+1]
    V = V[:, 1:j+1]
    Alpha = Alpha[1:j+1]
    Beta = Beta[1:j]

    k = j
    B = np.zeros((k, k))
    for i in range(k):
        B[i, i] = Alpha[i]
        if i < k - 1:
            B[i, i+1] = Beta[i]

    Ub, s, Vbt = np.linalg.svd(B)

    B = s[0] * unVec(U @ Ub[:, 0], m1, n1)
    C = unVec(V @ Vbt.T[:, 0], m2, n2)

    return B, C

def als_decomp(_A, m1, n1, m2, n2):
    '''
    A: matrix of size (m1*m2, n1*n2)
    return B, C where B is of size (m1, n1) and C is of size (m2, n2)
    '''
    # Initialize C
    B = np.random.rand(m1, n1)
    C = np.random.rand(m2, n2)
    for _ in range(50):
        gamma = np.sum(C * C)
        for i in range(m1):
            for j in range(n1):
                B[i, j] = np.trace(C.T @ Aij(_A, i, j, m1, n1, m2, n2)) / gamma
    
        beta = np.sum(B * B)
        for i in range(m2):
            for j in range(n2):
                C[i, j] = np.trace(B.T @ hatAij(_A, i, j, m1, n1, m2, n2)) / beta
    return B, C

def test_Ax(_A, m1, n1, m2, n2, A_tilde=None):
    print('======== Testing Ax ========')
    if A_tilde is None: A_tilde = construct_A_tilde(_A, m1, n1, m2, n2)
    x = np.random.rand(m2*n2)
    y1 = Ax(_A, x, m1, n1, m2, n2)
    y2 = A_tilde @ x
    print(f"Ax == A_tilde @ x: {np.allclose(y1, y2)}")

def test_ATx(_A, m1, n1, m2, n2, A_tilde=None):
    print('======== Testing ATx ========')
    if A_tilde is None: A_tilde = construct_A_tilde(_A, m1, n1, m2, n2)
    x = np.random.rand(m1*n1)
    y1 = ATx(_A, x, m1, n1, m2, n2)
    y2 = A_tilde.T @ x
    print(f"ATx == A_tilde.T @ y2: {np.allclose(y1, y2)}")

def reconstruct_test(_A, m1, n1, m2, n2, A_tilde=None):
    print('======== Testing Reconstruction ========')
    if A_tilde is None: A_tilde = construct_A_tilde(_A, m1, n1, m2, n2)
    A_reconstructed = np.zeros((m1*m2, n1*n2))
    for i in range(n1):
        for j in range(n2):
            A_reconstructed[:, i*n2 + j] = vec(Aij(A_tilde, i, j, m1, n1, m2, n2, True).T)
    print(f"A == A_reconstructed: {np.allclose(_A, A_reconstructed)}")

def test_kron_decomp(_A, m1, n1, m2, n2, A_tilde=None):
    print('======== Testing SVD Decomposition ========')
    if A_tilde is None: A_tilde = construct_A_tilde(_A, m1, n1, m2, n2)
    propB, propC = kron_decomp(_A, m1, n1, m2, n2)
    optU, opts, optVt = np.linalg.svd(A_tilde)
    optB = opts[0] * unVec(optU[:, 0], m1, n1)
    optC = unVec(optVt.T[:, 0], m2, n2)
    comp_to_optimal = np.linalg.norm(np.kron(propB, propC) - np.kron(optB, optC))
    print(f"||A - propB (x) propC||_F = {np.linalg.norm(_A - np.kron(propB, propC))}")
    print(f"||propB (x) propC - optB (x) optC||_F = {comp_to_optimal} | Is optimal? {np.isclose(comp_to_optimal, 0)}")

def test_als_decomp(_A, m1, n1, m2, n2):
    print('======== Testing ALS Decomposition ========')
    propB, propC = als_decomp(_A, m1, n1, m2, n2)
    A_tilde = construct_A_tilde(_A, m1, n1, m2, n2)
    optU, opts, optVt = np.linalg.svd(A_tilde)
    optB = opts[0] * unVec(optU[:, 0], m1, n1)
    optC = unVec(optVt.T[:, 0], m2, n2)
    comp_to_optimal = np.linalg.norm(np.kron(propB, propC) - np.kron(optB, optC))
    print(f"||A - propB (x) propC||_F = {np.linalg.norm(_A - np.kron(propB, propC))}")
    print(f"||propB (x) propC - optB (x) optC||_F = {comp_to_optimal} | Is optimal? {np.isclose(comp_to_optimal, 0)}")

if __name__ == "__main__":
    l, u = 2, 5
    m1, n1 = (np.random.randint(l, u), np.random.randint(l, u))
    m2, n2 = (np.random.randint(l, u), np.random.randint(l, u))
    A = np.random.rand(m1*m2, n1*n2)

    test_Ax(A, m1, n1, m2, n2)
    test_ATx(A, m1, n1, m2, n2)
    reconstruct_test(A, m1, n1, m2, n2)
    test_kron_decomp(A, m1, n1, m2, n2)
    test_als_decomp(A, m1, n1, m2, n2)
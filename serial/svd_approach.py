from scipy.linalg import svd
import numpy as np

m1, n1 = (4, 5)
m2, n2 = (3, 2)

# A = np.arange(m1*m2*n1*n2).reshape((m1*m2, n1*n2))
A = np.random.rand(m1*m2, n1*n2)

def Aij(_A, i, j, is_tilde=False):
    '''
    i: integer from [0, m1]
    j: integer from [0, n1]
    '''
    if is_tilde: 
        return _A[i*m1:(i+1)*m1, j*m2:(j+1)*m2]
    return _A[i*m2:(i+1)*m2, j*n2:(j+1)*n2]

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

def Ax(x):
    '''
    x: vector of size m2*n2
    return vector of size m1*n1
    '''
    y = np.zeros(m1*n1)
    for j in range(n1):
        for i in range(m1):
            y[j*m1 + i] = np.dot(vec(Aij(A, i, j)), x)
    return y

def ATx(x):
    '''
    x: vector of size m1*n1
    return vector of size m2*n2
    '''
    y = np.zeros(m2*n2)
    for j in range(n1):
        for i in range(m1):
            y += x[j*m1 + i] * vec(Aij(A, i, j))
    return y

def reconstruct_test():
    A_reconstructed = np.zeros((m1*m2, n1*n2))
    for i in range(n1):
        for j in range(n2):
            print(i*n2 + j)
            A_reconstructed[:, i*n2 + j] = vec(Aij(A_tilde, i, j, True).T)
    print(f"A_reconstructed:\n{A_reconstructed}\n")
    print(f"A == A_reconstructed: {np.allclose(A, A_reconstructed)}\n")

A_tilde = np.zeros((m1*n1, m2*n2))
for j in range(n1):
    for i in range(m1):
        A_tilde[j*m1 + i] = vec(Aij(A, i, j))

        

x = np.arange(m2*n2)
y1 = Ax(x)
y2 = A_tilde @ x
print(f"Ax == A_tilde @ x: {np.allclose(y1, y2)}")

y1 = ATx(y1)
y2 = A_tilde.T @ y2
print(f"ATx == A_tilde.T @ y2: {np.allclose(y1, y2)}\n")


def kron_decomp(A, m1, n1, m2, n2):
    # maxit = 100
    # maxit = n1 * n2
    maxit = 100
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

    while not np.isclose(Beta[j], 0):
        V[:, j+1] = P[:, j] / Beta[j]
        j += 1
        R[:, j] = Ax(V[:, j]) - Beta[j-1] * U[:, j-1]
        # print(f'Ax {Ax(V[:, j])}')
        Alpha[j] = np.linalg.norm(R[:, j])
        if Alpha[j] == 0:
            print('Alpha is zero')
            break
        U[:, j] = R[:, j] / Alpha[j]
        P[:, j] = ATx(U[:, j]) - Alpha[j] * V[:, j]
        Beta[j] = np.linalg.norm(P[:, j])

        if j >= maxit - 1:
            print('I dont think this should happen')
            break
        # print(Beta[j])
    print(f'Beta[{j}]: {Beta[j]}')

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

    Ub, s, Vbt = svd(B)

    B = s[0] * unVec(U @ Ub[:, 0], m1, n1)
    C = unVec(V @ Vbt.T[:, 0], m2, n2)

    optU, opts, optVt = svd(A_tilde)
    optB = opts[0] * unVec(optU[:, 0], m1, n1)
    optC = unVec(optVt.T[:, 0], m2, n2)
    
    comp_to_optimal = np.linalg.norm(np.kron(optB, optC) - np.kron(B, C))
    print(f"||A - B (x) C||_F = {np.linalg.norm(A - np.kron(B, C))}")
    print(f"||A - optB (x) optC||_F = {np.linalg.norm(A - np.kron(optB, optC))}")
    print(f"||optB (x) optC - B (x) C||_F = {comp_to_optimal} | Is optimal? {np.isclose(comp_to_optimal, 0)}")
    return B, C
kron_decomp(A, m1, n1, m2, n2)
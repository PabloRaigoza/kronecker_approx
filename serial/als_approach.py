import numpy as np
from svd_approach import kron_decomp

m1, n1 = (4, 5)
m2, n2 = (3, 2)

def Aij(_A, i, j):
    '''
    i: integer from [0, m1]
    j: integer from [0, n1]
    '''
    return _A[i*m2:(i+1)*m2, j*n2:(j+1)*n2]

def hatAij(_A, i, j):
    '''
    i: integer from [0, m2]
    j: integer from [0, n2]
    '''
    m, n = _A.shape 
    return _A[i:m:m2, j:n:n2]

def als_decomp(A, m1, n1, m2, n2):
    '''
    A: matrix of size (m1*m2, n1*n2)
    return B, C where B is of size (m1, n1) and C is of size (m2, n2)
    '''
    # Initialize C
    B = np.random.rand(m1, n1)
    C = np.random.rand(m2, n2)
    print(C[0,0])
    for _ in range(50000):
        # gamma = np.trace(C.T @ C)
        gamma = np.sum(C * C)
        for i in range(m1):
            for j in range(n1):
                B[i, j] = np.trace(C.T @ Aij(A, i, j)) / gamma
    
        # beta = np.trace(B.T @ B)
        beta = np.sum(B * B)
        for i in range(m2):
            for j in range(n2):
                C[i, j] = np.trace(B.T @ hatAij(A, i, j)) / beta

    optB, optC = kron_decomp(A, m1, n1, m2, n2)
    comp_to_optimal = np.linalg.norm(np.kron(optB, optC) - np.kron(B, C))
    print(f'Reconstruction: {np.kron(B, C)}')
    print(f"||A - B (x) C||_F = {np.linalg.norm(A - np.kron(B, C))}")
    print(f"||A - optB (x) optC||_F = {np.linalg.norm(A - np.kron(optB, optC))}")
    print(f"||optB (x) optC - B (x) C||_F = {comp_to_optimal} | Is optimal? {np.isclose(comp_to_optimal, 0)}")
    return B, C

# set seed for reproducibility
np.random.seed(42)
# A = np.random.rand(m1*m2, n1*n2)
A = np.arange(m1*m2*n1*n2).reshape((m1*m2, n1*n2))
print(A[0,0])
B, C = als_decomp(A, m1, n1, m2, n2)
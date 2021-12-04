""" Example 3

Let A is a matrix of size N-by-L.
Let x is a vector of size L-by-1.
Let b is a vector of size N-by-1.

Prove that

|| A x + b ||^2  >=  2 Re{[x0H @ AH @ A + bH @ A] @ x}
                      + (bH @ b - || A @ x0 ||^2)
where
AH = the conjugate transpose of A
bH = the conjugate transpose of b
x0 is an ARBITRARY vector of size L-by-1
x0H = the conjugate transpose of x0
Re{ a + jb } = a just takes the real part

"""

import numpy as np
import numpy.linalg as LA


def f(A, b, x):
    return LA.norm( A@x + b )**2


def f_LowerBound(A, b, x, x0):
    x0H = x0.conj().T  # conj. transpose
    xH = x.conj().T  # conj. transpose
    AH = A.conj().T
    bH = b.conj().T
    term_1 = (x0H @ AH @ A + bH @ A) @ x
    term_2 = np.real(bH @ b)
    term_3 = LA.norm(A@x0)**2
    result = 2*np.real(term_1) + term_2 - term_3
    # turn np.array([[abc]]) into abc
    result = result[0][0]
    return result


""" Main program """
N = 4
L = 3

# CHECK IF f > f_lower
# There is a violation if f < f_lower
# The program runs perfectly when there is NO violation

n_violations = 0
for _ in range(1000):  # 1000 experiments
    A = np.random.randn(N, L) + 1j*np.random.randn(N, L)
    b = np.random.randn(N, 1) + 1j*np.random.randn(N, 1)
    x = np.random.randn(L, 1) + 1j*np.random.randn(L, 1)
    delta_x = 0.1*(np.random.randn(L, 1) + 1j*np.random.randn(L, 1))
    x0 = x - delta_x
    # passing parameters into functions
    val_of_f = f(A, b, x)
    val_of_f_lower = f_LowerBound(A, b, x, x0)
    # check if there is a violation
    if val_of_f < val_of_f_lower:
        print('There is a violation.')
        n_violations += 1
    # end of if

print('The number of violations:', n_violations)
print('Current value of f =', val_of_f)
print('Current value of f_lower =', val_of_f_lower)

''' Example 1 '''

import numpy as np


def f_and_LowerBound(A_PSD, z_column, z0_column):
    """ z and z0 are column vectors, which are real-valued """
    z = z_column
    z0 = z0_column
    zT = z.T
    z0T = z0.T
    """ A is (P)ositive (S)emi(D)efinite """
    A = A_PSD
    AT = A.T
    """ Calculate the function f """
    f = zT @ A @ z
    """ Calculate the lower bound of f """
    f_lower = z0T @ (A + AT) @ z - z0T @ AT @ z0
    # np.real(a + 0j) = a
    f = np.real(f)
    f_lower = np.real(f_lower)
    return f[0][0], f_lower[0][0]


""" MAIN PROGRAM """
n_violations = 0
for i in range(100):
    N = 4
    temp = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    A = temp @ (temp.conj().T)  # A is now a positive semidefinite
    z = np.random.randn(N, 1)  # z is real-valued
    dz = 0.1*(np.random.randn(N, 1))  # dz is real-valued
    z0 = z - dz
    f, f_lower = f_and_LowerBound(A, z, z0)
    if f <= f_lower:
        n_violations += 1
    # end if
    print('f =', np.round(f, 2))
    print('f_lower =', np.round(f_lower, 2))
    print('------')

print('f is convex if A is a PSD matrix and z is real.')
print('Thus, A should be carefully generated.')
print('If f < f_lower, then there is a violation.')
print('n_violations =', n_violations)

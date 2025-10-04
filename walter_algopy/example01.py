import numpy as np
from algopy import UTPM, qr, solve, dot, eigh

def f(x):
    N, M = x.shape
    Q, R = qr(x)
    Id = np.eye(M)
    Rinv = solve(R,Id)
    C = dot(Rinv,Rinv.T)
    l, U = eigh(C)
    return l[0]

x = UTPM.init_jacobian(np.random.random((50,10)))
y = f(x)
J = UTPM.extract_jacobian(y)

print('Jacobian dy/dx =', J)
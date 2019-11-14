import numpy as np
import cvxopt

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    
    P = 0.5 * (P + P.T)  # make sure P is symmetric
    
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    
    sol = cvxopt.solvers.qp(*args)
    
    if 'optimal' not in sol['status']:
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))


M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)
q = -np.dot(M.T, np.array([3., 2., 3.]))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))

x = cvxopt_solve_qp(P, q, G, h)
print("x = ")
print(x)
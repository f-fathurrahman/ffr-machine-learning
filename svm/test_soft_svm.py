import numpy as np
import matplotlib.pyplot as plt
import cvxopt

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    
    P = 0.5 * (P + P.T)  # make sure P is symmetric
    
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    
    sol = cvxopt.solvers.qp(*args)
    
    if "optimal" not in sol["status"]:
        return None
    
    return np.array(sol["x"]).reshape((P.shape[1],)) # use np.ravel?


rng = np.random.RandomState(1234)

Ndata = 40
Nfeatures = 2

x1 = rng.randn(int(Ndata/2),Nfeatures)
x2 = rng.randn(int(Ndata/2),Nfeatures) + 4.0

t1 = -1.0*np.ones( (int(Ndata/2),1) )
t2 = np.ones( (int(Ndata/2),1) )

x = np.concatenate( (x1, x2), axis=0 )
t = np.concatenate( (t1, t2), axis=0 )

plt.clf()
plt.scatter(x1[:,0], x1[:,1], marker="o")
plt.scatter(x2[:,0], x2[:,1], marker="s")
plt.savefig("TEMP_test_svg_hard.pdf")

# Gram matrix
K = np.zeros((Ndata, Ndata))
for i in range(Ndata):
    for j in range(Ndata):
        K[i,j] = linear_kernel(x[i], x[j])

P = np.outer(t,t) * K
q = -1.0*np.ones(Ndata)
A = np.transpose(t) #np.matrix(t, (1,Ndata))
b = 0.0


C = 0.01

tmp1 = np.diag(np.ones(Ndata) * -1.0)
tmp2 = np.identity(Ndata)
G = np.vstack((tmp1, tmp2))

tmp1 = np.zeros(Ndata)
tmp2 = np.ones(Ndata) * C
h = np.hstack( (tmp1, tmp2) )

alpha = cvxopt_solve_qp( P, q, G, h, A, b )

sv = alpha > 1e-5
print(sv)
ind = np.arange(len(alpha))[sv]
sv_alpha = alpha[sv]
sv_x = x[sv]
sv_t = t[sv]
print("Terdapat %d support vector dari total %d data" % (len(sv_alpha), Ndata))


# Finding b
imax = 0
ss_old = 0.0
ss_max = 0.0
for i in range(len(sv_alpha)):
    ss = 0.0
    for m in range(len(sv_alpha)):
        ss = ss + sv_alpha[m] * sv_t[m]* np.dot( sv_x[m], sv_x[i] )
    if ss > ss_old:
        imax = i
        ss_max = ss
    ss_old = ss
    print("i = %4d ss = %18.10f" % (i, ss))

print("imax = ", imax)
b = sv_t[imax] - ss_max
print("b = %f" % b)

#b = b/len(sv_alpha)
#print("b = ", b)

# Weight vector
w = np.zeros(Nfeatures)
for n in range(len(sv_alpha)):
    w[:] = w[:] + sv_alpha[n] * sv_t[n] * sv_x[n]
    print(sv_t[n])

print("w = ", w)

t_new = np.sign(np.dot(x, w) + b)
print(t_new - t[:,0])


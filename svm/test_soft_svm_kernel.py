import numpy as np
import matplotlib.pyplot as plt
import cvxopt

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

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

def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = rng.multivariate_normal(mean1, cov, 100)
    y1 = np.ones( (len(X1),1) )
    X2 = rng.multivariate_normal(mean2, cov, 100)
    y2 = np.ones( (len(X2),1) ) * -1
    return X1, y1, X2, y2


def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = rng.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, rng.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones( (len(X1),1) ) 
    X2 = rng.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, rng.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones( (len(X2),1) ) * -1
    return X1, y1, X2, y2

#x1, t1, x2, t2 = gen_lin_separable_data()
x1, t1, x2, t2 = gen_non_lin_separable_data()


x = np.concatenate( (x1, x2), axis=0 )
t = np.concatenate( (t1, t2), axis=0 )

Ndata = x.shape[0]

print(x.shape)
print(t.shape)
Nfeatures = 2

plt.clf()
plt.scatter(x1[:,0], x1[:,1], marker="o")
plt.scatter(x2[:,0], x2[:,1], marker="s")
#plt.savefig("TEMP_lin_separable_data.pdf") 
plt.savefig("TEMP_non_lin_separable_data.pdf") 


# Gram matrix
K = np.zeros((Ndata, Ndata))
for i in range(Ndata):
    for j in range(Ndata):
        #K[i,j] = linear_kernel(x[i], x[j])
        #K[i,j] = polynomial_kernel(x[i], x[j])
        K[i,j] = gaussian_kernel(x[i], x[j])

P = np.outer(t,t) * K
q = -1.0*np.ones(Ndata)
A = np.transpose(t)
b = 0.0

#C = 1000.1
C = None

if C is None:
    G = np.diag(np.ones(Ndata) * -1)
    h = np.zeros(Ndata)
else:
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
print("There are %d support vectors from %d data points" % (len(sv_alpha), Ndata))


# Finding b
if C is None:
    for i in range(len(sv_alpha)):
        ss = 0.0
        b = 0.0
        for m in range(len(sv_alpha)):
            ss = ss + sv_alpha[m] * sv_t[m]* np.dot( sv_x[m], sv_x[i] )
        b = sv_t[i] - ss
        print("b: %d %f" % (i, b))
else:
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


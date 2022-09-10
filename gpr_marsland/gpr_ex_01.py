import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def kernel2(data1, data2, theta, wantderiv=True, measnoise=1.):
    # Uses exp(theta) to ensure positive hyperparams
    theta = np.squeeze(theta)
    theta = np.exp(theta)
    # Squared exponential
    if np.ndim(data1) == 1:
        d1 = np.shape(data1)[0]
        n = 1
        data1 = data1*np.ones((d1,1))
        data2 = data2*np.ones((np.shape(data2)[0],1))
    else:
        (d1,n) = np.shape(data1)

    d2 = np.shape(data2)[0]
    sumxy = np.zeros((d1,d2))
    for d in range(n):
        D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
        D2 = [data2[:,d]] * np.ones((d1,d2))
        sumxy += (D1-D2)**2*theta[d+1]

    k = theta[0] * np.exp(-0.5*sumxy)
    if wantderiv:
        K = np.zeros((d1,d2,len(theta)+1))
        K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
        K[:,:,1] = k 
        K[:,:,2] = -0.5*k*sumxy
        K[:,:,3] = theta[2]*np.eye(d1,d2)
        return K
    else:    
        return k + measnoise*theta[2]*np.eye(d1,d2)

def gradLogPosterior(theta, *args):
    data,t = args
    theta = np.squeeze(theta)
    d = len(theta)
    K = kernel2(data,data,theta,wantderiv=True)

    L = np.linalg.cholesky(np.squeeze(K[:,:,0]))
    invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(np.shape(data)[0])))
    
    dlogpdtheta = np.zeros(d)
    for d in range(1,len(theta)+1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.transpose(), np.dot(invk, np.dot(np.squeeze(K[:,:,d]), np.dot(invk,t)))) - 0.5*np.trace(np.dot(invk,np.squeeze(K[:,:,d])))

    return -dlogpdtheta


def logPosterior(theta, *args):
    data, t = args
    k = kernel2(data,data,theta,wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
    logp = -0.5*np.dot(t.transpose(),beta) - np.sum(np.log(np.diag(L))) - \
            np.shape(data)[0]/2.0 * np.log(2*np.pi)
    return -logp

def demo_plot(filsave, theta, *args):
    colour = np.array([0,0,1.0])
    faded = 1-(1-colour)/2.0

    X, y = args
    n, D = np.shape(X)

    xrange = X.max() - X.min()
    Xtest = np.arange(X.min()-xrange/2,X.max()+xrange/2,(X.max()-X.min())/100)
    Xtest.shape = (len(Xtest),1)

    k = kernel2(X,X,theta,wantderiv=False)
    kstar = [kernel2(X,xs*np.ones((1,1)), theta, wantderiv=False, measnoise=False) for xs in Xtest]
    kstar = np.squeeze(kstar)
    kstarstar = [kernel2(xs*np.ones((1,1)), xs*np.ones((1,1)), theta, wantderiv=False, measnoise=False) for xs in Xtest]
    kstarstar = np.squeeze(kstarstar)

    L = np.linalg.cholesky(k)
    invk = np.linalg.solve(L.transpose(), np.linalg.solve(L, np.eye(np.shape(X)[0])))
    mean = np.dot(kstar,np.dot(invk,y))
    var = kstarstar - np.diag(np.dot(kstar, np.dot(invk, kstar.T)))
    #var = np.reshape(var,(100,1))

    plt.clf()
    plt.plot(Xtest, mean,'-k')
    plt.fill_between(np.squeeze(Xtest), np.squeeze(mean-2*np.sqrt(var)),
        np.squeeze(mean+2*np.sqrt(var)), color='0.75')
    plt.plot(X,y,'ko')
    plt.savefig(filsave)

def plot_data(X, y):
    x_sort = np.copy(X.flatten())
    y_sort = np.copy(y.flatten())

    idx_sort = np.argsort(x_sort)
    x_sort = x_sort[idx_sort]
    y_sort = y_sort[idx_sort]

    print(x_sort)
    print(y_sort)

    plt.clf()
    plt.plot(x_sort, y_sort, marker="o")
    plt.grid()
    plt.savefig("IMG_data.pdf")



#data = np.loadtxt("data.txt")
data = np.loadtxt("data2.txt")
X = data[:,0:-1]
y = data[:,-1]

print("X = "); print(X); print(X.shape)
print("y = "); print(y); print(y.shape)

print(np.mean(X))
print(np.mean(y))


args = (X, y)

np.random.seed(12)
#theta = np.array([1.65593058, -0.86208073,-1.14525511])
theta = np.zeros(3)
theta[0] = np.random.normal(0,1)
theta[1] = np.random.normal(0,1)
theta[2] = np.random.normal(0,1)

print("theta = ", theta)
print("exp theta = ", np.exp(theta))
print(logPosterior(theta, *args))
print(gradLogPosterior(theta, *args))
print(scipy.optimize.check_grad(logPosterior, gradLogPosterior, theta, *args))

newTheta = scipy.optimize.fmin_cg(logPosterior, theta,
    fprime=gradLogPosterior, args=args,
    gtol=1e-4,maxiter=100,disp=1)

print("theta = ", theta)
print("newTheta = ", newTheta)
print(newTheta, logPosterior(newTheta, *args))

K = kernel2(X, X, newTheta, wantderiv=False)
L = np.linalg.cholesky(K) 

β = np.linalg.solve(L.transpose(), np.linalg.solve(L,y))
print("β = ", β)

test = X

#pred = [predict(i,input,K,target,newTheta,L,beta) for i in input]
#pred = np.squeeze([predict(i,input,K,target,newTheta,L,beta) for i in input])
demo_plot("IMG_initial_param.pdf", theta, *args)
demo_plot("IMG_new_param.pdf", newTheta, *args)

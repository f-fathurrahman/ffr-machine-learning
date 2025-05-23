import numpy as np

def cosamp(phi, u, s, epsilon=1e-10, max_iter=1000):
    """
    Return an `s`-sparse approximation of the target signal
    Input:
        - phi, sampling matrix
        - u, noisy sample vector
        - s, sparsity
    """
    a = np.zeros(phi.shape[1])
    v = u
    it = 0 # count
    halt = False
    while not halt:
        it += 1
        print("Iteration {}\r".format(it), end="")
        
        y = np.dot(np.transpose(phi), v)
        omega = np.argsort(y)[-(2*s):] # large components
        omega = np.union1d(omega, a.nonzero()[0]) # use set instead?
        phiT = phi[:, omega]
        b = np.zeros(phi.shape[1])
        # Solve Least Square
        b[omega], _, _, _ = np.linalg.lstsq(phiT, u)
        
        # Get new estimate
        b[np.argsort(b)[:-s]] = 0
        a = b
        
        # Halt criterion
        v_old = v
        v = u - np.dot(phi, a)

        halt = (np.linalg.norm(v - v_old) < epsilon) or \
            np.linalg.norm(v) < epsilon or \
            it > max_iter
        
    return a


n_rows = 10000
n_cols = 1000
sparsity = 35

A = np.random.normal(0, 1, [n_rows, n_cols])
# Generate sparse x and noise
x = np.zeros(n_cols)
x[np.random.randint(1, n_cols, [sparsity])] = np.random.chisquare(15, [sparsity])
noise = np.random.normal(0, 1, [n_cols])

u = x + noise

y = np.dot(A, u)

x_est = cosamp(A, y, 50)
# Score estimation
np.linalg.norm(x - x_est) / np.linalg.norm(x)



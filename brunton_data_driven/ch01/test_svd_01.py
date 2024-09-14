# %%
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]


# %%
DATA_PATH = os.path.join("..", "ORIG_", "DATA")
A = imread(os.path.join(DATA_PATH, "dog.jpg"))
X = np.mean(A, -1); # Convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap("gray")
plt.axis("off")
plt.show()


# %%
U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()


# %%
## f_ch01_ex02_2
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

# %%
plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()


# %%

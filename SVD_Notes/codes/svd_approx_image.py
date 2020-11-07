# From:
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = matplotlib.image.imread(sys.argv[1])

# Or using PIL
img_gray = rgb2gray(img)
print("shape of img_gray = ", img_gray.shape)
plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
plt.savefig("IMG_orig.pdf")

U, Σ, Vt = np.linalg.svd(img_gray, full_matrices=False)
print("done SVD", flush=True)

for r in [5,10,20,50,100,150]:
    plt.clf()
    img_approx = np.matmul( np.matmul(U[:,:r], np.diag(Σ[:r])), Vt[:r,:] )
    print("Done img_approx", flush=True)
    plt.imshow(img_approx, cmap=plt.get_cmap("gray"))
    plt.savefig("IMG_r_" + str(r) + ".pdf")

# From:
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = matplotlib.image.imread("/home/efefer/LAPTOP_LAMA/rf/sm8830dcc84e486a9c4beab790088ea339.jpg")

# Or using PIL
img_gray = rgb2gray(img)    
#plt.imshow(gray, cmap=plt.get_cmap("gray"), vmin=0, vmax=1)
plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
plt.savefig("IMG_savefig.pdf")

U, Σ, Vt = np.linalg.svd(img_gray, full_matrices=False)

for r in [5,20,100]:
    plt.clf()
    img_approx = np.matmul( np.matmul(U[:,:r], np.diag(Σ[:r])), Vt[:r,:] )
    plt.imshow(img_approx, cmap=plt.get_cmap("gray"))
    plt.savefig("IMG_r_" + str(r) + ".pdf")

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

from my_utils import save_fig, plot_digits

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
print("Data loading finished")

X = mnist["data"]
y = mnist["target"]

some_digit = X[17] # X[0,:]
some_digit_image = some_digit.reshape(28, 28)

#plt.clf()
#plt.imshow(some_digit_image, cmap=mpl.cm.binary)
#plt.axis("off")
#save_fig("some_digit_plot")
#plt.show()

y = y.astype(np.uint8)

plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()
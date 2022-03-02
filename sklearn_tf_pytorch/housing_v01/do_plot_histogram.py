import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv("housing.csv")

housing.hist(bins=50, figsize=(20,15))
plt.savefig("IMG_attribute_histogram_plots.pdf")


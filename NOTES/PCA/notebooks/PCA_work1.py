# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <h1 style="text-align: center;">Principal component analysis</h1>

# %% [markdown] heading_collapsed=true jp-MarkdownHeadingCollapsed=true
# # Teori singkat

# %% [markdown] hidden=true
# Akan dilakukan proyeksi dari data berdimensi $M$ menjadi berdimensi $D$.
#
# PCA akan mendefinisikan $D$ vektor, $\mathbf{w}_{d}$, masing-masing berdimensi $M$. Elemen ke $d$ dari proyeksi $x_{nd}$ di mana $\mathbf{x}_{nd} = [ x_{n1}, x_{n2}, \ldots, x_{nD} ]^{\mathrm{T}}$ dihitung sebagai:
#
# $$
# x_{nd} = \mathbf{w}^{\mathrm{T}}_{d} \mathbf{y}_{n}
# $$
#

# %% [markdown] hidden=true
# Proses pembelajaran dalam hal ini adalah berapa dimensi $D$ dan memilih vektor proyeksi $\mathbf{w}_{d}$ untuk setiap dimensi.

# %% [markdown] hidden=true
# PCA menggunakan variansi pada ruang yang diproyeksikan sebagai kriteria untuk memilih $\mathbf{w}_{d}$.
#
# Misalnya: $\mathbf{w}_{1}$ adalah proyeksi yang akan membuat variansi pada $x_{n1}$ semaksimal mungkin.
#
# Dimensi proyeksi kedua juga dipilih untuk memaksimalkan variansi, namum $\mathbf{w}_{2}$ harus ortogonal terhadap $\mathbf{w}_{1}$:
# $$
# \mathbf{w}_{1}^{\mathrm{T}} \mathbf{w}_{1} = 0
# $$
#
# Begitu juga untuk dimensi proyeksi yang ketiga dan seterusnya.
#
# Secara umum:
# $$
# \mathbf{w}_{i}^{\mathrm{T}} \mathbf{w}_{j} = 0 \, \forall\, j \neq i
# $$

# %% [markdown] hidden=true
# Asumsi:
# $$
# \bar{\mathbf{y}} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{y}_{n} = 0
# $$

# %% [markdown] hidden=true
# Misalkan kita ingin mencari proyeksi ke $D=1$ dimensi, dalam kasus ini hasil proyeksi adalah nilai skalar $x_{n}$ untuk tiap observasi yang diberikan oleh:
# $$
# x_{n} = \mathbf{w}^{\mathsf{T}} \mathbf{y}_{n}
# $$
# dan variansi $\sigma^{2}_{x}$ diberikan oleh:
# $$
# \sigma^{2}_{x} = \frac{1}{N} \sum_{n=1}^{N} \left( x_{n} - \bar{x} \right)^2
# $$

# %% [markdown] hidden=true
# Dengan asumsi bahwa $\bar{y} = 0$:
# $$
# \begin{align}
# \bar{x} & = \frac{1}{N} \sum_{n=1}^{N} \mathbf{w}^{\mathsf{T}} \mathbf{y}_{n} \\
# & = \mathbf{w}^{\mathsf{T}} \left(
# \frac{1}{N} \sum_{n=1}^{N} \mathbf{y}_{n}
# \right) \\
# & = \mathbf{w}^{\mathsf{T}} \bar{\mathbf{y}} \\
# & = 0
# \end{align}1
# $$

# %% [markdown] hidden=true
# sehingga variansinya menjadi:
# $$
# \begin{align}
# \sigma_{x}^{2} & = \frac{1}{N} \sum_{n=1}^{N} x^{2}_{n} \\
# & = \frac{1}{N} \sum_{n=1}^{N} \left(
# \mathbf{w}^{\mathsf{T}} \mathbf{y}_{n} \right)^2 \\
# & = \frac{1}{N} \sum_{n=1}^{N} \mathbf{w}^{\mathsf{T}} \mathbf{y}_{n}
# \mathbf{y}_{n}^{\mathsf{T}} \mathbf{w} \\
# & = \mathbf{w}^{\mathsf{T}} 
# \left( \frac{1}{N} \sum_{n=1}^{N}
# \mathbf{y}_{n} \mathbf{y}_{n}^{\mathsf{T}}
# \right)
# \mathbf{w} \\
# & = \mathbf{w}^{\mathsf{T}} \mathbf{C} \mathbf{w}
# \end{align}
# $$

# %% [markdown] hidden=true
# $\mathbf{C}$ adalah matriks kovariansi dari sampel:
# $$
# \mathbf{C} = \frac{1}{N} \sum_{n=1}^{N}
# (\mathbf{y}_{n} - \bar{\mathbf{y}})
# (\mathbf{y}_{n} - \bar{\mathbf{y}})^{\mathsf{T}}
# $$
# di mana $\bar{\mathbf{y}} = 0$ dalam kasus yang kita tinjau.
#

# %% [markdown]
# # Kode program

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# %%
import matplotlib
matplotlib.style.use("dark_background")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "grid.color" : "gray",
    "grid.linestyle" : "--"
})

# %% [markdown]
# Generate data:

# %%
np.random.seed(1234) # supaya reproducible

# %%
Y_1 = np.random.randn(20,2)
Y_2 = np.random.randn(20,2) + 5
Y_3 = np.random.randn(20,2) - 5

# %%
idx_1 = range(0,20)
idx_2 = range(20,40)
idx_3 = range(40,60)

# %%
Y = np.concatenate( (Y_1, Y_2, Y_3), axis=0 );

# %%
plt.clf()
plt.scatter(Y[idx_1,0], Y[idx_1,1], marker="*", color="blue")
plt.scatter(Y[idx_2,0], Y[idx_2,1], marker="s", color="green")
plt.scatter(Y[idx_3,0], Y[idx_3,1], marker="o", color="red");

# %% [markdown]
# Add random dimensions:

# %%
Ndata = Y.shape[0]
Y = np.concatenate( (Y, np.random.randn(Ndata,5)), axis=1 )

# %%
Y.shape

# %% [markdown]
# Coba plot data pada dimensi lain (tidak penting) yang sudah ditambahkan:

# %%
plt.clf()
dim1 = 2 # idx start from 0
dim2 = 3
plt.scatter(Y[idx_1,dim1], Y[idx_1,dim2], marker="*", color="blue")
plt.scatter(Y[idx_2,dim1], Y[idx_2,dim2], marker="s", color="green")
plt.scatter(Y[idx_3,dim1], Y[idx_3,dim2], marker="o", color="red");

# %%
plt.clf()
dim1 = 3 # idx start from 0
dim2 = 6
plt.scatter(Y[idx_1,dim1], Y[idx_1,dim2], marker="*", color="blue")
plt.scatter(Y[idx_2,dim1], Y[idx_2,dim2], marker="s", color="green")
plt.scatter(Y[idx_3,dim1], Y[idx_3,dim2], marker="o", color="red");

# %%
plt.clf()
dim1 = 6 # idx start from 0
dim2 = 1
plt.scatter(Y[idx_1,dim1], Y[idx_1,dim2], marker="*", color="blue")
plt.scatter(Y[idx_2,dim1], Y[idx_2,dim2], marker="s", color="green")
plt.scatter(Y[idx_3,dim1], Y[idx_3,dim2], marker="o", color="red");

# %%
labels = np.concatenate( ([0]*20, [1]*20, [2]*20) )

# %%
labels

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    plt.scatter(Y[idx,0], Y[idx,1], marker=markers[i])
plt.grid()

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    plt.scatter(Y[idx,0], Y[idx,1], marker=markers[i])
plt.gca().axis("square");

# %%

# %% [markdown]
# Hitung rata-rata: $\bar{\mathbf{y}}$, gunakan metode `np.mean`.

# %%
ybar = np.mean(Y,axis=0)
ybar

# %% [markdown]
# Geser data terhadap rata-rata:

# %%
Yshifted = Y - ybar

# %% [markdown]
# Rata-rata dari data yang sekarang seharusnya adalah nol (vektor).

# %%
np.mean(Yshifted,axis=0)

# %% [markdown]
# Hitung matriks kovariansi: $\mathbf{C}$
#
# $$
# \mathbf{C} = \frac{1}{N} \mathbf{Y}^{\mathsf{T}} \mathbf{Y}
# $$

# %%
N = Yshifted.shape[0]

# %%
C = Yshifted.T @ Yshifted / N

# %% [markdown]
# (7,60) (60,7)

# %%
C.shape

# %% [markdown]
# Alternatif:
# ```python
# C = np.matmul( Yshifted.transpose(), Yshifted )/N
# ```

# %%
C[1,2], C[2,1]

# %% [markdown]
# Hitung pasangan eigen dari matriks kovariansi:

# %%
λss, ws = np.linalg.eigh(C)

# %%
λss

# %%
λ_unsrt, w_unsrt = np.linalg.eig(C)

# %%
λ_unsrt

# %%
w_unsrt.shape

# %% [markdown]
# Cek persamaan eigen:
# $$
# \mathbf{C} \mathbf{w} = \lambda \mathbf{w}
# $$

# %%
C @ w_unsrt[:,2]

# %%
λ_unsrt[2] * w_unsrt[:,2]

# %%
np.dot(w_unsrt[:,6], w_unsrt[:,6])

# %% [markdown]
# (Lakukan sort jika perlu)

# %%
idx_sorted = np.argsort(λ_unsrt)[::-1]

# %%
idx_sorted

# %%
λ = λ_unsrt[idx_sorted]

# %% [markdown]
# Eigenvalue yang sudah disort:

# %%
λ

# %% [markdown]
# Eigenvektor yang sudah disort:

# %%
w = w_unsrt[:,idx_sorted]

# %% [markdown]
# Perbandingan nilai eigen (variansi)

# %%
plt.bar(range(len(λ)), λ);

# %% [markdown]
# Lakukan proyeksi:
#
# $$
# \mathbf{X} = \mathbf{Y} \mathbf{W}
# $$

# %%
Yshifted.shape, w.shape

# %% [markdown]
# Matrix Y: (60,7), jumlah data 60, jumlah fitur 7
#
# Matrix W: (7,D), D: jumlah dimensi baru
#
# Matrix X: (60,D)

# %% [markdown]
# Proyeksikan data ke dua dimensi pertama:

# %%
Yproj = Y @ w[:,0:2]

# %%
Yproj.shape

# %%
plt.scatter(Yproj[idx_1,0], Yproj[idx_1,1], marker="*", color="blue")
plt.scatter(Yproj[idx_2,0], Yproj[idx_2,1], marker="s", color="green")
plt.scatter(Yproj[idx_3,0], Yproj[idx_3,1], marker="o", color="red");
plt.xlabel("1st proj dim")
plt.ylabel("2nd proj dim");

# %%
plt.plot(Yproj_1, np.ones(Yproj_1.shape[0]), marker="o", lw=0)

# %% [markdown]
# Hanya memproyeksikan ke 1st proj dim (nilai eigen paling besar):

# %%
Yproj_1 = np.matmul(Y, w[:,0:1] )
Yproj_1.shape

# %%
plt.plot(Yproj_1[idx_1], np.zeros(len(idx_1)), marker="*", color="blue", lw=0)
plt.plot(Yproj_1[idx_2], np.zeros(len(idx_2)), marker="s", color="green", lw=0)
plt.plot(Yproj_1[idx_3], np.zeros(len(idx_3)), marker="o", color="red", lw=0)
plt.xlabel("proj dim 1st");

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    yplot = Yproj_1[idx,0]
    plt.plot(yplot, np.ones(len(yplot)), marker=markers[i], lw=0)
plt.grid()

# %% [markdown]
# **NOTE: sumbu x bukan menyatakan fitur pertama (desc1), namum fitur (dimensi) yang sudah ditransformasi.**

# %% [markdown]
# Bagaimana jika data diproyeksi ke dimensi proyeksi ke-dua saja?

# %%
plt.clf()
markers = ["o", "s", "*"]
for i in range(3):
    idx = labels==i
    yplot = Yproj_2[idx]
    plt.plot(yplot, np.ones(len(yplot)), marker=markers[i], lw=0)
plt.grid()

# %% [markdown]
#

# %% [markdown] heading_collapsed=true
# # Pair plot

# %% hidden=true
import pandas as pd

# %% hidden=true
import seaborn as sns
sns.set()

# %% hidden=true
labelsString = []
for i in range(len(labels)):
    labelsString.append("class"+str(labels[i]))

# %% hidden=true
columns = ["desc"+str(i) for i in range(1,8)]
columns

# %% hidden=true
Ypd = pd.DataFrame(Y, columns=columns)

# %% hidden=true
labelsDF = pd.DataFrame(labelsString, columns=["class"])
labelsDF.head()

# %% hidden=true
YDF = pd.merge(Ypd, labelsDF, left_index=True, right_index=True)
YDF.head()

# %% hidden=true
columns = ["desc"+str(i) for i in range(1,8)]
columns

# %% hidden=true
sns.pairplot(YDF, hue="class")

# %% hidden=true
YDF.head()

# %% hidden=true
YDF.loc[:,:"desc7"].head()

# %% [markdown]
# # Coba lagi:

# %%
Y = []

# %%
Y_1 = np.random.randn(20,2)
Y_2 = np.random.randn(20,2) + 10
Y_3 = np.random.randn(20,2) - 10
Y_4 = np.random.randn(20,2) + np.array([-10,10])
Y_5 = np.random.randn(20,2) + np.array([10,-10])

# %%
Y = np.concatenate( (Y_1, Y_2, Y_3, Y_4, Y_5), axis=0 );

# %%
plt.clf()
plt.scatter(Y[:,0], Y[:,1])

# %%
Ndata = Y.shape[0]
Y = np.concatenate( (Y, np.random.randn(Ndata,5)), axis=1 )

# %%
plt.clf()
plt.scatter(Y[:,4], Y[:,5])

# %%

# %%
ybar = np.mean(Y,axis=0)
ybar

# %%
Yshifted = Y - ybar

# %%
np.mean(Yshifted,axis=0)

# %%
N = Yshifted.shape[0]
N

# %%
C = np.matmul( Yshifted.transpose(), Yshifted )/N

# %%
λ_unsrt, w_unsrt = np.linalg.eig(C)

# %%
λ_unsrt

# %%
idx_sorted = np.argsort(λ_unsrt)[::-1]
idx_sorted

# %%
W = w_unsrt[:,idx_sorted]
λ = λ_unsrt[idx_sorted]

# %%
λ

# %%
plt.bar(np.arange(len(λ)), λ)

# %%

# %%

# %%

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter: Multivariate Statistical Process Monitoring for Linear and Steady-State Processes: Part 1
#
#
# # Topic: Process Monitoring using PLS

# %%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          train PLS model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
# import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# fetch data
data = pd.read_csv('LDPE.csv', usecols = range(1,20)).values
data_train = data[:-4,:] # exclude last 4 samples

# plot quality variables
quality_var = 5

plt.figure()
plt.plot(data[:,13+quality_var], '*')
plt.xlabel('sample #')
plt.ylabel('quality variable ' + str(quality_var))
plt.show()

# %%
# scale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_train_normal = scaler.fit_transform(data_train)

# %%
# build PLS model
from sklearn.cross_decomposition import PLSRegression

X_train_normal = data_train_normal[:,:-5]
Y_train_normal = data_train_normal[:,-5:]

pls = PLSRegression(n_components = 3)
pls.fit(X_train_normal, Y_train_normal)

# %%
# compute X and Y variance captured
from sklearn.metrics import r2_score

print('Y variance captured: ', 100*pls.score(X_train_normal, Y_train_normal), '%')

Tscores = pls.x_scores_
X_train_normal_reconstruct = np.dot(Tscores, pls.x_loadings_.T) 
# can also use pls.inverse_transform(Tscores)

print('X variance captured: ', 100*r2_score(X_train_normal, X_train_normal_reconstruct), '%')

# %%
# visualize t vs u
Tscores = pls.x_scores_
Uscores = pls.y_scores_

comp = 3
plt.figure()
plt.plot(Tscores[:,comp-1], Uscores[:,comp-1], '*')
plt.xlabel('t{0}'.format(comp))
plt.ylabel('u{0}'.format(comp))
plt.show()

# %%
# monitoring indices for training data
# T2
T_cov = np.cov(Tscores.T)
T_cov_inv = np.linalg.inv(T_cov)

T2_train = np.zeros((data_train_normal.shape[0],))

for i in range(data_train_normal.shape[0]):
    T2_train[i] = np.dot(np.dot(Tscores[i,:],T_cov_inv),Tscores[i,:].T)

# SPEx
x_error_train = X_train_normal - X_train_normal_reconstruct
SPEx_train = np.sum(x_error_train*x_error_train, axis = 1)

# SPEy
y_error_train = Y_train_normal - pls.predict(X_train_normal)
SPEy_train = np.sum(y_error_train*y_error_train, axis = 1)


# %%
# control limits
#T2
import scipy.stats

N = data_train_normal.shape[0]
k = 3

alpha = 0.01 # 99% control limit
T2_CL = k*(N**2-1)*scipy.stats.f.ppf(1-alpha,k,N-k)/(N*(N-k))

#SPEx
mean_SPEx_train = np.mean(SPEx_train)
var_SPEx_train = np.var(SPEx_train)

g = var_SPEx_train/(2*mean_SPEx_train)
h = 2*mean_SPEx_train**2/var_SPEx_train
SPEx_CL = g*scipy.stats.chi2.ppf(1-alpha, h)

#SPEy
mean_SPEy_train = np.mean(SPEy_train)
var_SPEy_train = np.var(SPEy_train)

g = var_SPEy_train/(2*mean_SPEy_train)
h = 2*mean_SPEy_train**2/var_SPEy_train
SPEy_CL = g*scipy.stats.chi2.ppf(1-alpha, h)

# %%
# monitoring charts
# T2_train plot with CL
plt.figure()
plt.plot(T2_train)
plt.plot([1,len(T2_train)],[T2_CL,T2_CL], color='red')
plt.xlabel('Sample #')
plt.ylabel('T$^2$ for training data')
plt.show()

# SPEx plot with CL
plt.figure()
plt.plot(SPEx_train)
plt.plot([1,len(SPEx_train)],[SPEx_CL,SPEx_CL], color='red')
plt.xlabel('Sample #')
plt.ylabel('SPEx for training data')
plt.show()
         
# SPEy plot with CL
plt.figure()
plt.plot(SPEy_train)
plt.plot([1,len(SPEy_train)],[SPEy_CL,SPEy_CL], color='red')
plt.xlabel('Sample #')
plt.ylabel('SPEy for training data')
plt.show()

# %%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          fault detection on complete data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%
# get test data, normalize it
data_normal = scaler.transform(data)
X_normal = data_normal[:,:-5]
Y_normal = data_normal[:,-5:]

# %%
# get model predictions 
Tscores_test = pls.transform(X_normal)
X_normal_reconstruct = np.dot(Tscores_test, pls.x_loadings_.T)
Y_normal_pred = pls.predict(X_normal)

# %%
# compute monitoring statistics
T2_test = np.zeros((data_normal.shape[0],))
for i in range(data_normal.shape[0]):
    T2_test[i] = np.dot(np.dot(Tscores_test[i,:],T_cov_inv),Tscores_test[i,:].T)

x_error_test = X_normal - X_normal_reconstruct
SPEx_test = np.sum(x_error_test*x_error_test, axis = 1)

y_error_test = Y_normal - pls.predict(X_normal)
SPEy_test = np.sum(y_error_test*y_error_test, axis = 1)

# %%
# plot
plt.figure()
plt.plot(T2_test, '*')
plt.plot([1,len(T2_test)],[T2_CL,T2_CL], color='red')
plt.xlabel('Sample #')
plt.ylabel('T$^2$ for complete dataset')
plt.show()

plt.figure()
plt.plot(SPEx_test, '*')
plt.plot([1,len(SPEx_test)],[SPEx_CL,SPEx_CL], color='red')
plt.xlabel('Sample #')
plt.ylabel('SPEx for complete dataset')
plt.show()

plt.figure()
plt.plot(SPEy_test, '*')
plt.plot([1,len(SPEy_test)],[SPEy_CL,SPEy_CL], color='red')
plt.xlabel('Sample #')
plt.ylabel('SPEy for complete dataset')
plt.show()


# %%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                  fault diagnosis by contribution plots
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sample = 54
data_point = np.transpose(data_normal[sample-1,])

# SPEx contribution
x_error_test_sample = x_error_test[sample-1,]
SPEx_contri = x_error_test_sample*x_error_test_sample # vector of contributions

plt.figure(figsize=[8,4])
plt.bar(['var ' + str((i+1)) for i in range(len(SPEx_contri))], SPEx_contri, color='brown')
plt.xticks(rotation = 80, fontsize=18)
plt.ylabel('SPEx contribution plot', fontsize=20)
plt.show()

# SPEy contribution
y_error_test_sample = y_error_test[sample-1,]
SPEy_contri = y_error_test_sample*y_error_test_sample # vector of contributions

plt.figure(figsize=[4,4])
plt.bar(['var ' + str((i+1)) for i in range(len(SPEy_contri))], SPEy_contri, color='brown')
plt.xticks(rotation = 80, fontsize=18)
plt.ylabel('SPEy contribution plot', fontsize=20)
plt.show()

# T2 contribution
W = pls.x_weights_
P = pls.x_loadings_
R = np.dot(W, np.linalg.inv(np.dot(P.T, W)))
Ghe = np.dot(scipy.linalg.sqrtm(T_cov_inv), R.T)

T2_contri = np.zeros((X_train_normal.shape[1],))
for i in range(X_train_normal.shape[1]):
    vect = Ghe[:,i]*data_point[i]
    T2_contri[i] = np.dot(vect, vect)

plt.figure(figsize=[8,4])
plt.bar(['var ' + str((i+1)) for i in range(len(T2_contri))], T2_contri, color='brown')
plt.xticks(rotation = 80, fontsize=18)
plt.ylabel('T$^2$ contribution plot', fontsize=20)

# variable plot
plt.figure(figsize=[6,4])
plt.plot(data[:,8], '-*')
plt.xlabel('Sample #', fontsize=20)
plt.ylabel('Variable 9', fontsize=20)
plt.grid()
plt.show()

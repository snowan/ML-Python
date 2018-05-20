
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:

import os
path = '/Users/xwan/developers/ML/ML-Python/coursea_ml/data/ex1data1.txt' 
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()


# In[5]:

data.describe()


# In[6]:

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))


# In[7]:

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# In[8]:

data.insert(0, 'Ones', 1)


# In[9]:

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]


# In[10]:

X.head()


# In[11]:

y.head()


# In[12]:

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))


# In[13]:

theta


# In[14]:

X.shape, theta.shape, y.shape


# In[15]:

computeCost(X, y, theta)


# In[16]:

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
    
    return theta, cost


# In[17]:

alpha = 0.01
iters = 1000


# In[18]:

g, cost = gradientDescent(X,y,theta,alpha,iters)
g

# In[20]:

computeCost(X, y, g)


# In[21]:

# plot the linear model along with the data to visually see how well it gits
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + (g[0,1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Population')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Preditcted Profit vs. Population Size')


# In[23]:

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# In[27]:

# Linear regression with multiple variables
path = '/Users/xwan/developers/ML/ML-Python/coursea_ml/data/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()


# In[28]:

# pre-processing data - normalizing the features
data2 = (data2 - data2.mean()) / data2.std()
data2.head()


# In[38]:

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)


# In[40]:

data2.head()
X2
y2
theta2 = np.matrix(np.array([0, 0, 0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)


# In[41]:

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Itertations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# In[42]:

# use scikit-learn' linear regression function
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,y)


# In[47]:

# predict
x = np.array(X[:,1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label = 'Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population size')


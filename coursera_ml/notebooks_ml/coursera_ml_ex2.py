
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

import os
path = os.getcwd() + '/../data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()


# In[4]:

# Let's create a scatter plot of the two scores and use color coding to visualize if the example is positive (admitted) or negative (not admitted).

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# In[5]:

# create a sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[6]:

# quick sanity check to make sure the function is working

nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')


# In[7]:

#  write the cost function to evaluate a solution.

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# In[8]:

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)


# In[11]:

# quickly check the shape of our arrays to make sure everything looks good.
X.shape, theta.shape, y.shape


# In[12]:

# compute the cost for our initial solution (0 values for theta).
cost(theta, X, y)


# In[13]:

# need a function to compute the gradient (parameter updates) given our training data, labels, and some parameters theta.

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad


# In[14]:

# Note that we don't actually perform gradient descent in this function - we just compute a single gradient step. 
# In the exercise, an Octave function called "fminunc" is used to optimize the parameters given functions to 
# compute the cost and the gradients. Since we're using Python, we can use SciPy's "optimize" namespace to do the same thing.

# Let's look at a single call to the gradient method using our data and initial paramter values of 0.

gradient(theta, X, y)


# In[15]:

# use SciPy's truncated newton (TNC) implementation to find the optimal parameters
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
result


# In[16]:

cost(result[0], X, y)


# In[17]:

# write a function that will output predictions for a dataset X using our learned parameters theta. 
# We can then use this function to score the training accuracy of our classifier.
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# In[19]:

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))


# In[20]:

# Our logistic regression classifer correctly predicted if a student was admitted or not 89% of the time. 
# Not bad! Keep in mind that this is training set accuracy though. 
# We didn't keep a hold-out set or use cross-validation to get a true approximation of the accuracy 
# so this number is likely higher than its true perfomance (this topic is covered in a later exercise).


# In[22]:

### Regularized logistic regression
# To help you make the decision, you have a dataset of test results on past microchips, 
# from which you can build a logistic regression model.

# Start visualizing the data
path = os.getcwd() + '/../data/ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
data2.head()


# In[23]:

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')


# In[24]:

'''
This data looks a bit more complicated than the previous example. 
In particular, you'll notice that there is no linear decision boundary
that will perform well on this data. One way to deal with this using a linear technique 
like logistic regression is to construct features that are derived from polynomials of the original features.
Let's start by creating a bunch of polynomial features.
'''

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

data2.head()


# In[25]:

# modify the cost and gradient functions from part 1 to include the regularization term. First the cost function:
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


# In[26]:

'''
Notice the "reg" term in the equation. Also note the addition of a "learning rate" parameter. 
This is a hyperparameter that controls the effectiveness of the regularization term. 
Now we need to add regularization to the gradient function:
'''
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    
    return grad


# In[27]:

# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)


# In[28]:

# Let's initialize our learning rate to a sensible value. We can play with this later if necessary
# (i.e. if the penalization is too strong or not strong enough).
learningRate = 1


# In[29]:

costReg(theta2, X2, y2, learningRate)


# In[30]:

gradientReg(theta2, X2, y2, learningRate)


# In[31]:

result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
result2


# In[33]:

#  prediction function from part 1 to see how accurate our solution is on the training data.
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))


# In[34]:

###  implemented these algorithms from scratch, it's worth noting that we could also use a high-level python library 
### like scikit-learn to solve this problem.

from sklearn import linear_model
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())


# In[39]:


model.score(X2,  y2)


# In[ ]:




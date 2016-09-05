__author__ = "Nadimozzaman Pappo"
__github__ = "http://github.com/mnpappo"

"""
This scripts simulates(by 2d ploting) various activation functions used in ANN.
"""

import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import numpy as np

# sigmoid/logistic function
def sigmoid(a):
    x = T.dmatrix('x')
    s = 1/(1+T.exp(-x))
    logistic = theano.function([x], s) #performs elementwise operations
    result = logistic([a]) # we got the logistic value for every element in i
    return result

# sample input
a = np.arange(-10,10,.2)
result = sigmoid(a)
# A plot of the logistic function, with a on the x-axis and sigmoid(a) on
# the y-axis
# our logistic theano func retuns a 2d array/matrix thats why we neeed to
# flatten it to plot
plt.plot(a, np.array(result).flatten())
plt.title('logistic/sigmodid')
plt.grid(True)
plt.xlabel("input", color='red')
plt.xlabel("logistic output", color='red')
plt.show()

#

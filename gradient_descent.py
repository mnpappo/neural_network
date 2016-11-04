__author__ = "Nadimozzaman Pappo"
__github__ = "http://github.com/mnpappo"

"""
Gradient Descent used in ANN.
"""

import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

# some input sample
xi = np.arange(-100, 400, 0.1)

# given expression
x = T.dscalar('x')
exp1 = 5*x+2
fnexp1 = theano.function([x], exp1)

# hypothetical expression
m = T.scalar('m')
c = T.scalar('c')
exp2 = m*x+c
fnexp2 = theano.function([m, x, c], exp2)

# error function
error = T.sum(T.sqr(exp2 - exp1)) / (len(xi))

# gradient expression of error with respect to(wrt) m,c
gwrtm = T.grad(error, m)
gwrtc = T.grad(error, c)

# gradient value wrt m,c
gwrtmval = theano.function([m, x, c], gwrtm)
gwrtcval = theano.function([m, x, c], gwrtc)

# let initial m,c
m = 0
c = 0

# learning rate
lr = 0.0001
# empty list to save changes of m,c
fm = []
fc = []
# empty list to save m,c difference with actual m,c
mdiff = []
cdiff = []

# learn trough iterations
for i in xi:
    # update m,c simultenously
    m = m - (gwrtmval(m,  i, c) * lr)
    c = c - (gwrtcval(m, i, c) * lr)
    print (m, c)
    # save m and c for plotting
    fm.append(m)
    fc.append(c)
    # save m,c difference with actual values
    mdiff.append(5-m)
    cdiff.append(2-c)
    # print(mdiff, cdiff)

# plt.plot(fm, xi)
# plt.xlabel("values of m", color='red')
# plt.ylabel("inputs", color='red')
# plt.grid()
# plt.show()
#
# plt.plot(fc, xi)
# plt.xlabel("values of c", color='red')
# plt.ylabel("inputs", color='red')
# plt.grid()
# plt.show()

plt.plot(mdiff, xi)
plt.title("")
plt.xlabel("values of m", color='red')
plt.ylabel("inputs", color='red')
plt.grid()
plt.show()

plt.plot(cdiff, xi)
plt.xlabel("values of c", color='red')
plt.ylabel("inputs", color='red')
plt.grid()
plt.show()

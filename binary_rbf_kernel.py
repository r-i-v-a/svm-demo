#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from sklearn import svm

# training examples
X = np.array([[1,2], [1,5], [2,4], [4,0], [5,2], [3,7], [4,8], [6,5], [7,1], [9,7]])
Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# regularization parameter: try different values
c = 0.5

# fit support vector machine
clf = svm.SVC(kernel='rbf', C=c)
clf.fit(X, Y)

# grid for plotting
xx, yy = np.meshgrid(np.linspace(0, 10, 500),
                     np.linspace(0, 10, 500))

# plot the decision function for each point on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.imshow(
	Z, interpolation='nearest', 
	extent=(xx.min(), xx.max(), yy.min(), yy.max()), 
	aspect='auto', origin='lower', cmap=pl.cm.PuOr_r
)

# plot original data points
pl.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=pl.cm.Paired)

pl.axis([0, 10, 0, 10])
pl.title("non-linear with kernel, C = %.2f" % c)
pl.savefig("non_linear_kernel.png")
pl.show()
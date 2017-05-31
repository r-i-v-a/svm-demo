#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from sklearn import svm

# training examples
X = np.array([[1,2], [1,5], [2,5], [4,0], [5,2], [3,4], [4,8], [6,5], [7,1], [9,7]])
Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# regularization parameter: try different values
c = 0.5

# fit support vector machine
clf = svm.SVC(kernel='linear', C=c)
clf.fit(X, Y)

# get the decision boundary
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 10)
yy = a * xx - (clf.intercept_[0]) / w[1]

# get the margin around the decision boundary
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin

# plot the points + boundary
pl.clf()
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
pl.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=pl.cm.Paired)
pl.title("not linearly separable, C = %.2f" % c)
pl.savefig("non_separable.png")
pl.show()
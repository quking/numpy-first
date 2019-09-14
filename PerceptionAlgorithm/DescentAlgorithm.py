import numpy as np
import matplotlib.pyplot as plt


x_values = [1, 2, 3, 0.6, 3.6, 1.5, 3.1, 2.4, 1.8, 2.5, 2.6, 0.3, 1.5, 1.65, 2.6, 4.2, 1.5, 0.5, 2.13]
y_values = [0.3, 1.5, 2.2, 0.2, 3.2, 0.5, 2.8, 1.3, 1.5, 3, 3, 2.0, 2, 1.9, 3, 5, 3, 1, 4]

plt.scatter(x_values, y_values, s=100, color="blue")
plt.show()
m = 10
X0 = np.ones((m, 1))  # 为了对公式进行矩阵化，需要增加一个维度
X1 = np.array([x_values]).reshape(m, 1)
X = np.hstack((X0, X1))
y = np.array([y_values]).reshape(m, 1)
learnRate = 0.01


def gradient_function(X, theta, y):
    diff = np.dot(X*theta)-y
    return (1/m)*np.dot(X.T, diff)


def DesentProcess(X, y, rate):
    theta = np.array([1,1]).reshape(2,1)
    gradient = gradient_function(X,theta, y)
    while np.all(np.absolute(gradient) > 1e-5):
        theta = theta - rate*gradient
        gradient = gradient_function(X, theta, y)
    return theta




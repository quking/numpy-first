import numpy as np
import matplotlib.pyplot as plt

# 感知机算法实现

x_values = [1, 2, 3, 5, 0.6, 3.6, 1.5, 3.1, 2.4, 1.8, 2.5, 2.6, 0.3, 1.5, 1.65, 2.6, 4.2, 1.5, 0.5, 2.13]
y_values = [0.3, 1.5, 2.2, 0.3, 0.2, 3.2, 0.5, 2.8, 1.3, 1.5, 3, 3, 2.0, 2, 1.9, 3, 5, 3, 1, 4]
mark = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
plt.scatter(x_values[0:10], y_values[0:10], s=100, color="blue")
plt.scatter(x_values[10:20], y_values[10:20], s=100, color="red")
x = np.arange(0, 5, 0.1)
y = (8.7 / 10.2) * x + 5 / 10.2
plt.plot(x, y)
plt.show()


def trainPerceptron(dataMat, labelMat, eta):
    m, n = dataMat.shape
    weight = np.zeros(n)
    bias = 0
    flag = True
    while flag:
        for i in range(m):
            if np.all(labelMat[i] * (np.dot(weight, dataMat[i]) + bias) <= 0):
                weight = weight + eta * labelMat[i] * dataMat[i].T
                bias = bias + eta * labelMat[i]
                print("weight, bias: ", end="")
                print(weight, end="  ")
                print(bias)
                flag = True
                break
            else:
                flag = False

    return weight, bias


if __name__ == "__main__":
    datamat = []
    for i, j in zip(x_values, y_values):
        datamat.append([i, j])
    dataMat = np.array(datamat)
    k, b = trainPerceptron(dataMat, mark, 1)

    print(k, b)

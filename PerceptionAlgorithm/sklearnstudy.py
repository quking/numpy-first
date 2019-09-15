from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target
# print(X.shape, y.shape)
knn = KNeighborsClassifier(n_neighbors=1)
# print(knn)
knn.fit(X, y)
print(knn.predict([[1, 2, 3, 4]]))


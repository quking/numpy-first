from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
# print(X.shape, y.shape)
knn = KNeighborsClassifier(n_neighbors=5)
# print(knn)
knn.fit(X, y)
# y_predict = knn.predict(X)

# 准确率计算
# print(accuracy_score(y, y_predict))

# 分离测试数据和训练数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)  # 这里是把40%的数据拿来当测试数据集
# 下面可以看出
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train, y_train)
y_train_pre = knn_5.predict(X_train)
y_test_pre = knn_5.predict(X_test)
print(accuracy_score(y_train, y_train_pre))
print(accuracy_score(y_test, y_test_pre))


# 如何确定k的话，准确率最高
# k:1-25
# 遍历所有的可能参数组合
# 建立相应的model
# model进行训练和预测
# 给予测试数据的准确率计算
# 查看最高准确率对应的k值







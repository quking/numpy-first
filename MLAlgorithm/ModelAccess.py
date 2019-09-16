import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 这里只通过怀孕次数，胰岛素水平，体重指数，年龄四个特征来预测是否有糖尿病


pima = pd.read_csv("pima_data.csv")
feature_name = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_name]
y = pima.label
# print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 模型训练
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pre = logreg.predict(X_test)
# print(metrics.accuracy_score(y_test, y_pre))
# 确定正负样本的数据量
# print(y_test.value_counts())

# 混淆矩阵
print(metrics.confusion_matrix(y_test, y_pre))


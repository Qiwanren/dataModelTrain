from sklearn import  datasets

# 加载手写字体数据集
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

# 划分数据集
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=8)


from sklearn import svm

# SVM 高斯核, 可以将输入特征映射到无限维度中，缺点：计算速度慢，容易产生过拟合
svc = svm.SVC(C=1.0,kernel='rbf',gamma=0.001)

# 训练模型
svc.fit(X_train,y_train)

# 预测
y_pred = svc.predict(x_test)

## 准确率
print("SVM准确率 ：",accuracy_score(y_test,y_pred))


# SVM 多项式
svc = svm.SVC(C=1.0,kernel='poly',degree=3)

## 训练模型
svc.fit(X_train,y_train)

# 预测
y_pred = svc.predict(x_test)

# 用准确率进行评估
print("SVM(多项式)准确率 ：",accuracy_score(y_test,y_pred))
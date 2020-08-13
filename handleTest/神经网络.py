from sklearn import  datasets

# 加载手写字体数据集
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

# 划分数据集
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=8)

print(X_train)
print(y_train)

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

mlp = MLPClassifier(hidden_layer_sizes=(128,64),max_iter=50,alpha=2,solver='sgd')

# 训练模型
mlp.fit(X_train,y_train)

# 预测
y_pred = mlp.predict(x_test)

# 准确率
print("神经网络准确率为： ",accuracy_score(y_test,y_pred))
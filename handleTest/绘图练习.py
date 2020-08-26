import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

sns.set() #恢复seaborn的默认主题

def readData():
    from sklearn.datasets import load_iris

    # 加载数据
    iris = load_iris()
    # 创建X，y
    X ,y = iris.data,iris.target

    return X,y
def plot(X,y,title,x_label,y_label):
    target_names = {0:'setosa',1:'versicolor',2:'virginica'}
    #label_dict = {i:k for i,k in enumerate(['setosa' 'versicolor' 'virginica'])}
    label_dict = target_names
    for i in range(3):
        print(label_dict[i])
    ax = plt.subplot(111)
    for label,marker,color in zip(range(3),('^','s','o'),('blue','red','green')):
        plt.scatter(x=X[:,0].real[y == label],
                    y=X[:,1].real[y == label],
                    color=color,
                    alpha=0.5,
                    label=label_dict[label]
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    leg = plt.legend(loc='upper right',fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
    plt.show()

# 显示矩阵散点图
def analysisCenterForPCA(X,y):
    # 导入缩放模块
    from sklearn.preprocessing import StandardScaler
    # 中心化数据
    X_centered = StandardScaler(with_std=False).fit_transform(X)  # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
    #messagePrint(X)
    #messagePrint(X_centered)
    # 绘制中心化后的数据
    plot(X_centered, y,"Iris: Data Centered", "sepa1 1ength (crn) ", "sepa1 width(cm)")
if __name__ == '__main__':
    X,y = readData()
    analysisCenterForPCA(X,y)
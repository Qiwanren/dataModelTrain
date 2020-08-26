'''
LDA主要用于数据降维，但并不会计算整体数据的协方差姐阵的特征值，而是计算类内( within-class )和类问( between-class )散布矩阵的特征值和特征向量。
   LDA 分为5个步骤:
    (1) 计算每个类别的均值向量;
    (2) 计算类内和类间的散布矩阵;
    (3) 计算 S;'SB 的特征值和特征向量;
    (4) 降序排列特征值，保留前 个特征向量;
    (5) 使用前几个特征向量将数据投影到新空间。
  要求：用LDA拟合N个类别的数据，最多只需要N-1次切割

'''

import numpy as np
# 导入画图模式
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def messagePrint(x):
    print(x)
    print('----------------------------------------')

def readData():
    # 加载数据
    iris = load_iris()
    # 创建X，y
    X ,y = iris.data,iris.target

    #查看需要预测的花名
    #messagePrint(iris.target_names)
    return X,y,iris.target_names

# 显示散点图
def plot(X,y,title,x_label,y_label):
    target_names = {0:'setosa',1:'versicolor',2:'virginica'}
    #label_dict = {i:k for i,k in enumerate(['setosa' 'versicolor' 'virginica'])}
    label_dict = target_names
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

'''
第一步：计算每个类别的均值向量
    首先计算每个类别中每列的均值向量，分别是：setosa，versicolor，virginica:
'''
def stepOne(X,y):
    # 每个类别的均值向量,按列计算
    # 将鸢尾花数据集分为三块
    # 每块代表一种鸢尾花，计算均值
    mean_vectors = []
    labe1_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    for c1 in [0, 1, 2]:
        c1ass_mean_vector = np.mean(X[y == c1],axis = 0)
        mean_vectors.append(c1ass_mean_vector)
        print(labe1_dict[c1], c1ass_mean_vector)
    return mean_vectors
'''
第二步：计算类内和类间的散布矩阵
返回类内散布矩阵和类间散布矩阵
'''
def stepTwo(X,y,mean_vectors):
    # 类内散布矩阵（S_W）
    S_W = np.zeros((4, 4))
    # 对于每种聋儿花
    for cl, mv in zip([0,1, 2], mean_vectors):
        # 从 开始，每个类别的散布短阵
        class_sc_mat = np.zeros((4, 4))
        # 对于每个样本
        for row in X[y == cl]:
            #列向量
            row, mv = row.reshape(4, 1), mv.reshape(4, 1)
            # 4 x 的短阵
            class_sc_mat += (row - mv).dot((row - mv).T)
        # 散布矩阵的和
        S_W+=class_sc_mat

    # 类问散布矩阵
    # 数据集的均值
    overall_mean = np.mean(X,axis = 0).reshape(4, 1)
    # 会变成散布矩阵
    S_B = np.zeros((4, 4))
    for i,mean_vec in enumerate(mean_vectors):
        # 每种花的数量
        n = X[y == i,:].shape[0]
        # 每种花的列向量
        mean_vec = mean_vec.reshape(4, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_W,S_B

# 通过sklean实现LDA
def sklean_lda(X,y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #拟合并转换原始的莺尾花数据，绘制新的投影 ，以便和 PCA 的结果进行比较
    # 实例化LDA模块
    lda = LinearDiscriminantAnalysis(n_components=2)
    # 拟合并转化鸢尾花数据
    x_lda_iris = lda.fit_transform(X,y)
    # 绘制投影数据
    plot(x_lda_iris,y,"LDA Projection","LDA1","LDA2")

if __name__ == '__main__':
    X,y,target_names = readData()
    #print(target_names)
    #mean_vectors = stepOne(X,y)
    sklean_lda(X,y)

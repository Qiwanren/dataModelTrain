'''
PCA :利用了协方差矩阵的特征值分解，对数据进行降维处理
    原理：PCA为非监督型算法，试图用数据最主要的若干方面来代替原有的数据，这些最主要的方面首先需要保证蕴含了原始数据中的大量信息，
         其次需要保证相互之间不相关。因为相关代表了数据在某种程度上的“重叠”，也就相当于冗余性没有清除干净

    对象：主要针对定量数据分析
    适用场景：数据量大指数据记录多和维度多两种情况，PCA对大型数据集的处理效率高

    方法：
        1、pca = PCA(n_components=2)   PCA初始化，n_components设置所选特征方差最高的前几位,保留前N个最大的特征值对应的特征向量
        2、fit :数据拟合
        2、transform 方法：将数据投影到新的二维平面上
    参数解释：
        1、n_components:  我们可以利用此参数设置想要的特征维度数目，可以是int型的数字，也可以是阈值百分比，如95%，让PCA类根据样本特征方
            差来降到合适的维数，也可以指定为string类型，MLE。
        2、copy： bool类型，TRUE或者FALSE，是否将原始数据复制一份，这样运行后原始数据值不会改变，默认为TRUE。
        3、whiten：bool类型，是否进行白化（就是对降维后的数据进行归一化，使方差为1），默认为FALSE。如果需要后续处理可以改为TRUE。
        4、explained_variance_： 代表降为后各主成分的方差值，方差值越大，表明越重要。
        5、explained_variance_ratio_： 代表各主成分的贡献率。
        6、inverse_transform()： 将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)。

'''
import numpy as np

# 导入画图模式
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# 导入缩放模块
from sklearn.preprocessing import StandardScaler


def messagePrint(x):
    print(x)
    print('----------------------------------------')

def readData():
    from sklearn.datasets import load_iris

    # 加载数据
    iris = load_iris()
    # 创建X，y
    X ,y = iris.data,iris.target

    #查看需要预测的花名
    #messagePrint(iris.target_names)
    return X,y

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
    使用均方差和碎石图完成特征筛选和分析
'''
def analysisDataByCovAndSst(X,y):
    #计算均值向量
    mean_vector = X.mean(axis = 0)
    # 计算协方差矩阵
    cov_mat = np.cov((X).T)
    #messagePrint(cov_mat.shape)
    # 计算鸢尾花数据集的特征向量和特征值
    eig_val_cov ,eig_vec_cov = np.linalg.eig(cov_mat)
    # 按降序打印特征向量和相应的特征值
    for i in range(len(eig_val_cov)):
        eigvec_cov = eig_vec_cov[:,i]
        #print('Eigenvector {}: \n{}'.format(i+1,eigvec_cov))
        #print('Eigenvalue {} from covariance matrix:{} '.format(i+1,eig_val_cov[i]))
        #messagePrint(30 * '-')
    '''
    使用碎石图分析特征
        原理：碎石图是一种简单的折线图，显示每个主成分解释数据总方差的百分比，绘制碎石图的时候需要对特征值进行降序排列，
              绘制每个主成分和之前所有主成分方差的和
              图中的数据点代表一个主成分，每个主成分解释了总方差的某个百分比，相加后，所有的主成分应该解释了数据集中的总方差的100%
              根据特征所占百分比，选出特征值（本例中：[0.92461872 0.05306648 0.01710261 0.00521218]  前两个主成分占比就将近达到98%，可将
              特征从四个减为两个）
    '''
    # 每个主成分解释的百分比是特征值除以特征之和
    explained_variance_ratio = eig_val_cov/eig_val_cov.sum()
    messagePrint(explained_variance_ratio)
    # 绘制碎石图
    plt.plot(np.cumsum(explained_variance_ratio))
    plt.title('Scree Plot')
    plt.xlabel('Principal Component (k)')
    plt.ylabel('% of Variance Explained <= k')
    #plt.show()
    '''
    
    '''
    # 保存两个特征向量
    top_2_eigenvectors = eig_vec_cov[:,:2] # 选取前两个特征数据
    # 转置，每行是一个主成分，两行代表两个主成分
    messagePrint(top_2_eigenvectors.shape)
    messagePrint(top_2_eigenvectors)
    # 数组代表了两个特征向量
    '''
    将获得的top_2_eigenvectors矩阵和原始数据（X）相乘，获得新的数据集(new_X),并用新获得的数据代替原来的数据集X
    备注：第一个矩阵的列数必须与第二个矩阵的行数相同
    '''
    new_X = np.dot(X,top_2_eigenvectors)
    new_X1 = np.dot(X-mean_vector,top_2_eigenvectors)  ## 对数据先进行中心话操作，类似PCA
    messagePrint(new_X)
'''
使用PCA完成analysisDataByCovAndSst的分析过程
'''
def analysisDataByPCA(X,y):
    # 导入PCA
    from sklearn.decomposition import PCA
    # 为模仿鸢尾花数据集的操作过程，实例化有两个组件的PCA对象
    pca = PCA(n_components=2)
    # 用PCA拟合数据
    pca.fit(X)
    # 查看PCA的属性，看看是不是和手动计算的结果匹配，
    messagePrint(pca.components_)
    # 使用PAC对象的transform方法，将数据投影到新的二维平面上,此处因为PCA会将数据进行中心化，所以和手动的可能不一样
    pca.transform(X)[:5,]
    # 绘制图像，比较差异
    plot(X, y, "OriginalIrisData", "sepallength", "sepalwidth")
    plot(pca.transform(X), y, "Iris : Data projected onto first two PCA components", "PCA1", "PCA2")

    # 提取每个主成分解释的方差量
    #pca.explained_variance_ratio_

'''
使用PCA优化模型
'''
# 获取数据集中列之间的相关性系数的平均数
def getFeatureCorr(X):
    # 首先计算莺尾花数据集的相关矩阵,矩阵每个值代表两个特征间的相关性系数:
    #X1 = np.corrcoef(X.T)  # 矩阵中值为 1，表示自身与自身的相关性
    # 我们提取对角线上的 1，计算特征间的平均相关性:
    #X2 = np.corrcoef(X.T)[[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]]
    # 获取原始数据集的均值
    #x_mean = np.corrcoef(X.T)[[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]].mean()

    return np.corrcoef(X.T)[[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]].mean()

def downCorrByPCA(X):
    #messagePrint(X)
    # 取所有主成分
    full_pca = PCA(n_components=4)  ## 保留下来的特征数
    # PCA 拟合数据集
    full_pca.fit(X)     ##  用数据X来训练PCA模型
    #仍然后用老办法计算(应该是线性独立的)新列的平均相关系数:
    pca_iris = full_pca.transform(X)  # 将数据X转换成降维后的数据，当模型训练好后，对于新输入的数据，也可以用transform方法来降维
    #fit_transform(X)：用X来训练PCA模型，同时返回降维后的数据。
    #inverse_transform(pca_iris) ：将降维后的数据转换成原始数据，但可能不会完全一样，会有些许差别。
    # PCA 后的平均相关系数
    messagePrint(pca_iris)
    messagePrint(full_pca.explained_variance_ratio_)  # 保留下来的各个特征的方差百分比
    #return getFeatureCorr(pca_iris)
'''
分析中心化对PCA的影响
'''
def analysisCenterForPCA(X,y):
    # 中心化数据
    X_centered = StandardScaler(with_std=False).fit_transform(X)  # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本你
    #messagePrint(X)
    #messagePrint(X_centered)
    # 绘制中心化后的数据
    plot(X_centered, y,"Iris: Data Centered", "sepa1 1ength (crn) ", "sepa1 width(cm)")
    full_pca = PCA(n_components=2)  ## 保留下来的特征数
    full_pca.fit(X_centered)
    messagePrint(full_pca.components_)
    # 查询拟合筛选后的特征，
    # PCA中心化后的数据图,数据分类更加明确
    plot(full_pca.transform(X_centered),y,'Iris : Data projected onto first two PCA components with centered data','PCA1','PCA2')
    # 每个主成分解释方差的百分比
    messagePrint(full_pca.explained_variance_ratio_)

'''
主成分分析流程总结：深入主成分分析
    
'''

def deepPCA(X,y):
    pca = PCA(n_components=2)  ## 保留下来的特征数
    # 数据拟合
    pca.fit(X)
    #messagePrint(pca.explained_variance_ratio_)  # 0.92461872 0.05306648 0.01710261 0.00521218  选取前两个特征
    #messagePrint(pca.components_.T)
    #将原始矩阵 (150 x 4) 和转置主成分矩阵(4 * 2)相乘,得到投影数据 （150 x 2 )
    # 对原始数据进行z分数缩放
    x_scaled = StandardScaler().fit_transform(X)
    X1 = np.dot(x_scaled,pca.components_.T)[:5,]
    #messagePrint(X1)
    # 删除后两个特征
    iris_2_dim = X[:,2:4]
    # 中心花
    iris_2_dim = iris_2_dim - iris_2_dim.mean(axis = 0)
    plot(iris_2_dim,y,"Iris: only 2 dimensions","sepal length","sepal width")

    # 实例化保留两个主成分的PCA
    twodim_pca = PCA(n_components=2)
    # 拟合并转化截断的数据
    iris_2_dim_transformed = twodim_pca.fit_transform(iris_2_dim)
    plot(iris_2_dim_transformed,y,"Iris: PCA performed on only 2 dimensions","PCA1","PCA2")


def elevateModelByPca(X,y):
    pass


if __name__ == '__main__':
    X,y = readData()
    #X_corr_mean = getFeatureCorr(X)
    #X_corr_mean = downCorrByPCA(X)
    #print(X)
    #analysisCenterForPCA(X,y)
    deepPCA(X,y)


'''
特征工程

'''

def readData():
    from sklearn.datasets import load_iris
    # 导入IRIS数据集
    iris = load_iris()
    # 特征矩阵
    iris.data
    # 目标向量
    iris.target
    return iris.data,iris.target

# 对定性特征哑编码
def one_hot_code(X,y):
    from sklearn.preprocessing import OneHotEncoder
    # 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
    rs = OneHotEncoder().fit_transform(y.reshape((-1, 1)))
    print(y)
    print(type(rs))

if __name__ == '__main__':
    X,y = readData()
    one_hot_code(X,y)
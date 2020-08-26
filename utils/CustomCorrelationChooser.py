from sklearn.base import TransformerMixin,BaseEstimator
import pandas as pd
import numpy as np
'''
    通过corr函数筛选出两列之间相关性系数的绝对值大于某个阈值的字段名称
'''
class CustomCorrelationChooser(TransformerMixin,BaseEstimator):
    def __init__(self,response,cols_to_keep=[],threshold=None):
        # 保存响应变量
        self.response = response
        # 保存阈值
        self.threshold = threshold
        # 初始化一个变量，存放要保留的列名
        self.cols_to_keep = cols_to_keep

    def transform(self,X):
        # 转换-选择合适的列
        return X[self.cols_to_keep]

    def fit(self,X,*_):
        # 创建新的DataFrame,存放特征和响应
        df = pd.concat([X,self.response],axis=1)
        # 保存高于阈值的列的名称
        self.cols_to_keep = df.columns[df.corr()[df.columns[-1]].abs() > self.threshold]
        # 只保留X的列，删除响应变量
        self.cols_to_keep = [c for c in self.cols_to_keep if c in X.columns]
        return self

def method01():
    np.random.seed(123)
    path = 'D:/data/python/test/credit_card_default.csv'

    # 导入数据集
    credit_card_default = pd.read_csv(path)
    # 进行基础数据的信息分析
    print(credit_card_default.shape)
    #messagePrint(credit_card_default.describe().T)
    # 特征矩阵
    X = credit_card_default.drop('default payment next month',axis=1)
    # 响应变量
    y = credit_card_default['default payment next month']
    # 输出空准确率
    #print(y.value_counts(normalize=True))
    return X,y
if __name__ == '__main__':
    x,y = method01()
    print(x.info())
    ccc = CustomCorrelationChooser(threshold=0.2,response=y)
    ccc.fit(x)
    print(ccc.cols_to_keep)
    print(ccc.transform(x).head())

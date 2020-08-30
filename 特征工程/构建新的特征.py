'''
数据分类
   分类
        定类数据
        定序数据
        定距数据
        定比数据
  实例：
        boolean : 布尔型，此列是二元分类数据（是否）,定类数据
        city : 此列是分类数据，定类数据
  处理方法
        1、定类数据：
            a、用pandas自动找到分类变量并进行编码
            b、创建自定义虚拟变量编码器，在流水线中工作
        2、连续特征分箱
            可以借助Pandas中的cut函数，
            pd.cut(X['feature1'],bins=3,labels=False)  # 可设置也可不设置

'''
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

def messagePrint(x):
    print(x)
    print('------ -------------------------------------')


# 自定义虚拟变量
class CustomDummifier(TransformerMixin):
    def __init__(self,cols=None):
        self.cols = cols
    def transform(self,X):
        return pd.get_dummies(X,columns=self.cols)
    def fit(self,*_):
        return self

cd = CustomDummifier(cols=['boolean','city'])
#cd.fit_transform(X)

# 连续特征分箱
class CustomCutter(TransformerMixin):
    def __init__(self,col,bins,labels=False):
        self.labels = labels
        self.bins = bins
        self.col = col
    def transform(self,df):
        X = df.copy()
        X[self.col] = pd.cut(X[self.col],bins=self.bins,labels=self.labels)
        return X
    def fit(self,*_):
        return self
cc = CustomCutter(col='feature1',bins=3)
cc.fit_transform(X)
























def method01():
    X = pd.DataFrame({'city': ['tokyo', None, 'london', 'seattle', 'san francisco', 'tokyo'],
                      'boolean': ['yes', 'no ', None, 'no', 'no', 'yes '],
                      'ordinal_column': ['somewhat like', 'like', 'somewhat like', 'like', 'somewhat like', 'dislike'],
                      'quantitative_column': [1, 11, -.5, 10, None, 20]})

    # 默认的类别名就是分箱
    x1 = pd.cut(X['quantitative_column'], bins=3, labels=False)
    print(x1)

'''
    通过该方法获取最优的多项式特征组合
'''
def method02():
    trainFilePath = 'D:/data/python/test/1.csv'
    all_params = ['x', 'y', 'z', 'activity']
    train = pd.read_csv(filepath_or_buffer=trainFilePath, names=all_params, encoding='utf-8')

    X = train[['x', 'y', 'z']]
    y = train['activity']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

    '''
    X_poly = poly.fit_transform(X)
    messagePrint(X_poly.shape)
    '''

    # X1 = pd.DataFrame(X_poly,columns=poly.get_feature_names())
    # print(X1.head())

    # 设置需要试验的KNN模型参数
    knn_params = {'n_neighbors': [3, 4, 5, 6]}
    knn = KNeighborsClassifier()

    # 设置流水线
    pipe_params = {'poly_features__degree': [1, 2, 3], 'poly_features__interaction_only': [True, False],
                   'classify__n_neighbors': [3, 4, 5, 6]}
    # 实例化流水线
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('poly_features', poly), ('classify', knn)])

    # 设置网格搜索，输出最佳准确率和学习到的参数
    grid = GridSearchCV(pipe, pipe_params)
    grid.fit(X, y)

    ## 输出最佳准确率和学习到的学习率
    print(grid.best_score_, grid.best_params_)

if __name__ == '__main__':
    method02()
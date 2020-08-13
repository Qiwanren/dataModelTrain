# 在环境变量中加入安装的Graphviz路径
import os
os.environ["PATH"] += os.pathsep + 'D:/anzhuang/graphviz-2.38/bin'

from sklearn import datasets

'''
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
cancer_names = cancer['feature_names']

X_train,X_test,y_train,t_test = train_test_split(X,y,test_size=0.2,random_state=8)

train_xgb = xgb.DMatrix(X_train,y_train)
test_xgb = xgb.DMatrix(X_test)

params = {
    'booster': 'gbtree',   ####  gbtree   gblinear
    'objective': 'binary:logistic',  # 多分类的问题  'objective': 'binary:logistic' 二分类，multi:softmax 多分类问题
    'gamma': 0.01,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,               # 构建树的深度，越大越容易过拟合
    'min_child_weight': 1.5,
    'eta': 0.01,                  # 如同学习率
    'learning_rate':0.1,
    'subsample':0.5,
    'colsample_bytree':1.0,
    'reg_alpha':1.0
}

model = xgb.train(dtrain=train_xgb,params=params)
y_pred = model.predict(test_xgb)
'''
def method1():
    import pandas as pd
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data)

    # 生成特征名map文件；注意：变量名中不能带有空格；i代表indicator数据类型，q代表quantity数据类型
    X.columns = pd.Series(cancer.feature_names).str.replace(' ', '_')

    def create_feature_map(features):
        outfile = open('clf.fmap', 'w')
        for i, f in enumerate(features):
            outfile.write('{0}\t{1}\tq\n'.format(i, f))
        outfile.close()

    create_feature_map(X.columns)

    # 训练模型
    model = xgb.XGBClassifier(n_estimators=5)
    model.fit(X, cancer.target)

    digraph = xgb.to_graphviz(model, num_trees=1)
    digraph.format = 'png'
    digraph.view('./iris_xgb')
    # 模型树可视化
    '''
    xgb.plot_tree(model, num_trees=0, fmap='clf.fmap')
    fig = plt.gcf()
    fig.set_size_inches(120,120)
    fig.savefig('tree.png')
    '''
import xgboost as xgb
from sklearn.datasets import load_iris
def method2():
    iris = load_iris()
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(iris.data, iris.target)
    xgb.to_graphviz(xgb_clf, num_trees=1)
    digraph = xgb.to_graphviz(xgb_clf, num_trees=1)
    digraph.format = 'png'
    digraph.view('./iris_xgb')

if __name__ == '__main__':
    method2()
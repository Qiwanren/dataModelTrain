'''
使用PCA和LDA分析提升鸢尾花数据集

'''
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
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

def getBasicKnnRate(iris_X,iris_y):
    # 创建三个变量，其中一个代表LDA，一个代表PCA，最后一个代表KNN模型
    # 创建一个主成分的PCA模块
    single_pca = PCA(n_components=1)
    # 创建有一个判别式的LDA模块
    single_lda = LinearDiscriminantAnalysis(n_components=1)

    # 实例化KNN模型
    knn = KNeighborsClassifier(n_neighbors=3)

    # 获取基准准确率
    knn_average = cross_val_score(knn, iris_X,iris_y).mean()
    messagePrint(knn_average)

    # 使用LDA，只保留最好的线性判别式：
    lda_pipeline = Pipeline([('lda',single_lda),('knn',knn)])
    lda_average = cross_val_score(lda_pipeline,iris_X,iris_y).mean()
    messagePrint(lda_average)

    # 创建执行PCA的流水线
    pca_pipeline = Pipeline([('pca',single_pca),('knn',knn)])
    pca_average = cross_val_score(pca_pipeline,iris_X,iris_y).mean()
    messagePrint(pca_average) # 表现较差，针对类别可分性，LDA优于PCA

    # 使用两个LDA判别式，查看准确率的变化
    lda_pipeline = Pipeline([('lda',LinearDiscriminantAnalysis(n_components=2)),('knn',knn)])
    lda_average = cross_val_score(lda_pipeline,iris_X,iris_y).mean()
    messagePrint(lda_average)  #准确率有所提升

    # 导入SelectKBest模块，尝试选择优化特征
    # 用特征选择工具和特征转换工具做对比
    from sklearn.feature_selection import SelectKBest
    # 尝试所有K值，但是不包括全部保留
    for k in [1,2,3]:
        # 构建流水线
        select_pipeline = Pipeline([('select',SelectKBest(k=k)),('knn',knn)])
        # 交叉验证流水线
        select_average = cross_val_score(select_pipeline,iris_X,iris_y).mean()
        print(k,"best feature has accuracy:",select_average)

def get_best_model_and_accuracy(name,model,params,X,y):
    grid = GridSearchCV(model, # 要搜索的模型
                        params, # 要尝试的参数
                        error_score=0 # 如果报错，则结果为零
                        )
    grid.fit(X,y) # 拟合模型和参数
    # 经典的性能指示
    print(name + ' - Best Accuracy : {}'.format(grid.best_score_))
    # 得到最佳准确率的最佳参数
    print(name + ' - Best Parameters : {}'.format(grid.best_params_))
    # 拟合的平均时间（秒）
    print(name + " - Average Time to Fit (s) ：{}".format(round(grid.cv_results_['mean_fit_time'].mean(),3)))
    # 预测的平均时间 （秒）,从该指标可以看出模型在真实世界的性能
    print(name + " - Average Time to Score (s) : {}".format(round(grid.cv_results_['mean_score_time'].mean(),3)))

'''
使用缩放，PCA，LDA和KNN对流水线进行测试
'''
def gridSerarchHandle(x,y):
    iris_params = {
        'preprocessing__scale__with_std':[True,False],
        'preprocessing__scale__with_mean':[True,False],
        'preprocessing__pca__n_components':[1,2,3,4],
        # 根据scikit-learn文档，LDA的最大n_components是类别数减1
        'preprocessing__lda__n_components':[1,2],
        'clf__n_neighbors':range(1,9)
    }
    # 更大的流水线
    preprocessing = Pipeline([('scale',StandardScaler()),
                              ('pca',PCA()),
                              ('lda',LinearDiscriminantAnalysis())])
    iris_pipeline = Pipeline(steps=[('preprocessing',preprocessing),
                                    ('clf',KNeighborsClassifier())])
    get_best_model_and_accuracy('综合测试',iris_pipeline,iris_params,x,y)

if __name__ == '__main__':
    x,y = readData()
    getBasicKnnRate(x,y)
    #gridSerarchHandle(x,y)
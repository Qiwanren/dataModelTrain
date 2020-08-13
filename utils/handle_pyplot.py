import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import xgboost as xgb

def showZft(data,feature):
    # 设置matplotlib正常显示中文和负号
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    # 随机生成（10000,）服从正态分布的数据
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(data, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel("用户数")
    # 显示图标题
    plt.title(feature+" 分布直方图")
    plt.show()

##绘制决策树
def showTree(model):

    xgb.to_graphviz(model, num_trees=1)
    digraph = xgb.to_graphviz(model, num_trees=1)
    digraph.format = 'png'
    digraph.view('./iris_xgb')

if __name__ == '__main__':
    data = np.random.randn(10000)

    store = pd.DataFrame([['Snow', 'M', 22], [0.0, 11.0, 'unknow'], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                       columns=['name', 'gender', 'age'])
    feature = 'gender'
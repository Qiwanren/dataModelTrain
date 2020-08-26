import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import seaborn as sns ## 另一个数据可视化工具


'''
    绘制直方图
'''
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

'''
输入数据为一个Series
'''
def showZft2(data):
    data.plot.hist(grid=True,bins=20,rwidth=0.9,color='#607c8e')
    plt.title('Commute Times for 1000 Commuters')
    plt.xlabel('Counts')
    plt.ylabel('Commute Time')
    plt.grid(axis='y',alpha=0.75)
    plt.show()


'''
绘制xgboost模型决策树：
    model为模型
'''
def showTree(model):
    xgb.to_graphviz(model, num_trees=1)
    digraph = xgb.to_graphviz(model, num_trees=1)
    digraph.format = 'png'
    digraph.view('./iris_xgb')
'''
生成折线图
    x,y为数据list
'''
def showZxt(x,y):
    # 处理乱码
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    # "r" 表示红色，ms用来设置*的大小
    plt.plot(x, y, "r", marker='*', ms=10, label="a")
    plt.xticks(rotation=45)
    plt.xlabel("x轴")
    plt.ylabel("y轴")
    plt.title("数据显示折线图")
    plt.legend(loc="upper left")
    # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    for x1, y1 in zip(x, y):
        plt.text(x1, y1 + 1, str(y1), ha='center', va='bottom', fontsize=20, rotation=0)
    #plt.savefig("a.jpg")
    plt.show()

'''
    显示直方图：
        data为一个dataFrame
        f1为特征1
        f2为特征2
'''
def showZft2(data,f1,f2):
    data[f1].hist()
    # 绘制直方图
    data[f1].hist(by=data[f2],
                  sharex=True,
                  sharey=True,
                  figsize=(10, 10),
                  bins=20)

'''
显示数据热力图：
    值越接近1，则两者之间的正关联性越强，值越接近-1，则两者之间的负关联性越强，为零则是无相关性
    输入项为一个DataFrame
'''
def draw_heatmap(data):
    ylabels = data.columns.values.tolist()
    ss = StandardScaler()  # 归一化
    data = ss.fit_transform(data)
    df = pd.DataFrame(data)
    dfData = df.corr()
    plt.subplots(figsize=(15, 15))  # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, yticklabels=ylabels, xticklabels=ylabels, cmap="RdBu")
    #plt.savefig('../img/thermodynamicDiagram.jpg')
    plt.show()

# 输入参数为一个dataFrame，显示数据热力图
def showheatMap(df):
    import seaborn as sns
    import matplotlib.style as style
    # 选用一个干净的主题
    style.use('fivethirtyeight')
    sns.heatmap(df.corr())
    plt.show()

'''
绘制分类数据的散点图
    输入参数：特征参数，y：类别结果，
    预先处理：本样例中target_name的类别有三个分别为：setosa ，versicolor ，virginica
               y值有 0,1,2
              实际应用中需要根据实际数据对target_name进行处理
'''
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

if __name__ == '__main__':
    data = np.random.randn(10000)

    store = pd.DataFrame([['Snow', 'M', 22], [0.0, 11.0, 'unknow'], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                       columns=['name', 'gender', 'age'])

    x = [1, 2, 3, 4]
    y = [10, 50, 20, 100]
    showZft2()

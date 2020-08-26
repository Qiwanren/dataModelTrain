import csv

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
# 划分数据集
from sklearn.model_selection import train_test_split
from utils.handle_pyplot import showTree, draw_heatmap, showZft

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 定义全局变量
#nums_params = ['cert_age', 'fj_arpu', 'total_flux', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'total_flow', 'active_days']

le = LabelEncoder()
from time import strftime, localtime

# 打印当前时间
def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return
def messagePrint(x):
    print(x)
    print('----------------------------------------')

def dropExcepRows(data):
    # 删除不满足的列
    product_type_values = ['2I', 'bjl', '5G', 'other']
    data = data.drop(data[~data.product_type.isin(product_type_values)].index)
    return data
def is_number(num):
    from builtins import str
    num =str(num)
    strs = num.split('.')
    flag = False
    if len(strs) > 1:
        strs[1] = strs[1][0:3]
    for s in strs:
        if s.strip().isnumeric() == False:
            flag = False
            break
        else:
            flag = True
    return flag
## 处理省分ID字段,填补零
def handleProvID(x):
    if is_number(x):
        return int(x)
    return None
def handleServiceType(x):
    if x == '40AAAAAA':
        return 4
    elif x == '50AAAAAA':
        return 5
    elif x == '90AAAAAA':
        return 9
    else:
        return None

## 处理类别对象，返回整数handleTypeFlag
def handleTypeFlag(df):
    df['flag'] = df['flag'].map(lambda x:getIntNumber(x))
    type_value = df['flag'].value_counts().index[0]
    df['flag'].fillna(type_value, inplace=True)
    return df

def getIntNumber(x):
    if x != 1.0 and x != 0.0:
        return None
    else:
        strs = str(x).split('.')
        return int(strs[0])

def getIntNumberData(x):
    if is_number(x):
        strs = str(x).split('.')
        return strs[0].zfill(2)
    else:
        return None

def getDataValue(x):
    from builtins import str
    if is_number(x):
        num = str(x)
        strs = num.split('.')
        len_num = 0
        if len(strs[1]) <= 4:
            len_num = len(strs[0]) + len(strs[1]) + 1
        else:
            len_num = len(strs[0]) + 4
        return float(num[0:len_num])
    else:
        #print(x)
        return None

## 检查数据中的异常值
def unusualValueForCol(data):
    # 年龄数据均为大于零的数据值
    features1 = ['cert_age','fj_arpu']
    for f in features1:
        data[f] = data[f].map(lambda x: x if x > 0 else None)
    # 流量及语音数据进行单位转换
    features2 = ['visit_dura']
    features3 = ['up_flow', 'down_flow','total_flux']
    # 将日期数据转换以小时为单位的数据值
    for f1 in features2:
        data[f1] = data[f1].map(lambda x:x/60/60 if x>0 else None)
    # 将流量数据转换为G为单位的数据值
    for f3 in features3:
        data[f3] = data[f3].map(lambda x:x/1024/1024 if x>0 else None)
    return data

# 使用占比较多的类别，填补类别缺失值
def fileDeficiencyValue1(data):
    # 处理service_type字段
    data['service_type'] = data['service_type'].apply(lambda x:handleServiceType(x))
    # 处理以数字为类别的特征
    type_feature = ['prov_id', 'cust_sex', 'brand_flag', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag','service_type']
    # 对定性数据，进行填充，填充值为占比最大的类别
    for f in type_feature:
        type_value = data[f].value_counts().index[0]
        data[f].fillna(type_value, inplace=True)

    # 为省分值数据，若缺零，则进行填补
    data['prov_id'] = data['prov_id'].apply(lambda x:handleProvID(x))
    return data

def handleTypeFeatureToInt(data):
    type_feature1 = ['cust_sex', 'brand_flag', 'heyue_flag', 'is_limit_flag', '5g_flag','service_type']
    for f in type_feature1:
        data[f] = data[f].apply(lambda x: getIntNumberData(x))
    return data

# 使用均值填充缺失的数据，同时对数据进行归一化处理
def fileDeficiencyValue2(data):
    nums_params = ['cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_dura',
                   'roam_dura', 'total_times', 'total_nums', 'local_nums', 'roam_nums', 'in_cnt', 'out_cnt',
                   'in_dura', 'out_dura', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'total_flow',
                   'active_days', 'imei_duration', 'avg_duratioin']

    # 单独处理avg_duratioin字段
    data['avg_duratioin'] = data['avg_duratioin'].apply(pd.to_numeric, errors='coerce').fillna(13.6)


    # 处理数值型特征
    for param in nums_params:
        data[param] = data[param].apply(lambda x: getDataValue(x))

    ## 使用均值填充定量数据的缺失字段
    for param in nums_params:
        avg_n = data[param].mean()
        #data[param] = data[param].apply(pd.to_numeric, errors='coerce').fillna(avg_n)
        data[param].fillna(avg_n,inplace=True)
        data[param] = data[param].round(2)

    # 通过min_max方式对数据进行归一化处理
    data[param] = data[param].map(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data

def handleTypeFeature(data):
    type_feature = ['cust_sex', 'brand_flag', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag','service_type']  # app_type_id
    for f in type_feature:
        dummies = pd.get_dummies(data[f], prefix=f)
        data = pd.concat([data, dummies], axis=1)
    # 删除完成编码的特征值
    for f1 in type_feature:
        data = dropFeature(data, f1)
    return data

## 删除完成one-hot编码的特征
def dropFeature(dataF,features):
    dataF.drop(features, axis=1, inplace=True)  # inplace=True, 直接从内部删除
    return dataF

printTime()

def readData(path,labels):
    all_params = ['prov_id','user_id','cust_id','product_id','area_id','device_number','cust_sex','cert_age','total_fee','jf_flux','fj_arpu',
                  'ct_voice_fee','total_flux','total_dura','roam_dura','total_times','total_nums','local_nums','roam_nums','in_cnt','out_cnt',
                  'in_dura','out_dura','heyue_flag','is_limit_flag','product_type','5g_flag','visit_cnt','visit_dura','up_flow','down_flow',
                  'total_flow','active_days','brand','brand_flag','brand_detail','imei_duration','avg_duratioin','service_type']  # ,'app_type_id','app_visit_dura'

    if len(labels):
        train = pd.read_csv(filepath_or_buffer=path, sep=",", names=all_params + labels, encoding='utf-8')
    else:
        train = pd.read_csv(filepath_or_buffer=path, sep=",", names=all_params,quoting=csv.QUOTE_NONE, encoding='utf-8')
    return train

def dataHandle(dataset):
    # 处理service_type字段
    # 处理异常的数据
    dataset = unusualValueForCol(dataset)
    # 处理定性数据
    dataset = fileDeficiencyValue1(dataset)
    # 处理类别数据值中的小数部分
    dataset = handleTypeFeatureToInt(dataset)
    # 处理定量数据
    dataset = fileDeficiencyValue2(dataset)
    # 对定性数据进行编码
    dataset = handleTypeFeature(dataset)
    return dataset


def xgboostModelTrain(x_train,y_train,x_test,y_test):
    # 生成DMatrix,字段内容必须为数字或者boolean
    gb_train = xgb.DMatrix(x_train, y_train)
    xgb_test = xgb.DMatrix(x_test)
    ## 定义模型训练参数
    params = {
        'booster': 'gbtree',  ####  gbtree   gblinear
        'objective': 'binary:logistic',  # 多分类的问题  'objective': 'binary:logistic' 二分类，multi:softmax 多分类问题
        'gamma': 0.3,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 7,  # 构建树的深度，越大越容易过拟合
        'min_child_weight': 5,
        'eta': 0.15,  # 如同学习率
        'learning_rate': 0.08,
        'subsample': 0.5,
        'lambda':1.5,   # 默认值为1,  权重的L2正则化项 增大该值可以防止过拟合
        'colsample_bytree': 0.77,
        'reg_alpha': 1.0
    }

    ## 训练轮数
    num_rounds = [215, 300, 400, 500]
    num_round = 500
    print(num_round)
    printTime()
    print('--- begin to train model -----')
    ## 模型训练
    model = xgb.train(params, gb_train, num_round)
    ## 特征拟合
    #model.fit(x_train,y_train)
    # showTree(model)

    ## 分析特征值
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    ans = model.predict(xgb_test)
    #print('预测值AUC为 ：%f' % roc_auc_score(y_test, ans))
    # 显示重要特征
    plot_importance(model)
    plt.show()
    printTime()
    return ans

def writeDataToCsv(df,ans,path):
    df['score'] = ans
    # 对结果归一化处理
    df['score1'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    # 将结果输出到文件
    df.to_csv(path)

def getModelResult(trainFilePath,testFilePath):
    labels = ['flag']
    label = 'flag'
    filename = testFilePath[-7:-4]

    # 读取数据
    train = readData(trainFilePath, labels)
    test = readData(testFilePath, '')

    # 基本处理，删除异常行数据
    train = dropExcepRows(train)
    test = dropExcepRows(test)

    # 处理标签数据
    train = handleTypeFlag(train)
    y_train = train[label]
    # test = handleTypeFlag(test)
    # y_test = test['flag']

    # 备份测试数据
    test1 = test.copy(deep=True)

    # 设置所有参与分析的特征
    x_feature_params = ['prov_id', 'cust_sex', 'cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee',
                        'total_flux', 'total_dura', 'roam_dura', 'total_times',
                        'total_nums', 'local_nums', 'roam_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura',
                        'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'total_flow',
                        'imei_duration', 'avg_duratioin', 'brand_flag', 'heyue_flag', 'is_limit_flag', 'product_type',
                        '5g_flag', 'active_days', 'service_type']  ## 'app_type_id', 'app_visit_dura'
    train = train[x_feature_params]
    test = test[x_feature_params]

    # 数据特征处理
    print('-----------------------开始特征处理-------------------------')
    train = dataHandle(train)
    test = dataHandle(test)

    # 训练模型
    x_train = train.iloc[:, train.columns != label]
    x_test = test.iloc[:, test.columns != label]
    # 预测分析时，需要加注释
    y_test = []
    # 模型训练数据
    print('------------------------- 开始模型数据训练和预测 -------------------------------')
    ans = xgboostModelTrain(x_train, y_train, x_test, y_test)  # 预测值AUC为 ：0.940659
    print('------------------------- 开始数据写入 -------------------------------')
    path = 'D:/data/python/work/result_' + filename + '.csv'
    writeDataToCsv(test1, ans, path)

if __name__ == '__main__':
    trainFilePath = 'D:/data/python/work/data1/qwr_woyinyue_basic_result0501.txt'
    test_path = 'D:/data/python/work/data1/qwr_woyinyue_basic_result0502.txt'
    #paths = [test_path]
    test_path1 = 'D:/data/python/work/data1/qwr_woyinyue_user_result2006_087.txt'
    test_path2 = 'D:/data/python/work/data1/qwr_woyinyue_user_result2006_036.txt'
    test_path3 = 'D:/data/python/work/data1/qwr_woyinyue_user_result2006_089.txt'
    test_path4 = 'D:/data/python/work/data1/qwr_woyinyue_user_result2006_070.txt'
    test_path5 = 'D:/data/python/work/data1/qwr_woyinyue_user_result2006_013.txt'

    paths = [test_path4,test_path5]
    for path in paths:
        getModelResult(trainFilePath,path)
        print(path +' : 完成')
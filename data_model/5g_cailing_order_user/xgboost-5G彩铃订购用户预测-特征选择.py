import pandas as pd

from time import strftime, localtime
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot as plt

from utils.featureProjectUtils import getFeatureNoneRate

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return


def messagePrint(x):
    print(x)
    print('----------------------------------------')


'''
读取数据，并设置特征名称，返回读取后的数据集
'''


# 打印当前时间
def readData(path, names):
    train = pd.read_csv(filepath_or_buffer=path, sep=",", names=names, encoding='utf-8')
    return train


def handleServiceType(x):
    if x == '40AAAAAA':
        return 4
    elif x == '50AAAAAA':
        return 5
    elif x == '90AAAAAA':
        return 9
    else:
        return None


def is_number(num):
    from builtins import str
    num = str(num)
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


# 处理数据格式
def getDataValue(x):
    from builtins import str
    if is_number(x):
        num = str(x)
        strs = num.split('.')
        if len(strs) > 1:
            len_num = 0
            if len(strs[1]) <= 4:
                len_num = len(strs[0]) + len(strs[1]) + 1
            else:
                len_num = len(strs[0]) + 4
            return float(num[0:len_num])
        else:
            return int(x)
    else:
        # print(x)
        return None


def getIntNumber(x):
    if x != 1.0 and x != 0.0:
        return None
    else:
        strs = str(x).split('.')
        return int(strs[0])


## 处理类别对象，返回整数handleTypeFlag
def handleTypeFlag(df):
    df['flag'] = df['flag'].map(lambda x: getIntNumber(x))
    type_value = df['flag'].value_counts().index[0]
    df['flag'].fillna(type_value, inplace=True)
    return df


# 使用使用众数填补类别缺失值
def fileTypeFeature(data):
    # 处理service_type字段
    data['service_type'] = data['service_type'].apply(lambda x: handleServiceType(x))
    # 处理以数字为类别的特征
    type_feature = ['cust_sex', 'area_id', 'brand_flag', 'heyue_flag', 'activity_type', 'is_limit_flag', 'product_type',
                    '5g_flag', 'service_type', '5g_city_flag', 'one_city_flag', 'app_type_id']  # prov_id
    # 对定性数据，进行填充，填充值为占比最大的类别
    for f in type_feature:
        type_value = data[f].value_counts().index[0]
        data[f].fillna(type_value, inplace=True)

    # 为省分值数据，若缺零，则进行填补
    # data['prov_id'] = data['prov_id'].apply(lambda x:handleProvID(x))
    return data


# 使用均值填充缺失的数据，同时对数据进行归一化处理
def fileNumberFeature(data):
    nums_params = ['innet_months', 'cust_sex', 'cert_age', 'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee',
                   'total_flux', 'total_dura', 'roam_dura',
                   'total_times', 'total_nums', 'local_nums', 'roam_nums', 'in_cnt', 'out_cnt', 'out_dura', 'price',
                   'imei_duration', 'avg_duratioin',
                   'shejiao_active_days', 'shejiao_visit_cnt', 'xinwen_active_days', 'xinwen_visit_cnt',
                   'shipin_active_days', 'shipin_visit_cnt',
                   'dshipin_active_days', 'dshipin_visit_cnt', 'zhibo_active_days', 'zhibo_visit_cnt',
                   'waimai_active_days', 'ditudaohang_visit_cnt',
                   'luntan_active_days', 'luntan_visit_cnt', 'shouji_shoping_active_days', 'shouji_shoping_visit_cnt',
                   'liulanqi_active_days', 'liulanqi_visit_cnt',
                   'wenhua_active_days', 'wenhua_visit_cnt', 'youxi_active_days', 'youxi_visit_cnt',
                   'yinyue_active_days', 'yinyue_visit_cnt', 'work_fze_active_days',
                   'work_fze_visit_cnt', 'jinrong_active_days', 'jinrong_visit_cnt', 'app_active_days',
                   'app_visit_dura']

    # 单独处理avg_duratioin字段
    # data['avg_duratioin'] = data['avg_duratioin'].apply(pd.to_numeric, errors='coerce').fillna(13.6)

    # 处理数值型特征
    for param in nums_params:
        data[param] = data[param].apply(lambda x: getDataValue(x))

    ## 使用均值填充定量数据的缺失字段
    for param in nums_params:
        avg_n = data[param].mean()
        # data[param] = data[param].apply(pd.to_numeric, errors='coerce').fillna(avg_n)
        data[param].fillna(avg_n, inplace=True)
        data[param] = data[param].round(2)
    return data

## 删除完成one-hot编码的特征
def dropFeature(dataF, features):
    dataF.drop(features, axis=1, inplace=True)  # inplace=True, 直接从内部删除
    return dataF


def handleTypeFeature(data):
    # type_feature = ['cust_sex', 'brand_flag', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag','service_type']  # app_type_id
    type_feature = ['product_type']  # app_type_id
    for f in type_feature:
        dummies = pd.get_dummies(data[f], prefix=f)
        data = pd.concat([data, dummies], axis=1)
    # 删除完成编码的特征值
    for f1 in type_feature:
        data = dropFeature(data, f1)
    return data


## 检查数据中的异常值
def unusualValueForCol(data):
    # 删除年龄和arpu值不符合常理的值
    data['cert_age'] = data['cert_age'].map(lambda x: x if x > 0 and x <= 70 else None)
    data['total_fee'] = data['total_fee'].map(lambda x: x if x > 0 else None)
    # 流量及语音数据进行单位转换
    #features2 = ['visit_dura']
    features3 = ['total_flux', 'total_dura']
    # 将日期数据转换以小时为单位的数据值
    #for f1 in features2:
        #data[f1] = data[f1].map(lambda x: x / 60 / 60 if x > 0 else None)
    # 将流量数据转换为G为单位的数据值
    for f3 in features3:
        data[f3] = data[f3].map(lambda x: x / 1024 / 1024 if x > 0 else None)
    return data


#
def dropErrorReadRows(data):
    # 删除不满足的列
    product_type_values = ['2I', 'bjl', '5G', 'other']
    data = data.drop(data[~data.product_type.isin(product_type_values)].index)
    return data


## 根据特征名称，删除异常值
def dropExceptionRows(dataF, features):
    for f in features:
        dataF.dropna(subset=[f], inplace=True)
    return dataF


# 去除空格
def featureStrip(x):
    from builtins import str
    x1 = str(x)
    return x1.strip()


# 去除空格
def stripAndToNumber(x):
    from builtins import str
    x1 = str(x)
    return int(x1.strip())


def readDataToDict(path):
    names = ['area_id', 'city_leave']
    train = pd.read_csv(filepath_or_buffer=path, sep=",", names=names, encoding='utf-8')
    # 数据去重
    train['area_id'] = train['area_id'].apply(lambda x: featureStrip(x))
    train['city_leave'] = train['city_leave'].apply(lambda x: stripAndToNumber(x))
    list1 = train['area_id']
    list2 = train['city_leave']
    data_dict = dict(zip(list1, list2))  # {'k2': 'b', 'k1': 'a'}
    return data_dict


# 处理area_id字段，生成城市等级
def handleAreaidFeature(data):
    area_id_path = 'D:/data/python/work/city_line_message.csv'
    map_dict = readDataToDict(area_id_path)
    data['area_id'] = data['area_id'].map(map_dict)
    return data

# 特征离散化，对连续特征分箱
def featurePSF(data):
    # 处理年龄和ARPU值字段
    # 分箱 - 不使用标签
    print('--------------------特征离散化----------------------')
    age_labels = [0,10,20,25,30,35,40,45,50,60,100]
    #cert_age_column = pd.cut(data['cert_age'],age_labels,labels=False)
    #cert_age_df = pd.DataFrame(cert_age_column)
    data['cert_age'] =  pd.cut(data['cert_age'],age_labels,labels=False)
    arpu_labels = [0,30,50,80,100,120,140,160,180,200,220,240,260,280,300,320,350,1000]
    data['total_fee'] = pd.cut(data['total_fee'], arpu_labels, labels=False)
    return data

'''
数据基础处理
    1、处理异常值
    2、填充空值
    3、数据格式化
'''
def dataHandles(data):
    # 删除读取串行的列
    dropErrorReadRows(data)
    # 填充列别特征
    data = fileTypeFeature(data)
    # 填充数字特征
    data = fileNumberFeature(data)
    # 删除数据异常行
    data = unusualValueForCol(data)
    # 删除异常值
    features = ['cert_age', 'total_fee']
    data = dropExceptionRows(data, features)
    return data
'''
处理类别字段和部分特征离散化
'''
def datahandles2(data):
    # 处理类别字段
    data = handleTypeFeature(data)
    # 特征离散化
    data = featurePSF(data)
    # 处理area_id字段
    data = handleAreaidFeature(data)
    return data

def xgboostModelTrain(x_train, y_train, x_test, y_test):
    # 生成DMatrix,字段内容必须为数字或者boolean
    gb_train = xgb.DMatrix(x_train, y_train)
    xgb_test = xgb.DMatrix(x_test)
    ## 定义模型训练参数
    params = {
        'booster': 'gbtree',  ####  gbtree   gblinear
        'objective': 'binary:logistic',  # 多分类的问题  'objective': 'binary:logistic' 二分类，multi:softmax 多分类问题
        'gamma': 0.3,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'min_child_weight': 5,
        'eta': 0.15,  # 如同学习率
        'learning_rate': 0.08,
        'subsample': 0.5,
        'lambda': 1.5,  # 默认值为1,  权重的L2正则化项 增大该值可以防止过拟合
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
    # model.fit(x_train,y_train)
    # showTree(model)

    ans = model.predict(xgb_test)
    print('预测值AUC为 ：%f' % roc_auc_score(y_test, ans))
    # 显示重要特征
    plot_importance(model)
    plt.show()
    printTime()
    return ans


def dataAnalysis(data, all_params):
    # messagePrint(train_data.describe())
    getFeatureNoneRate(data.iloc[:, data.columns != label], all_params, data.shape[0])
    # x_train = train.iloc[:, train.columns != label]


if __name__ == '__main__':
    all_params = ['prov_id', 'user_id', 'cust_id', 'product_id', 'area_id', 'device_number', 'innet_months',
                  'service_type', 'cust_sex', 'cert_age',
                  'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura',
                  'total_times', 'total_nums', 'local_nums',
                  'roam_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura', 'heyue_flag', 'activity_type',
                  'is_limit_flag', 'product_type', '5g_flag',
                  'brand', 'brand_flag', 'brand_detail', 'price', 'imei_duration', 'avg_duratioin', '5g_city_flag',
                  'one_city_flag', 'shejiao_active_days',
                  'shejiao_visit_cnt', 'xinwen_active_days', 'xinwen_visit_cnt', 'shipin_active_days',
                  'shipin_visit_cnt', 'dshipin_active_days',
                  'dshipin_visit_cnt', 'zhibo_active_days', 'zhibo_visit_cnt', 'waimai_active_days', 'waimai_visit_cnt',
                  'ditudaohang_active_days',
                  'ditudaohang_visit_cnt', 'luntan_active_days', 'luntan_visit_cnt', 'shouji_shoping_active_days',
                  'shouji_shoping_visit_cnt',
                  'liulanqi_active_days', 'liulanqi_visit_cnt', 'wenhua_active_days', 'wenhua_visit_cnt',
                  'youxi_active_days', 'youxi_visit_cnt',
                  'yinyue_active_days', 'yinyue_visit_cnt', 'work_fze_active_days', 'work_fze_visit_cnt',
                  'jinrong_active_days', 'jinrong_visit_cnt',
                  'app_type_id', 'app_active_days', 'app_visit_dura']

    feature_params = ['prov_id', 'area_id', 'innet_months', 'service_type', 'cust_sex', 'cert_age',
                      'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura',
                      'total_times', 'total_nums', 'local_nums',
                      'roam_nums', 'in_cnt', 'out_cnt', 'out_dura', 'heyue_flag', 'activity_type', 'is_limit_flag',
                      'product_type', '5g_flag',
                      'brand_flag', 'price', 'imei_duration', 'avg_duratioin', '5g_city_flag', 'one_city_flag',
                      'shejiao_active_days',
                      'shejiao_visit_cnt', 'xinwen_active_days', 'xinwen_visit_cnt', 'shipin_active_days',
                      'shipin_visit_cnt', 'dshipin_active_days',
                      'dshipin_visit_cnt', 'zhibo_active_days', 'zhibo_visit_cnt', 'waimai_active_days',
                      'ditudaohang_visit_cnt', 'luntan_active_days',
                      'luntan_visit_cnt', 'shouji_shoping_active_days', 'shouji_shoping_visit_cnt',
                      'liulanqi_active_days', 'liulanqi_visit_cnt', 'wenhua_active_days',
                      'wenhua_visit_cnt', 'youxi_active_days', 'youxi_visit_cnt', 'yinyue_active_days',
                      'yinyue_visit_cnt', 'work_fze_active_days',
                      'work_fze_visit_cnt', 'jinrong_active_days', 'jinrong_visit_cnt', 'app_type_id',
                      'app_active_days', 'app_visit_dura']

    # 根据字段的空值率（> 50%） 剔除 in_dura，waimai_visit_cnt，ditudaohang_active_days
    labels = ['flag']
    label = 'flag'
    train_path = 'D:/data/python/work/qwr_woyinyue_basic_result3.txt'
    test_path = 'D:/data/python/work/qwr_woyinyue_basic_result4.txt'
    names = all_params + labels

    # 读取数据
    train_data = readData(train_path, names)
    test_data = readData(test_path, names)

    # 数据分析
    # dataAnalysis(train_data,all_params)
    # 数据处理
    train_data = dataHandles(train_data)
    test_data = dataHandles(test_data)

    #获取标签字段
    train_data = handleTypeFlag(train_data)
    y_train = train_data['flag']
    test_data = handleTypeFlag(test_data)
    y_test = test_data['flag']

    # 获取分析特征数据
    train_data = train_data[feature_params]
    test_data = test_data[feature_params]

    # 处理类别特征和特征离散化
    train_data = datahandles2(train_data)
    test_data = datahandles2(test_data)

    print(train_data.shape)
    print(train_data.info())
    print(test_data.shape)
    print(test_data.info())
    # 训练模型
    xgboostModelTrain(train_data, y_train, test_data, y_test)
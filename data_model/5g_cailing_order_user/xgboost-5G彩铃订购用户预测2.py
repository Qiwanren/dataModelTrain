import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
# 划分数据集
from sklearn.model_selection import train_test_split
from utils.handle_pyplot import showTree

le = LabelEncoder()
from time import strftime, localtime

# 打印当前时间
def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return

def handleFlagField(x):
    if x !=1 and x !=0:
        return 0
    else:
        return x
'''
    指定数据字段的类型及处理空值
'''
### 处理类别型特征
def one_hot(dataFrame,feature):
    dummies = pd.get_dummies(dataFrame[feature], prefix=feature)
    dataFrame = pd.concat([dataFrame, dummies], axis=1)
    return dataFrame

## 删除完成one-hot编码的特征
def dropFeature(dataF,features):
    dataF.drop(features, axis=1, inplace=True)  # inplace=True, 直接从内部删除

def changeType(data):
    data['user_id'].fillna('9999999999', inplace=True)
    data['prov_id'] = data['prov_id'].apply(pd.to_numeric, errors='coerce').fillna(99)
    data['product_id'].fillna('9999999', inplace=True)
    data['area_id'].fillna('0991', inplace=True)
    data['device_number'].fillna('9999999999', inplace=True)
    data['cust_sex'].fillna(1, inplace=True)
    data['cert_age'] = data['cert_age'].apply(pd.to_numeric, errors='coerce').fillna(34.98)
    data["cert_age"] = data["cert_age"].round(2)
    data['total_fee'] = data['total_fee'].apply(pd.to_numeric, errors='coerce').fillna(60.66)
    data["total_fee"] = data["total_fee"].round(2)
    data['jf_flux'] = data['jf_flux'].apply(pd.to_numeric, errors='coerce').fillna(6.9)
    data["jf_flux"] = data["jf_flux"].round(1)
    data['fj_arpu'] = data['fj_arpu'].apply(pd.to_numeric, errors='coerce').fillna(5.4)
    data["fj_arpu"] = data["fj_arpu"].round(1)
    data['ct_voice_fee'] = data['ct_voice_fee'].apply(pd.to_numeric, errors='coerce').fillna(4.0)
    data["ct_voice_fee"] = data["ct_voice_fee"].round(1)
    data['total_flux'] = data['total_flux'].apply(pd.to_numeric, errors='coerce').fillna(16265.98)
    data["total_flux"] = data["total_flux"].round(2)
    data['total_dura'] = data['total_dura'].apply(pd.to_numeric, errors='coerce').fillna(322.45)
    data["total_dura"] = data["total_dura"].round(2)
    data['roam_dura'] = data['roam_dura'].apply(pd.to_numeric, errors='coerce').fillna(64.81)
    data["roam_dura"] = data["roam_dura"].round(2)
    data['total_times'] = data['total_times'].apply(pd.to_numeric, errors='coerce').fillna(434.88)
    data["total_times"] = data["total_times"].round(2)
    data['total_nums'] = data['total_nums'].apply(pd.to_numeric, errors='coerce').fillna(204.89)
    data["total_nums"] = data["total_nums"].round(2)
    data['local_nums'] = data['local_nums'].apply(pd.to_numeric, errors='coerce').fillna(156.85)
    data["local_nums"] = data["local_nums"].round(2)
    data['roam_nums'] = data['roam_nums'].apply(pd.to_numeric, errors='coerce').fillna(36.26)
    data["roam_nums"] = data["roam_nums"].round(2)
    data['in_cnt'] = data['in_cnt'].apply(pd.to_numeric, errors='coerce').fillna(100.04)
    data["in_cnt"] = data["in_cnt"].round(2)
    data['out_cnt'] = data['out_cnt'].apply(pd.to_numeric, errors='coerce').fillna(104.86)
    data["out_cnt"] = data["out_cnt"].round(2)
    data['in_dura'] = data['in_dura'].apply(pd.to_numeric, errors='coerce').fillna(152.31)
    data["in_dura"] = data["in_dura"].round(1)
    data['out_dura'] = data['out_dura'].apply(pd.to_numeric, errors='coerce').fillna(169.66)
    data["out_dura"] = data["out_dura"].round(1)
    data['visit_cnt'] = data['visit_cnt'].apply(pd.to_numeric, errors='coerce').fillna(2231.73)
    data["visit_cnt"] = data["visit_cnt"].round(2)
    data['visit_dura'] = data['visit_dura'].apply(pd.to_numeric, errors='coerce').fillna(35562.76)
    data["visit_dura"] = data["visit_dura"].round(2)
    data['up_flow'] = data['up_flow'].apply(pd.to_numeric, errors='coerce').fillna(1.14)
    data["up_flow"] = data["up_flow"].round(2)
    data['down_flow'] = data['down_flow'].apply(pd.to_numeric, errors='coerce').fillna(4.67)
    data["down_flow"] = data["down_flow"].round(2)
    data['total_flow'] = data['total_flow'].apply(pd.to_numeric, errors='coerce').fillna(4.91)
    data["total_flow"] = data["total_flow"].round(2)
    data['active_days'] = data['active_days'].apply(pd.to_numeric, errors='coerce').fillna(9.96)
    data["active_days"] = data["active_days"].round(2)
    data['imei_duration'] = data['imei_duration'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data["imei_duration"] = data["imei_duration"].round(2)
    data['avg_duratioin'] = data['avg_duratioin'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data["avg_duratioin"] = data["avg_duratioin"].round(2)
    data['brand_flag'] = data['brand_flag'].apply(pd.to_numeric, errors='coerce').fillna(13)
    data['heyue_flag'] = data['heyue_flag'].apply(pd.to_numeric, errors='coerce').fillna(0)
    data['is_limit_flag'] = data['is_limit_flag'].apply(pd.to_numeric, errors='coerce').fillna(0)
    data['product_type'].fillna('other', inplace=True)
    data['5g_flag'] = data['5g_flag'].apply(pd.to_numeric, errors='coerce').fillna(0)

    #data['flag'].fillna(0, inplace=True)
    return data

## 处理类别对象，返回整数
def handleTypeFlag(n):
    return round(n)

## 处理省分ID字段
def handleProvID(x):
    prov_id = str(x)
    return prov_id.zfill(3)

## 将流量的单位转换为G
def handleFluxBigValue(x):
    return x/1024/1024

def handleVoiceBigValue(x):
    return x /60/60

printTime()
trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result1.txt'
testFilePath = 'D:/data/python/work/qwr_woyinyue_user_result2006_087.txt'

all_params = ['prov_id','user_id','cust_id','product_id','area_id','device_number','cust_sex','cert_age','total_fee','jf_flux','fj_arpu',
              'ct_voice_fee','total_flux','total_dura','roam_dura','total_times','total_nums','local_nums','roam_nums','in_cnt','out_cnt',
              'in_dura','out_dura','heyue_flag','is_limit_flag','product_type','5g_flag','visit_cnt','visit_dura','up_flow','down_flow',
              'total_flow','active_days','brand','brand_flag','brand_detail','imei_duration','avg_duratioin']

labels = ['flag']
label = 'flag'

all_params1 = all_params
train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=all_params + labels, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep="|", names=all_params + labels, encoding='utf-8')


print('--- begin to basic handle  -----')
train1 = changeType(train)
test1 = changeType(test)

## 处理类别对象
full_data = [train1, test1]
for dataset in full_data:
    print('--- begin to handle type values -----')
    dataset['brand_flag'] = dataset['brand_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['heyue_flag'] = dataset['heyue_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['is_limit_flag'] = dataset['is_limit_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['5g_flag'] = dataset['5g_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['prov_id'] = dataset['prov_id'].apply(lambda x: handleProvID(x))
    dataset['cust_sex'] = dataset['cust_sex'].apply(lambda x: handleTypeFlag(x))

    ## 处理过大值
    print('--- begin to handle big values -----')
    dataset['total_dura'] = dataset['total_dura'].apply(lambda x: handleVoiceBigValue(x))
    dataset['visit_dura'] = dataset['visit_dura'].apply(lambda x: handleVoiceBigValue(x))
    dataset['up_flow'] = dataset['up_flow'].apply(lambda x: handleFluxBigValue(x))
    dataset['down_flow'] = dataset['down_flow'].apply(lambda x: handleFluxBigValue(x))
    dataset['total_flow'] = dataset['total_flow'].apply(lambda x: handleFluxBigValue(x))
    dataset['total_flux'] = dataset['total_flux'].apply(lambda x: handleFluxBigValue(x))


train1[label] = train1[label].apply(lambda x: handleFlagField(x))
train1[label] = train1[label].apply(lambda x: handleTypeFlag(x))
y_train = train1[label]

test1[label] = test1[label].apply(lambda x: handleFlagField(x))
test1[label] = test1[label].apply(lambda x: handleTypeFlag(x))
y_test = test1[label]

x_featur_params = ['prov_id','cust_sex','cert_age','total_fee','ct_voice_fee','total_flux','jf_flux','fj_arpu','product_type',
                   'total_dura','total_times','local_nums','roam_nums','in_cnt','out_cnt','in_dura','out_dura',
                   'heyue_flag','5g_flag','is_limit_flag','visit_cnt','visit_dura','up_flow','down_flow','total_flow','active_days',
                   'brand_flag','imei_duration','avg_duratioin']


train1 = train1[x_featur_params]
test1 = test1[x_featur_params]

type_feature = ['prov_id','cust_sex','brand_flag','heyue_flag','is_limit_flag','product_type','5g_flag']

## 对分类特征进行one-hot编码
for feature in type_feature:
    train1 = one_hot(train1, feature)
    test1 = one_hot(test1,feature)

## 删除被one-hot编码的列
no_avg_feature = ['jf_flux','fj_arpu','ct_voice_fee']
full_data = [train1, test1]
for dataset in full_data:
    dropFeature(dataset,type_feature)
    dropFeature(dataset, no_avg_feature)

### 切分数据集,均分数据集
x_train, x_test, y_train, y_test = train_test_split(train1, y_train, test_size = 0.5, random_state = 0)
print(train1.shape)
print('----------------------------------------------------')
print(test1.shape)

## 生成DMatrix,字段内容必须为数字或者boolean
gb_train = xgb.DMatrix(x_train,y_train)
xgb_test = xgb.DMatrix(x_test)

## 定义模型训练参数
params = {
    'booster': 'gbtree',   ####  gbtree   gblinear
    'objective': 'binary:logistic',  # 多分类的问题  'objective': 'binary:logistic' 二分类，multi:softmax 多分类问题
    'gamma': 0.0,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 7,               # 构建树的深度，越大越容易过拟合
    'min_child_weight': 5,
    'eta': 0.15,                  # 如同学习率
    'learning_rate':0.08,
    'subsample':0.5,
    'colsample_bytree':0.77,
    'reg_alpha':1.0
}

## 寻找最佳的 n_estimators
plst = params.items()
## 训练轮数
num_rounds = [215,300,400,500]
num_round = 145
print(num_round)
printTime()
print('--- begin to train model -----')
## 模型训练
model = xgb.train(params, gb_train, num_round)

## 特征拟合
#model.fit(x_train,y_train)
#showTree(model)

## 分析特征值
importance = model.get_fscore()
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

ans = model.predict(xgb_test)

## 显示决策树
showTree(model)

test1['score'] = ans
x_test['score'] = ans
x_test['flag'] = y_test

## 将数据写入文件

#print('预测值AUC为 ：%f' % roc_auc_score(y_test, ans))

# 对结果归一化处理
print('--- begin to gyh data -----')
x_test['score1'] = (test['score'] - x_test['score'].min()) / (x_test['score'].max() - x_test['score'].min())
print('--- begin to write data -----')
# 将结果输出到文件
x_test.to_csv('D:/data/python/work/qwr_woyinyue_basic_result1'+str(num_round)+'.csv')

# 显示重要特征
plot_importance(model)
plt.show()
printTime()


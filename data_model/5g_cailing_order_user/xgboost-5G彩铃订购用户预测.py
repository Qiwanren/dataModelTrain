import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


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

## 指标独热编码
def handeleOneHotCode(X):
    le.fit(X)
    return le.transform(X)

##  使用Z-SCORE方法进行归一化处理

def handleDataToOne(x):
    z_score = (x - x.mean()) / x.std()
    return abs(z_score)
'''
def handleDataToOne(x):
    return  x
'''

def changeType(data):
    data['prov_id'].fillna('099', inplace=True)
    data['prov_id'] = handeleOneHotCode(data['prov_id'])
    data['user_id'].fillna('9999999999', inplace=True)
    data['product_id'].fillna('9999999', inplace=True)
    data['area_id'].fillna('0991', inplace=True)
    data['device_number'].fillna('9999999999', inplace=True)
    data['cust_sex'].fillna('1', inplace=True)
    data['cert_age'] = data['cert_age'].apply(pd.to_numeric, errors='coerce').fillna(34.98)
    data["cert_age"] = data["cert_age"].round(2)
    data['total_fee'] = data['total_fee'].apply(pd.to_numeric, errors='coerce').fillna(60.66)
    data['total_fee'] = handleDataToOne(data['total_fee'])
    data["total_fee"] = data["total_fee"].round(2)
    data['jf_flux'] = data['jf_flux'].apply(pd.to_numeric, errors='coerce').fillna(6.9)
    data['jf_flux'] = handleDataToOne(data['jf_flux'])
    data["jf_flux"] = data["jf_flux"].round(1)
    data['fj_arpu'] = data['fj_arpu'].apply(pd.to_numeric, errors='coerce').fillna(5.4)
    data['fj_arpu'] = handleDataToOne(data['fj_arpu'])
    data["fj_arpu"] = data["fj_arpu"].round(1)
    data['ct_voice_fee'] = data['ct_voice_fee'].apply(pd.to_numeric, errors='coerce').fillna(4.0)
    data['ct_voice_fee'] = handleDataToOne(data['ct_voice_fee'])
    data["ct_voice_fee"] = data["ct_voice_fee"].round(1)
    data['total_flux'] = data['total_flux'].apply(pd.to_numeric, errors='coerce').fillna(16265.98)
    data['total_flux'] = handleDataToOne(data['total_flux'])
    data["total_flux"] = data["total_flux"].round(2)
    data['total_dura'] = data['total_dura'].apply(pd.to_numeric, errors='coerce').fillna(322.45)
    data['total_dura'] = handleDataToOne(data['total_dura'])
    data["total_dura"] = data["total_dura"].round(2)
    data['roam_dura'] = data['roam_dura'].apply(pd.to_numeric, errors='coerce').fillna(64.81)
    data['roam_dura'] = handleDataToOne(data['roam_dura'])
    data["roam_dura"] = data["roam_dura"].round(2)
    data['total_times'] = data['total_times'].apply(pd.to_numeric, errors='coerce').fillna(434.88)
    data['total_times'] = handleDataToOne(data['total_times'])
    data["total_times"] = data["total_times"].round(2)
    data['total_nums'] = data['total_nums'].apply(pd.to_numeric, errors='coerce').fillna(204.89)
    data['total_nums'] = handleDataToOne(data['total_nums'])
    data["total_nums"] = data["total_nums"].round(2)
    data['local_nums'] = data['local_nums'].apply(pd.to_numeric, errors='coerce').fillna(156.85)
    data['local_nums'] = handleDataToOne(data['local_nums'])
    data["local_nums"] = data["local_nums"].round(2)
    data['roam_nums'] = data['roam_nums'].apply(pd.to_numeric, errors='coerce').fillna(36.26)
    data['roam_nums'] = handleDataToOne(data['roam_nums'])
    data["roam_nums"] = data["roam_nums"].round(2)
    data['in_cnt'] = data['in_cnt'].apply(pd.to_numeric, errors='coerce').fillna(100.04)
    data['in_cnt'] = handleDataToOne(data['in_cnt'])
    data["in_cnt"] = data["in_cnt"].round(2)
    data['out_cnt'] = data['out_cnt'].apply(pd.to_numeric, errors='coerce').fillna(104.86)
    data['out_cnt'] = handleDataToOne(data['out_cnt'])
    data["out_cnt"] = data["out_cnt"].round(2)
    data['in_dura'] = data['in_dura'].apply(pd.to_numeric, errors='coerce').fillna(152.31)
    data['in_dura'] = handleDataToOne(data['in_dura'])
    data["in_dura"] = data["in_dura"].round(1)
    data['out_dura'] = data['out_dura'].apply(pd.to_numeric, errors='coerce').fillna(169.66)
    data['out_dura'] = handleDataToOne(data['out_dura'])
    data["out_dura"] = data["out_dura"].round(1)
    data['heyue_flag'].fillna(0, inplace=True)
    data['is_limit_flag'].fillna('0', inplace=True)
    data['product_type'].fillna('other', inplace=True)
    data['product_type'] = handeleOneHotCode(data['product_type'])
    data['5g_flag'] = data['5g_flag'].apply(lambda x: handleFlagField(x))
    data['visit_cnt'] = data['visit_cnt'].apply(pd.to_numeric, errors='coerce').fillna(2231.73)
    data['visit_cnt'] = handleDataToOne(data['visit_cnt'])
    data["visit_cnt"] = data["visit_cnt"].round(2)
    data['visit_dura'] = data['visit_dura'].apply(pd.to_numeric, errors='coerce').fillna(35562.76)
    data['visit_dura'] = handleDataToOne(data['visit_dura'])
    data["visit_dura"] = data["visit_dura"].round(2)
    data['up_flow'] = data['up_flow'].apply(pd.to_numeric, errors='coerce').fillna(1.14)
    data['up_flow'] = handleDataToOne(data['up_flow'])
    data["up_flow"] = data["up_flow"].round(2)
    data['down_flow'] = data['down_flow'].apply(pd.to_numeric, errors='coerce').fillna(4.67)
    data['down_flow'] = handleDataToOne(data['down_flow'])
    data["down_flow"] = data["down_flow"].round(2)
    data['total_flow'] = data['total_flow'].apply(pd.to_numeric, errors='coerce').fillna(4.91)
    data['total_flow'] = handleDataToOne(data['total_flow'])
    data["total_flow"] = data["total_flow"].round(2)
    data['active_days'] = data['active_days'].apply(pd.to_numeric, errors='coerce').fillna(9.96)
    data['active_days'] = handleDataToOne(data['active_days'])
    data["active_days"] = data["active_days"].round(2)
    data['brand_flag'].fillna(13, inplace=True)
    data['imei_duration'] = data['imei_duration'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data['imei_duration'] = handleDataToOne(data['imei_duration'])
    data["imei_duration"] = data["imei_duration"].round(2)
    data['avg_duratioin'] = data['avg_duratioin'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data['avg_duratioin'] = handleDataToOne(data['avg_duratioin'])
    data["avg_duratioin"] = data["avg_duratioin"].round(2)
    #data['flag'].fillna(0, inplace=True)
    return data

printTime()
trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result1.txt'
testFilePath = 'D:/data/python/work/qwr_woyinyue_user_result2006_87.txt'
## , 'fj_arpu'
all_params = ['prov_id','user_id','cust_id','product_id','area_id','device_number','cust_sex','cert_age','total_fee','jf_flux','fj_arpu',
              'ct_voice_fee','total_flux','total_dura','roam_dura','total_times','total_nums','local_nums','roam_nums','in_cnt','out_cnt',
              'in_dura','out_dura','heyue_flag','is_limit_flag','product_type','5g_flag','visit_cnt','visit_dura','up_flow','down_flow',
              'total_flow','active_days','brand','brand_flag','brand_detail','imei_duration','avg_duratioin']
labels = ['flag']
label = 'flag'

all_params1 = all_params
train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=all_params + labels, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep=",", names=all_params, encoding='utf-8')

## 对空值和分类字段进行处理
train1 = changeType(train)
train1[label] = train1[label].apply(lambda x: handleFlagField(x))
test1 = changeType(test)


## 生成DMatrix,字段内容必须为数字或者boolean    , 'fj_arpu'
x_featur_params = ['prov_id','cert_age','total_fee','fj_arpu','ct_voice_fee','total_flux','jf_flux',
                   'total_dura','total_times','local_nums','roam_nums','in_cnt','out_cnt','in_dura','out_dura',
                   'heyue_flag','5g_flag','visit_cnt','visit_dura','up_flow','down_flow','total_flow','active_days',
                   'brand_flag','imei_duration','avg_duratioin']

x_featur_params2 = ['prov_id','cert_age','total_fee','fj_arpu','total_flux',
                   'total_dura','total_times','local_nums','roam_nums','in_cnt','out_cnt',
                   'visit_cnt','visit_dura','up_flow','down_flow','total_flow','active_days','imei_duration','avg_duratioin']

xgb_train = xgb.DMatrix(train1[x_featur_params],train1[label])
xgb_test = xgb.DMatrix(test1[x_featur_params])

#y_test = test1['flag'].apply(lambda x: handleFlagField(x))

## 定义模型训练参数
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
## 寻找最佳的 n_estimators
plst = params.items()
## 训练轮数
num_rounds = [215,300,400,500]
num_round = 215
print(num_round)
printTime()
## 模型训练
model = xgb.train(params, xgb_train, num_round)

## 分析特征值
importance = model.get_fscore()
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

ans = model.predict(xgb_test)

test1['score'] = ans
#print('预测值AUC为 ：%f' % roc_auc_score(y_test, ans))

# 对结果归一化处理
test1['score1'] = (test1['score'] - test1['score'].min()) / (test1['score'].max() - test1['score'].min())

# 将结果输出到文件
test1.to_csv('D:/data/python/work/result'+str(num_round)+'.csv')

# 显示重要特征
plot_importance(model)
plt.show()
printTime()
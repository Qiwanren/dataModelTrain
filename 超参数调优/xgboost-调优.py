import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import numpy as np
from xgboost import XGBClassifier

le = LabelEncoder()

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
    data['product_id'].fillna('9999999', inplace=True)
    data['area_id'].fillna('0991', inplace=True)
    data['device_number'].fillna('9999999999', inplace=True)
    data['cust_sex'].fillna(1, inplace=True)
    data['cert_age'] = data['cert_age'].apply(pd.to_numeric, errors='coerce').fillna(34.98)
    data["cert_age"] = data["cert_age"].round(2)
    data['total_fee'] = data['total_fee'].apply(pd.to_numeric, errors='coerce').fillna(60.66)
    data["total_fee"] = data["total_fee"].round(2)
    #data['jf_flux'] = data['jf_flux'].apply(pd.to_numeric, errors='coerce').fillna(6.9)
    #data["jf_flux"] = data["jf_flux"].round(1)
    data['jf_flux'] = data['jf_flux'].apply(lambda x:checkIsZero(x))
    #data['fj_arpu'] = data['fj_arpu'].apply(pd.to_numeric, errors='coerce').fillna(5.4)
    #data["fj_arpu"] = data["fj_arpu"].round(1)
    data['fj_arpu'] = data['fj_arpu'].apply(lambda x: checkIsZero(x))
    #data['ct_voice_fee'] = data['ct_voice_fee'].apply(pd.to_numeric, errors='coerce').fillna(4.0)
    #data["ct_voice_fee"] = data["ct_voice_fee"].round(1)
    data['ct_voice_fee'] = data['ct_voice_fee'].apply(lambda x: checkIsZero(x))
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

## 处理数据中等于零较多的数据
def checkIsZero(x):
    if x == 0:
        return 0
    else:
        return 1

printTime()
trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result3.txt'
testFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result4.txt'

all_params = ['prov_id','user_id','cust_id','product_id','area_id','device_number','cust_sex','cert_age','total_fee','jf_flux','fj_arpu',
              'ct_voice_fee','total_flux','total_dura','roam_dura','total_times','total_nums','local_nums','roam_nums','in_cnt','out_cnt',
              'in_dura','out_dura','heyue_flag','is_limit_flag','product_type','5g_flag','visit_cnt','visit_dura','up_flow','down_flow',
              'total_flow','active_days','brand','brand_flag','brand_detail','imei_duration','avg_duratioin']

labels = ['flag']
label = 'flag'

all_params1 = all_params
train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=all_params + labels, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep="|", names=all_params, encoding='utf-8')



train1 = changeType(train)

### 处理类别对象
full_data = [train1]
for dataset in full_data:
    dataset['brand_flag'] = dataset['brand_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['heyue_flag'] = dataset['heyue_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['is_limit_flag'] = dataset['is_limit_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['5g_flag'] = dataset['5g_flag'].apply(lambda x : handleTypeFlag(x))
    dataset['prov_id'] = dataset['prov_id'].apply(lambda x: handleProvID(x))
    dataset['cust_sex'] = dataset['cust_sex'].apply(lambda x: handleTypeFlag(x))
    dataset['cust_sex'] = dataset['cust_sex'].apply(lambda x: handleTypeFlag(x))
    dataset['jf_flux'] = dataset['jf_flux'].apply(lambda x: handleTypeFlag(x))
    dataset['fj_arpu'] = dataset['fj_arpu'].apply(lambda x: handleTypeFlag(x))
    dataset['ct_voice_fee'] = dataset['ct_voice_fee'].apply(lambda x: handleTypeFlag(x))

    dataset[label] = dataset[label].apply(lambda x: handleFlagField(x))
    dataset[label] = dataset[label].apply(lambda x: handleTypeFlag(x))

y_train = train1[label].apply(lambda x: handleFlagField(x))
#y_test = test1[label].apply(lambda x: handleFlagField(x))

x_featur_params = ['prov_id','cust_sex','cert_age','total_fee','ct_voice_fee','total_flux','jf_flux','product_type',
                   'total_dura','total_times','local_nums','roam_nums','in_cnt','out_cnt','in_dura','out_dura',
                   'heyue_flag','5g_flag','is_limit_flag','visit_cnt','visit_dura','up_flow','down_flow','total_flow','active_days',
                   'brand_flag','imei_duration','avg_duratioin']


train1 = train1[x_featur_params]
#test1 = test1[x_featur_params]

type_feature = ['prov_id','cust_sex','brand_flag','heyue_flag','is_limit_flag','product_type','5g_flag']

## 对分类特征进行one-hot编码
for feature in type_feature:
    train1 = one_hot(train1, feature)

## 删除被one-hot编码的列
dropFeature(train1,type_feature)

print('----------------------------------------------------')


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[label].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics='auc',
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[label], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[label].values, dtrain_predictions))

    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[label], dtrain_predprob))
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

'''
第一步：确定学习速率和tree_based 参数调优的估计器数目。为了确定boosting 参数，我们要先给其它参数一个初始值。咱们先按如下方法取值：
    1、max_depth = 5 :这个参数的取值最好在3-10之间。我选的起始值为5，但是你也可以选择其它的值。起始值在4-6之间都是不错的选择。
    2、min_child_weight = 1:在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。因此，某些叶子节点下的值会比较小。
    3、gamma = 0: 起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的。
    4、subsample,colsample_bytree = 0.8: 这个是最常见的初始值了。典型值的范围在0.5-0.9之间。
    5、scale_pos_weight = 1: 这个值是因为类别十分不平衡。
    
    ###predictors = [x for x in train1.columns if x not in [label,'user_id']]
predictors = [x for x in train1.columns]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

print(train1[label].value_counts())

modelfit(xgb1,train1, predictors)

    
'''

'''
第一次尝试获得理想的max_depth值为7，理想的min_child_weight值为5。在这个值附近我们可以再进一步调整，来找出理想值。
把上下范围各拓展1，因为之前我们进行组合的时候，参数调整的步长是2
    
'''
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
###  7 ， 5
param_test2 = {
 'max_depth':[5,6,7],
 'min_child_weight':[4,5,6]
}

'''
第三步：gamma参数调优
'''
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

'''
第四步：调整subsample 和 colsample_bytree 参数
'''
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

param_test5 = {
 'subsample':[i/100.0 for i in range(50,70,2)],
 'colsample_bytree':[i/100.0 for i in range(75,100,2)]
}

'''
第五步：正则化参数调优
    下一步是应用正则化来降低过拟合。由于gamma函数提供了一种更加有效地降低过拟合的方法
'''
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch1 = GridSearchCV(
    estimator = XGBClassifier(
        learning_rate =0.1,
        n_estimators=140,
        max_depth=7,
        min_child_weight=5,
        gamma=0,
        subsample=0.5,
        colsample_bytree=0.77,
        objective= 'binary:logistic',
        nthread=5,
        scale_pos_weight=1,
        seed=27),
    param_grid = param_test6,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)


gsearch1.fit(train1,y_train)

print("参数的最佳取值：:", gsearch1.best_params_)
print("最佳模型得分:", gsearch1.best_score_)
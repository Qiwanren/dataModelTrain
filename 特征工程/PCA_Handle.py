'''
    主成分分析方法
        原理：主成分分析（PCA）是一种基于变量协方差矩阵对数据进行压缩降维、去噪的有效方法，PCA的思想是将n维特征映射到k维上（k<n），
        这k维特征称为主元，是旧特征的线性组合，这些线性组合最大化样本方差，尽量使新的k个特征互不相关。
        协方差是描述不同变量之间的相关关系，协方差>0时说明 X和 Y是正相关关系，协方差<0时 X和Y是负相关关系，协方差为0时 X和Y相互独立
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    data['cust_sex'].fillna('1', inplace=True)
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
    #data['flag'].fillna(0, inplace=True)
    return data

#输入文件的每行数据都以\t隔开
def loaddata(trainFilePath):

    labels = ['flag']
    all_params = ['prov_id', 'user_id', 'cust_id', 'product_id', 'area_id', 'device_number', 'cust_sex', 'cert_age',
                  'total_fee', 'jf_flux', 'fj_arpu',
                  'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura', 'total_times', 'total_nums', 'local_nums',
                  'roam_nums', 'in_cnt', 'out_cnt',
                  'in_dura', 'out_dura', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag', 'visit_cnt',
                  'visit_dura', 'up_flow', 'down_flow',
                  'total_flow', 'active_days', 'brand', 'brand_flag', 'brand_detail', 'imei_duration', 'avg_duratioin']

    features_params = ['prov_id', 'cust_sex', 'cert_age','total_fee', 'jf_flux', 'fj_arpu','product_type',
                  'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura', 'total_times', 'total_nums', 'local_nums',
                  'roam_nums', 'in_cnt', 'out_cnt','in_dura', 'out_dura','is_limit_flag','heyue_flag', 'is_limit_flag', '5g_flag', 'visit_cnt',
                  'visit_dura', 'up_flow', 'down_flow','total_flow', 'active_days','brand_flag','imei_duration', 'avg_duratioin']

    type_feature = ['prov_id','brand_flag','heyue_flag','is_limit_flag','product_type','5g_flag']

    train = pd.read_csv(filepath_or_buffer=trainFilePath,header=None,sep="|", names=all_params + labels, encoding='utf-8')

    train = changeType(train)

    train1 = train[features_params]

    for feature in type_feature:
        train1 = one_hot(train1,feature)

    dropFeature(train1,type_feature)
    print(train1.info())
    return np.array(train1).astype(np.float)

#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示依照列来求均值。假设输入list,则axis=1


"""
參数：
    - XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
    - k：表示取前k个特征值相应的特征向量
返回值：
    - finalData：參数一指的是返回的低维矩阵，相应于输入參数二
    - reconData：參数二相应的是移动坐标轴后的矩阵
"""
def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData

## 可视化分析
def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0])
        axis_y2.append(dataArr2[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()

def main():
    trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result3.txt'
    XMat = loaddata(trainFilePath)
    k = 2
    return pca(XMat, k)

if __name__ == "__main__":
    finalData, reconMat = main()
    plotBestFit(finalData, reconMat)

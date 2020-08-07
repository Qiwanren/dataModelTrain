import numpy as np
#import load_data
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def data_corection(data):
    ID=data['ID']
    for feature_num,feature in enumerate(data.keys()):
        print('正在处理的特征：',feature)
        Q1 = np.percentile(data[feature],25)
        Q3 = np.percentile(data[feature],75)
        step = (Q3-Q1)
        if feature in ['平均功率','功率A','功率B','功率C','former_power','peak_value','I_eta']:
            step=step*1.5
        elif feature in ['电压A','电压B','电压C']:
            step=step*100
        elif feature in ['光照强度','现场温度','板温']:
            step=step*1.5
        elif feature in ['电流A','电流B','电流C']:
            step=step*3
        elif feature in ['转换效率A','转换效率B','转换效率C','转换效率']:
            step=step*10
        elif feature in ['风向']:
            step=step*1.4
        elif feature in ['风速']:
            step=step*4
        else:
            step=step*1000

        feature_index=data[~((data[feature] >= (Q1 - step)) & (data[feature] <= (Q3 + step)))].index

        for i in range(len(feature_index)):
            if feature_index[i]==0:
                j=feature_index[i]+1
                while j in feature_index:
                    j=j+1
                data.iloc[feature_index[i],feature_num]=data.iloc[j,feature_num]
            else:
                j=feature_index[i]-1
                while j in feature_index:
                    j=j-1
                data.iloc[feature_index[i],feature_num]=data.iloc[j,feature_num]
	return data

def scale(features,scaler):
    scaler.fit(features)
    scaled_features=scaler.transform(features)
    return scaled_features,scaler

if __name__=='__main__':
    print('加载数据...')
    train_data=pd.read_csv('D:/data/python/work/ckzg/public.train.csv')
    print(['原始训练数据：',train_data.shape])
    test_data=pd.read_csv('D:/data/python/work/ckzg/public.test.csv')
    print(['测试数据：',test_data.shape])

    data=test_data.append(train_data.drop('发电量',axis=1))
    data.sort_values('ID',inplace=True)
    data=data.reset_index(drop=True)
    print('合并后的数据：',data.shape)
    data_prc=data_corection(data.copy())
    data_prc.to_csv('D:/data/python/work/ckzg/data_prc_lstm.csv',index=None)
    print('数据预处理完毕')


model1 = LGBMRegressor(n_estimators=1600,objective = 'regression',learning_rate = 0.01,random_state = 50,metric='rmse')
model2 = XGBRegressor(n_estimators=16000,objective = 'reg:linear',learning_rate = 0.01,n_jobs = -1,random_state = 50,eval_metric='rmse',silent=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# scaling function
def scale(train_features,test_features):
    scaler=MinMaxScaler()
    data=pd.concat([train_features,test_features])
    scaled_data=pd.DataFrame(scaler.fit_transform(data),columns=test_features.keys())
    train_features=scaled_data.iloc[0:train_features.shape[0]]
    test_features=scaled_data.iloc[train_features.shape[0]:]
    return train_features,test_features
# result saving function
def save_predictions(IDs,predictions,name='result.csv'):
    predictions=pd.DataFrame(list(zip(map(int,IDs),predictions)))
    predictions.to_csv('data/'+name,header=False,index=False,sep=',')
def do_LSTM_Regression():
    print("使用LSTM进行回归预测")
    # load precessed data
    train_data=pd.read_csv('data/train_all_features.csv')
    train_features=train_data.drop('发电量',axis=1)
    y=train_data['发电量']
    X_test=pd.read_csv('data/test_all_features.csv')
    # select best features for LSTM
    selected_features=['平均功率', '电流A^2', 'ID', 'former_power', '板温 风向', '风速 风向', '电压A temp_diff',
                       '光照强度^2', '电压A 风向', '电压A 风速', '现场温度', 'vdc_A', 'vdc_B', '电压A^2', '功率C',
                       '功率A', '板温 电压A', '电流A', '电压A', '光照强度', 'vdc_B_square', 'vdc_A_square', 'wind',
                       '转换效率A temp_diff', 'P_eta', '板温 temp_diff', '风向^2', '风速 temp_diff', '转换效率C',
                       '风向', 'std_power', '板温 风速', '板温', '板温 光照强度', '板温^2', '电压B', '转换效率A 风速',
                       '转换效率A 电流A', '板温 电流A', '电流A 风速', 'vdc_C', '转换效率A 电压A', 'vdc_C_square',
                       'mean_board_temp', '转换效率', 'idc_B_square', 'idc_B', '功率B', '电流B', '电压A 电流A',
                       'PN_I', '光照强度 风速', 'idc_C_square', '电压C', 'idc_C', 'I_eta', '电流C', '转换效率A^2',
                       '转换效率A', 'idc_A_square', 'dis2peak', 'idc_A', '转换效率A 风向', '光照强度 风向',
                       '风向 temp_diff', '风速^2', '风速', '板温 转换效率A']
    #adapt the data form to LSTM
    X_test=X_test[selected_features]
    X_train=train_features[selected_features]
    X_train,X_test=scale(X_train,X_test)# scaling
    # X_train=X_train.values.reshape(X_train.shape[0],1,X_train.shape[1])
    X_test=X_test.values.reshape(X_test.shape[0],1,X_test.shape[1])
    #split the data
    X_train,X_val,y_train,y_val=train_test_split(X_train,y,test_size=0.2,random_state=333)
    #construct LSTM based on keras
    model = Sequential()
    model.add(LSTM(200,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(X_train, y_train, epochs=1100, batch_size=100, validation_data=(X_val, y_val), verbose=2, shuffle=False)
    model=load_model('lstm_model_8805_a8485.h5')# load the history best model to recovery the best result
    # #prediction
    y_val_pred=model.predict(X_val)
    y_test_pred=model.predict(X_test)
    rmse=mean_squared_error(y_val,y_val_pred)**0.5
    score=1.0/(1.0+rmse)
    y_test_pred=np.array(y_test_pred).flatten()
    ID=pd.read_csv('data/test.csv')['ID']
    save_predictions(ID,y_test_pred,'result_lstm.csv')
    print('score:',score)
# module test
if __name__=='__main__':
    do_LSTM_Regression()
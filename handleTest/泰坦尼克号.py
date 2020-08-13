# Load in our libraries
import pandas as pd
import numpy as np

import xgboost as xgb

'''

##  删除噪音特征
# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)
'''


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
def handleTicket(x):
    if is_number(x) == False:
        list = str(x).split(' ')
        if is_number(list[len(list)-1]) == False:
            return 0
        else:
            return list[len(list)-1]
    else:
        return x
def handleNumberFeature(x):
    if is_number(x) == False:
        print('-- ',x)
        return 0
    else:
        return x


# Load in the train and test datasets
train = pd.read_csv('D:/data/python/work/titanic/train.csv')
test = pd.read_csv('D:/data/python/work/titanic/test.csv')

full_data = [train, test]

# 创建一个新的特征CategoricalAge
for dataset in full_data:
    ## 填充空值
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].apply(lambda x:handleNumberFeature(x))
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    dataset['Age'] = dataset['Age'].apply(lambda x: handleNumberFeature(x))
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list


train['CategoricalAge'] = pd.cut(train['Age'], 5)
test['CategoricalAge'] = pd.cut(test['Age'], 5)
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
test['CategoricalFare'] = pd.qcut(test['Fare'], 4)

for dataset in full_data:
    # 将Embarked映射至0-2。
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 将性别映射至0,1
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 将Fare分成四类0-3
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # 将年龄分为5类：0-4
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4;


train['Ticket'] = train['Ticket'].apply(lambda x:handleTicket(x))
train['Ticket'] = train['Ticket'].astype(int)
test['Ticket'] = test['Ticket'].apply(lambda x:handleTicket(x))
test['Ticket'] = test['Ticket'].astype(int)

##  删除噪音特征
# Feature selection
#drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','CategoricalAge','CategoricalFare']
#train = train.drop(drop_elements, axis = 1)
#test  = test.drop(drop_elements, axis = 1)

#x_train = xgb.DMatrix(train)
#y_train = xgb.DMatrix(train[label])

model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=550, max_depth=4, min_child_weight=5, seed=0,
                             subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=1, reg_lambda=1)

train_featur_params = ['Survived','Pclass','Sex','Age','Parch','Ticket','Fare','Embarked']
test_featur_params = ['Pclass','Sex','Age','Parch','Ticket','Fare','Embarked']
label = 'Survived'
x_train = train[train_featur_params]
y_train = train[label]

model.fit(x_train,y_train)

xgb.to_graphviz(model, num_trees=1)
digraph = xgb.to_graphviz(model, num_trees=1)
digraph.format = 'png'
digraph.view('./iris_xgb')



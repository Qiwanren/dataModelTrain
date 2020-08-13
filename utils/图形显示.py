import pandas as pd
from matplotlib import pyplot as plt

path1 = 'D:/data/python/test/GlobalLandTemperaturesByCity.csv'
#path2 = 'D:/data/python/test/Salary_Ranges_by_Job_Classification.csv'

train1 = pd.read_csv(filepath_or_buffer=path1, encoding='utf-8')

# 绘制条形图
train1['Grade'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')

# 绘制饼图
train1['Grade'].value_counts().sort_values(ascending=False).head(5).plot(kind='pie')

# 绘制箱体图
train1['Union Code'].value_counts().sort_values(ascending=False).head(5).plot(kind='box')

# 绘制直方图
train1['AverageTemperature'].hist()

# 为每个世纪（Century）绘制平均温度的直方图
train1['AverageTemperature'].hist(by=train1['Century'],
                                             sharex=True,
                                             sharey=True,
                                             figsize=(10, 10),
                                             bins=20)

# 绘制散点图
x = train1['year']
y = train1['AverageTemperature']

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.show()
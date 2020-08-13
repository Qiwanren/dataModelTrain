#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def method01():
    arr = np.array([1, 2, 3, 4, 5])
    arr += 1
    # print(arr)

    ## 切片操作
    # print(arr[0:3])
    # print(arr[-2:])

    # 定义二维数组,索引从零开始
    ndarray1 = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
    # print(ndarray1)
    # print(ndarray1[0,0])

    ### 读取第一行数据
    # print(ndarray1[0])

    # 读取第一列列数据
    # print(ndarray1[:,0])

    # 取第一行的前两个值
    # print(ndarray1[0,0:2])

    # 对值进行修改
    ndarray1[1, 1] = 100
    # print(ndarray1)

    # 复制操作
    arr1 = ndarray1  # 不会创建新的，只是将ndarray1的地址也指向了arr1
    arr2 = ndarray1.copy()  # 进行新的复制，两个不指向同一地址，两个互不干涉

    # arange函数，构造等差数组
    arr3 = np.arange(0, 100, 10)  # 在0到100之间，每十个选取一个数,左闭右开的数组
    # print(arr3)

    # 布尔类型数据做索引选取数据
    arr4 = np.array([0, 1, 1, 0, 1], dtype=bool)
    # print(arr4)
    # print(arr[arr4])

    # 条件判断
    arr5 = arr[np.where(arr > 2)]
    # print(arr5)

    #########  数值计算  ##################
    # 累加 根据某个维度进行计算
    # 按列进行计算
    ndarray2 = np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
    # 按列进行计算
    # arr6 = np.sum(ndarray2,axis=0)
    # 按行进行计算
    # arr6 = np.sum(ndarray2,axis=1)

    ### 乘积
    # print(ndarray2.prod())
    # print(ndarray2.prod(axis=0))   ###列乘积
    # print(ndarray2.prod(axis=1))   ###行乘积

    ### 最小值，最大值
    # print(ndarray2.min())
    # print(ndarray2.min(axis=0))
    # print(ndarray2.min(axis=1))

    # print(ndarray2.max())
    # print(ndarray2.max(axis=0))
    # print(ndarray2.max(axis=1))

    ### 平均值
    # print(ndarray2.mean())
    ### 标准差
    # print(ndarray2.std())
    ### 方差
    # print(ndarray2.var())

    #### 限制值大小
    ndarray3 = ndarray2.clip(2, 6)  ####  元素中小于2的都变为2，大于8的都变为8
    # print(ndarray3)

    ### 四舍五入  round(decimals=1)  指定精度到第一位小数

    ### 排序  数组排序时的基准，axis=0axis=0axis=0，按列排列；axis=1axis=1axis=1，按行排列
    ndarray4 = np.array([[1.5, 1.3, 7.5], [5.6, 7.8, 1.2]])
    ndarray4 = np.sort(ndarray4)  ## 默认按行排序
    # print(ndarray4)
    ### lexsort 指定不同列的排序规则

    ##### 生成数组结构 ######

    arr6 = np.arange(10)  # 生成一个0到10的数组
    # print(arr6)
    arr7 = np.arange(2, 20, 2, dtype=np.int32)  # 从2到20，每两个值取一个数
    # print(arr7)

    # logspace(a,b,n)，创建行向量，第一个是10^a，最后一个10^b，形成总数为n个元素的等比数列
    arr9 = np.logspace(0, 1, 10)

    arr8 = np.linspace(0, 10, 50)  ### 从0到10之间等差的构造50个数字
    # print(arr8)

    arr8 = np.linspace(0, 100, 10, dtype=np.int32)
    x = np.linspace(-10, 10, 5)
    y = np.linspace(-10, 10, 5)

    ## 构建一个网格
    x, y = np.meshgrid(x, y)

    # print(x,y)
    # print(arr9)

    # 构建矩阵
    ##  构建一个都是 0 或者都是 1 的矩阵
    arr9 = np.zeros(3)  ## 构建一个长度为3的都是0的数组
    ## 构建一个3*3的矩阵结构
    arr9 = np.zeros((3, 3))
    arr9 = np.ones((3, 3))
    arr9 = np.ones((3, 3)) * 8
    # print(arr9)

    ### 四则运算

    ### 随机模块
    ar1 = np.random.rand(3, 2)  # 构建一个3行2列的矩阵,所有元素都是0到1的
    # print(ar1)

    ar2 = np.random.randint(10, size=(5, 4))  ## 构建一个元素是0到10的，大小为5行4列的矩阵
    # print(ar2)
    val = np.random.rand()  ### 返回一个数值
    # print(val)

    ar2 = np.random.randint(0, 10, 3)
    # print(ar2)

    ### 数据洗牌
    ar4 = np.arange(10)
    np.random.shuffle(ar4)  ### 不产生新的数组，只是在原有的基础上进行打乱
    # print(ar4)

    ## 高斯分布
    mu, sigma = 0, 0.1
    ar5 = np.random.normal(mu, sigma, 10)

    ####  文件操作

    # ar6 = np.loadtxt('D:/data/machineLearn/pyspark/1.txt',delimiter=',')
    # print(ar6)

    # ar7 = np.loadtxt('D:/data/machineLearn/pyspark/2.txt',str,delimiter=',',skiprows = 1)   ##跳过第一行
    # ar7 = np.loadtxt('D:/data/machineLearn/pyspark/2.txt',str,delimiter=',',usecols = (1,2))  ### 只读取1,2, 两列

    ##print(ar7)

    ### 多文件一起读取
    '''
    arr1 = np.array([1, 2, 3]) #假设有三个ndarray
    arr2 = np.eye(5)
    arr3 = np.zeros((3, 4))
    np.savez('arr.npz', x0=arr1, x1=arr2, x2=arr3)  #将其保存
    f = np.load('arr.npz')  #加载
    print(f['x1'])    #读取

    '''

    #  保存文件
    # ndarray4 = np.array([[1.5,1.3,7.5],[5.6,7.8,1.2]])
    # np.savetxt('D:/data/machineLearn/pyspark/3.txt',ndarray4,fmt='%d',delimiter=',')   ## 指定格式为整数，分隔符为 ，

    ### 把一个变量可以存储为npy
    ### 保存为压缩文件
    ar8 = np.arange(10)
    # np.savez('1.npz',a=ar8,b=ar7) ## 压缩文件中为多个npy文件
    # data = np.load('1.npz')

    depths = np.arange(1, 10)
    print(depths)

def method02():
    ar2 = np.arange(0, 10, 1)
    print(ar2)
if __name__ == '__main__':
    method02()
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:02:48 2018
书上给出的yahooAPI的baseurl已经改变，github上有oauth2供python使用，
但是yahoo的BOOS GEO好像OAuth2验证出了问题，虽然写了新的placeFinder调用api的代码，
仍然会有403错误。
好在随书代码中已经给出place.txt，所以直接调用，这里略过获取数据的步骤。
@author: wzy
"""
import urllib
import json
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import sys
import math as math




"""
函数说明：数据向量计算欧式距离
Parameters:
    vecA - 数据向量A
    vecB - 数据向量B

Returns:
    两个向量之间的欧几里德距离
"""


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


"""
函数说明：随机初始化k个质心（质心满足数据边界之内）
Parameters:
    dataSet - 输入的数据集
    k - 选取k个质心

Returns:
    centroids - 返回初始化得到的k个质心向量
"""


def randCent(dataSet, k):
    # 得到数据样本的维度
    n = np.shape(dataSet)[1]
    # 初始化为一个(k,n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 遍历数据集的每一个维度
    for j in range(n):
        # 得到该列数据的最小值,最大值
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(maxJ - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    # 返回初始化得到的k个质心向量
    return centroids
"""
函数说明：k-means聚类算法
Parameters:
    dataSet - 用于聚类的数据集
    k - 选取k个质心
    distMeas - 距离计算方法,默认欧氏距离distEclud()
    createCent - 获取k个质心的方法,默认随机获取randCent()

Returns:
    centroids - k个聚类的聚类结果
    clusterAssment - 聚类误差
"""


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据集样本数
    m = np.shape(dataSet)[0]
    # 初始化一个（m,2）全零矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不发生变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为False
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离为正无穷，最小距离对应的索引为-1
            minDist = float('inf')
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    minDist = distJI
                    minIndex = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔值置为True，继续聚类算法
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i, :] = minIndex, minDist ** 2
            # 打印k-means聚类的质心
        # print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算这些数据的均值(axis=0:求列均值)，作为该类质心向量
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment
"""
函数说明：二分k-means聚类算法
Parameters:
    dataSet - 用于聚类的数据集
    k - 选取k个质心
    distMeas - 距离计算方法,默认欧氏距离distEclud()

Returns:
    centroids - k个聚类的聚类结果
    clusterAssment - 聚类误差
"""


def biKmeans(dataSet, k, distMeas=distEclud):
    # 获取数据集的样本数
    m = np.shape(dataSet)[0]
    # 初始化一个元素均值0的(m, 2)矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 获取数据集每一列数据的均值，组成一个列表
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 当前聚类列表为将数据集聚为一类
    centList = [centroid0]
    # 遍历每个数据集样本
    for j in range(m):
        # 计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1]=distMeas(np.mat(centroid0),dataSet[j,:])**2
    # 循环，直至二分k-Means值达到k类为止
    while(len(centList)<k):
        # 将当前最小平方误差置为正无穷
        lowerSSE=float('inf')
        # 遍历当前每个聚类
        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster=dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对该类利用二分k-means算法进行划分，返回划分后的结果以及误差
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)
            # 计算该类划分后两个类的误差平方和
            sseSplit=np.sum(splitClustAss[:,1])
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit=np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # 打印这两项误差值
            print('sseSplit = %f, and notSplit = %f' % (sseSplit, sseNotSplit))
            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit+sseNotSplit)<lowerSSE:
                # 第i类作为本次划分类
                bestCentToSplit=i
                # 第i类划分后得到的两个质心向量
                bestNewCents=centroidMat
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss=splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowerSSE=sseNotSplit+sseSplit
        # 数组过滤选出本次2-means聚类划分后类编号为1数据点，将这些数据点类编号变为
        # 当前类个数+1， 作为新的一个聚类
        bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        # 同理，将划分数据中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        # 连续不出现空缺
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # 打印本次执行2-means聚类算法的类
        print('the bestCentToSplit is %d' % bestCentToSplit)
        # 打印被划分的类的数据个数
        print('the len of bestClustAss is %d' % len(bestClustAss))
        # 更新质心列表中变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0, :]
        # 添加新的类的质心向量
        centList.append(bestNewCents[1, :])

        # 更新clusterAssment列表中参与2-means聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    # 返回聚类结果
    return centList, clusterAssment
"""
函数说明：利用地名、城市获取位置处经纬度
Parameters:
    stAddress - 地名
    city - 城市

Returns:
    None
"""


def geoGrab(stAddress, city):
    # 获取经纬度网址
    apiStem = "http://where.yahooapis.com/geocode?"
    # 初始化一个字典，存储相关参数
    params = {}
    # 返回类型为json
    params['flags'] = 'J'
    # 参数appid
    params['appid'] = 'ppp68N8t'
    # 参数地址位置信息
    params['location'] = ('%s %s' % (stAddress, city))
    # 利用urlencode函数将字典转为URL可以传递的字符串格式
    url_params = urllib.parse.urlencode(params)
    # 组成完整的URL地址api
    yahooApi = apiStem + url_params
    # 打印该URL地址
    print('%s' % yahooApi)
    # 打开URL，返回JSON格式数据
    c = urllib.request.urlopen(yahooApi)
    # 返回JSON解析后的数据字典
    return json.load(c.read())


"""
函数说明：具体文本数据批量地址经纬度获取
Parameters:
    fileName - 文件名称

Returns:
    None
"""


def massPlaceFind(fileName):
    # "wb+" 以二进制写方式打开,可以读\写文件,如果文件不存在,创建该文件.如果文件已存在,先清空,再打开文件
    # 以写方式打开,只能写文件,如果文件不存在,创建该文件如果文件已存在,先清空,再打开文件
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        # 获取该地址的经纬度
        retDict = geoGrab(lineArr[1], lineArr[2])
        # 获取到相应的经纬度
        if retDict['ResultSet']['Error'] == 0:
            # 从字典中获取经度
            lat = float(retDict['ResultSet']['Results'][0]['latitute'])
            # 从字典中获取维度
            lng = float(retDict['ResultSet']['Results'][0]['longitute'])
            # 打印地名及对应的经纬度信息
            print('%s\t%f\t%f' % (lineArr[0], lat, lng))
            # 保存入文件
            fw.write('%s\t%f\t%f' % (line, lat, lng))
        else:
            print('error fetching')
        # 为防止频繁调用API，造成请求被封，使函数调用延迟一秒
        sleep(1)
    # 文本写入关闭
    fw.close()


"""
函数说明：球面距离计算
Parameters:
    vecA - 数据向量A
    vecB - 数据向量B

Returns:
    球面距离

"""


def distSLC(vecA, vecB):
    a = math.sin(vecA[0, 1] * np.pi / 180) * math.sin(vecB[0, 1] * np.pi / 180)
    b = math.cos(vecA[0, 1] * np.pi / 180) * math.cos(vecB[0, 1] * np.pi / 180) * math.cos(
        np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return math.acos(a + b) * 6371.0


"""
函数说明：k-means聚类
Parameters:
    numClust - 聚类个数

Returns:
    None
"""


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    # 利用2-means聚类算法聚类
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], \
                    ptsInCurrCluster[:, 1].flatten().A[0], \
                    marker=markerStyle, s=90)
    for i in range(numClust):
        ax1.scatter(myCentroids[i].tolist()[0][0], myCentroids[i].tolist()[0][1], s=300, c='k', marker='+', alpha=.5)
    plt.show()


if __name__ == '__main__':
    # 不能通过URL访问了
    # massPlaceFind('portlandClubs.txt')
    clusterClubs()

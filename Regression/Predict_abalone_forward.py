# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
def regularize(xMat, yMat):
	"""
	函数说明:数据标准化
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		inxMat - 标准化后的x数据集
		inyMat - 标准化后的y数据集
	"""
	inxMat = xMat.copy()														#数据拷贝
	inyMat = yMat.copy()
	yMean = np.mean(yMat, 0)													#行与行操作，求均值
	inyMat = yMat - yMean														#数据减去均值
	inMeans = np.mean(inxMat, 0)   												#行与行操作，求均值
	inVar = np.var(inxMat, 0)     												#行与行操作，求方差
	inxMat = (inxMat - inMeans) / inVar											#数据减去均值除以方差实现标准化
	return inxMat, inyMat
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    函数说明:前向逐步线性回归
    Parameters:
    	xArr - x输入数据
    	yArr - y预测数据
    	eps - 每次迭代需要调整的步长
    	numIt - 迭代次数
    Returns:
    	returnMat - numIt次迭代的回归系数矩阵
    """
    xMat = np.mat(xArr);
    yMat = np.mat(yArr).T  # 数据集
    xMat, yMat = regularize(xMat, yMat)  # 数据标准化
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))  # 初始化numIt次迭代的回归系数矩阵
    ws = np.zeros((n, 1))  # 初始化回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):  # 迭代numIt次
        # print(ws.T)																	#打印当前回归系数矩阵
        lowestError = float('inf');  # 正无穷
        for j in range(n):  # 遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign  # 微调回归系数
                yTest = xMat * wsTest  # 计算预测值
                rssE = rssError(yMat.A, yTest.A)  # 计算平方误差
                if rssE < lowestError:  # 如果误差更小，则更新当前的最佳回归系数
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T  # 记录numIt次迭代的回归系数矩阵
    return returnMat

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
def rssError(yArr,yHatArr):
    """
    函数说明:计算平方误差
    Parameters:
    	yArr - 预测值
    	yHatArr - 真实值
    """
    return np.sum(((yArr-yHatArr)**2))
def plotstageWiseMat():
	"""
	函数说明:绘制岭回归系数矩阵
	"""
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	xArr, yArr = loadDataSet('abalone.txt')
	returnMat = stageWise(xArr, yArr, 0.005, 1000)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(returnMat)
	ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties = font)
	ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties = font)
	ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
	plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
	plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
	plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
	plt.show()
if __name__ == '__main__':
    plotstageWiseMat()
# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
def standRegress(xArr,yArr):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("矩阵为奇异矩阵，不可逆")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws
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
def ridgeRegress(xMat, yMat, lam = 0.2):
	"""
	函数说明:岭回归
	Parameters:
		xMat - x数据集
		yMat - y数据集
		lam - 缩减系数
	Returns:
		ws - 回归系数
    """
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def ridgeTest(xArr,yArr):
    """
    	函数说明:岭回归测试
    	Parameters:
    		xMat - x数据集
    		yMat - y数据集
    	Returns:
    		wMat - 回归系数矩阵
    """
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat
def rssError(yArr,yHatArr):
    """
    函数说明:计算平方误差
    Parameters:
    	yArr - 预测值
    	yHatArr - 真实值
    """
    return np.sum(((yArr-yHatArr)**2))
if __name__ == '__main__':
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()
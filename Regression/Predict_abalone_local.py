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
def lwlr(testpoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat(np.eye((m)))#创建权重对角矩阵
    for j in range(m):
        diffMat=testpoint-xMat[j,:]
        weights[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))#权重大小值以指数级衰减
    xTx=xMat.T*(weights*xMat)
    if np.linalg.det(xTx)==0.0:
        print("矩阵为奇异矩阵，不可逆")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testpoint*ws
def lwlrtest(testArr,xArr,yArr,k=1.0):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat
def rssError(yArr,yHatArr):
    return np.sum(((yArr-yHatArr)**2))
if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrtest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrtest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrtest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[0:99], yHat10.T))

    print('')

    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrtest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrtest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrtest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))

    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = standRegress(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))
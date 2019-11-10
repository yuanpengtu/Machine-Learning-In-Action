#标准回归函数
import numpy as np
import matplotlib.pyplot as plt
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
if __name__ == '__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    yHat=lwlrtest(xArr,xArr,yArr,0.003)
    xMat=np.mat(xArr)
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red')
    plt.show()



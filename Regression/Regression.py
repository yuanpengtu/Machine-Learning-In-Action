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
def standRegress(xArr,yArr):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("矩阵为奇异矩阵，不可逆")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws
if __name__ == '__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegress(xArr,yArr)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

    yHat_copy=xMat*ws
    print(np.corrcoef(yHat_copy.T,yMat))


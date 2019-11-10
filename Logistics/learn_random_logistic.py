import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
"""
函数说明:加载数据
Parameters:
	无
Returns:
	dataMat - 数据列表
	labelMat - 标签列表
"""
def loadDataSet():
	dataMat = []														#创建数据列表
	labelMat = []														#创建标签列表
	fr = open('testSet.txt')											#打开文件
	for line in fr.readlines():											#逐行读取
		lineArr = line.strip().split()									#去回车，放入列表
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		#添加数据
		labelMat.append(int(lineArr[2]))								#添加标签
	fr.close()															#关闭文件
	return dataMat, labelMat											#返回

"""
函数说明:随机梯度上升算法
Parameters:
	dataMatIn - 数据集
	classLabels - 数据标签
Returns:
	weights.getA() - 求得的权重数组(最优参数)
"""
def stocgradAscent(dataMatIn, classLabels,numIter=150):
    dataMatrix=np.array(dataMatIn)
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    alpha=0.01
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights
"""
函数说明:改进后的随机梯度上升算法
Parameters:
	dataMatIn - 数据集
	classLabels - 数据标签
Returns:
	weights.getA() - 求得的权重数组(最优参数)
"""
def stocgradAscent_improve(dataMatIn, classLabels,numIter=150):
    dataMatrix=np.array(dataMatIn)
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    weights_array=np.array([])
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(list(dataIndex)[randIndex])
    return weights


"""
函数说明:绘制数据集
Parameters:
	weights - 权重参数数组
Returns:
	无
"""
def plotBestFit(weights,weights_improved):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)
    x = np.arange(-4.0, 4.0, 0.01)
    y = (-weights[0] - weights[1] * x) / weights[2]#此式子即为0=w0x0+w1x1+w2x2且x0=1时的x2的表达式
    x1 = np.arange(-4.0, 4.0, 0.01)
    y1 = (-weights_improved[0] - weights_improved[1] * x1) / weights_improved[2]
    # ax.plot(x, y)
    ax.plot(x1, y1)
    plt.title('改进随机梯度上升Logistic',fontproperties=font_set)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocgradAscent(dataMat, labelMat)
    weights_improve=stocgradAscent_improve(dataMat,labelMat,500)
    plotBestFit(weights,weights_improve)


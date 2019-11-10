from sklearn.linear_model import LogisticRegression
import numpy as np
import random

"""
函数说明:sigmoid函数
Parameters:
	inX - 数据
Returns:
	sigmoid函数
"""
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

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
函数说明:分类函数
Parameters:
	inX - 特征向量
	weights - 回归系数
Returns:
	分类结果
"""
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
"""
函数说明:使用Python写的Logistic分类器做预测
Parameters:
	无
Returns:
	无
"""
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currline=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currline)-1):
            lineArr.append(float(currline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currline[-1]))
    trainWeights=stocgradAscent_improve(np.array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currline=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currline)-1):
            lineArr.append(float(currline[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currline[-1]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)*100
    print("自构建logistic分类测试集错误率为: %.2f%%" % (errorRate))
    return errorRate
def colicSklearn():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    testSet=[];testLabels=[]
    for line in frTrain.readlines():
        currline=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currline)-1):
            lineArr.append(float(currline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currline[-1]))
    for line in frTest.readlines():
        currline=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currline)-1):
            lineArr.append(float(currline[i]))
        testSet.append(lineArr)
        testLabels.append(float(currline[-1]))
    classifier=LogisticRegression(solver='sag',max_iter = 5000).fit(trainingSet,trainingLabels)
    test_accuracy=classifier.score(testSet,testLabels)*100
    print('sklearn logistics正确率为：%f%%'%test_accuracy)

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("自构建logistic在%d次迭代后，平均错误率为：%f"%(numTests,errorSum/float(numTests)))
if __name__ == '__main__':
    colicSklearn()
    colicTest()
    multiTest()
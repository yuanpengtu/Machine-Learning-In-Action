# -*- coding UTF-8 -*-
"""==========================================
@Project -> File :Machine Learning in Action->Bayes-Email.py
@IDE    :PyCharm
@Author :YuanPeng Tu
@Date   :2019-11-10 17:00
@Desc   :
=========================================="""
import numpy as np
from functools import reduce
import re
import random
"""
函数说明：创建实验样本
Parameters:
    None

Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
"""
def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本切分的词条、类别标签向量
    return postingList, classVec


"""
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表

Returns:
    returnVec - 文档向量，词集模型
"""
def setOfWord2Vec(vocabList,inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        if word in vocabList:
            # 如果词条存在于词汇表中，则置1
            # index返回word出现在vocabList中的索引
            # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中德对应值
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary" % word)
    # 返回文档向量
    return returnVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2CLassify,p0Vec,p1Vec,pCLass1):
    p1=sum(vec2CLassify*p1Vec)+np.log(pCLass1)
    p0=sum(vec2CLassify*p0Vec)+np.log(1.0-pCLass1)
    if p1>p0:
        return 1
    else:
        return 0

def textParse(bigString):
    # 用特殊符号作为切分标志进行字符串切分，即非字母、非数字
    # \W* 0个或多个非字母数字或下划线字符（等价于[^a-zA-Z0-9_]）
    listOfTockens = re.split(r'\W*', bigString)
    # 除了单个字母，例如大写I，其他单词变成小写，去掉少于两个字符的字符串
    return [tok.lower() for tok in listOfTockens if len(tok) > 2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pspam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0v,p1v,pspam)!=classList[docIndex]:
            errorCount+=1
            print("分类错误的测试集：", docList[docIndex])
        print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
    spamTest()


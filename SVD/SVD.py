# -*- coding UTF-8 -*-
"""==========================================
@Project -> File :Machine Learning in Action->SVD.py
@IDE    :PyCharm
@Author :YuanPeng Tu
@Date   :2019-11-10 20:22
@Desc   :
=========================================="""
from numpy import *
def loadExData():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
def standEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:
            continue
        overLap=nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))
        if len(overLap)==0:
            similarity=0
        else:
            similarity=simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal


def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    U,sigma,VT=linalg.svd(dataMat)
    Sig4=mat(eye(4)*sigma[:4])
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
def ecludSim(inA, inB):
    return 1.0 / (1.0 + linalg.norm(inA - inB))

def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:
        return ("you've rated everything")
    itemScores=[]
    for item in unratedItems:
        estimateScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimateScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]
if __name__ == '__main__':
    myMat=mat(loadExData())
    recommend(myMat,1,estMethod=svdEst)
    A=recommend(myMat,1,estMethod=svdEst,simMeas=pearsSim)
    print(A)
    A=recommend(myMat,1,estMethod=svdEst)
    print(A)

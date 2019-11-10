# -*- coding UTF-8 -*-
"""==========================================
@Project -> File :Machine Learning in Action->SVD-ImageCompress.py
@IDE    :PyCharm
@Author :YuanPeng Tu
@Date   :2019-11-10 21:10
@Desc   :
=========================================="""
from numpy import *
def imgLoadData(filename):
    myl=[]
    for line in open(filename).readlines():
        newRow=[]
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat=mat(myl)
    return myMat

def analyse_data(sigma,loopnum=20):
    sig2=sigma**2
    sigmasum=sum(sig2)
    for i in range(loopnum):
        sigmaI=sum(sig2[:i+1])
        print('主成分：%s, 方差占比: %s%%' % (format(i+1, '2.0f'), format(sigmaI / sigmasum * 100, '4.2f')))
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1)
            else:
                print(0)
        print('')
def imgCompress(numSV=3,thresh=0.8):
    myMat=imgLoadData('0_5.txt')
    print('****original matrix****')
    printMat(myMat,thresh)
    U,Sigma,VT=linalg.svd(myMat)
    analyse_data(Sigma,20)
    SigRecon=mat(eye(numSV)*Sigma[:numSV])
    reconMat=U[:,:numSV]*SigRecon*VT[:numSV,:]
    print('****reconstructed matrix using %d singular values ****' % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    imgCompress(2)
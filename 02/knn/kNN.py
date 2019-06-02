from numpy import *
import operator

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]    # 获取数据集的大小
    diffMat = tile(inX, (dataSetSize, 1))-dataSet  #将未知的数据先在行上展开与dataSet一样的行数，列上不变，然后与dataSet进行相减
    diffMat = diffMat**2    #平方
    sumMat = diffMat.sum(axis=1)  #每一行上进行求和
    distance = sumMat**0.5    #开放计算距离
    sortedDistIndex = distance.argsort()   #排序，返回索引列表
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndex[i]]   #获得排序完后第i个位置的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #计算标签出现的次数
    sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)   #将前k个标签进行排序，选出出现次数最多的标签作为最后结果
    return sortedClassCount[0][0]
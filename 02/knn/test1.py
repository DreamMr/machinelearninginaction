from numpy import *
import kNN
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

if __name__ == '__main__':
    group, labels = createDataSet()
    ans = kNN.classify0(array([[1.0,0.9]]), dataSet=group, labels=labels, k=2)
    print(ans)

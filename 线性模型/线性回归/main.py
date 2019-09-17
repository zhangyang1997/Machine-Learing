import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    xArr=[]
    yArr=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr,yArr

def standRegres(xArr,yArr):
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("矩阵维奇异矩阵，不能求逆")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

def plotRegression(xCopy,xMat,yHat,yMat):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def get_mse(real,predict):
    return ((real-predict)**2).mean()

if __name__=='__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    A=np.array([3,4,5])
    B=np.array([2,4,6])
    print(get_mse(A,B))
    print(get_mse(yMat,yHat))
    plotRegression(xCopy,xMat,yHat,yMat)
    

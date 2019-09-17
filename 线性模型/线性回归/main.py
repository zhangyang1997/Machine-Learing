import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):#加载数据
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

def standRegres(xArr,yArr):#计算权重向量
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("矩阵维奇异矩阵，不能求逆")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

def plotRegression(xCopy,xMat,yHat,yMat):#画图
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')#拟合的曲线
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=.5)#真实样本点
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def get_mse(real,predict):#计算均方误差
    return ((real-predict)**2).mean()

if __name__=='__main__':
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)#权重向量
    xMat=np.mat(xArr)#x向量
    yMat=np.mat(yArr)#y向量
    xCopy=xMat.copy()#深拷贝x向量
    xCopy.sort(0)#排序
    yHat=xCopy*ws#预测值
    A=np.array([3,4,5])
    B=np.array([2,4,6])
    print(get_mse(A,B))
    print(get_mse(yMat,yHat))
    plotRegression(xCopy,xMat,yHat,yMat)

import matplotlib.pyplot  as plt
import numpy as np

def loadDataSet():#加载数据集
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')#打开文件
    for line in fr.readlines():#逐行读取
        lineArr=line.strip().split()#去回车，放入列表
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):#激活函数
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)#列表转换成矩阵
    labelMat=np.mat(classLabels).transpose()#列表转换成矩阵，并转置
    m,n=np.shape(dataMatrix)#m行数，n列数
    alpaha=0.001#学习率
    maxCycles=500#最大迭代次数
    weights=np.ones((n,1))#权重向量
    print(dataMatrix.shape)
    print(weights.shape)
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=h-labelMat#预测值与真实值的差
        weights=weights-alpaha*dataMatrix.transpose()*error#梯度下降法
    print(weights.getA().type)
    return weights.getA()#返回权重

def plotFig(weights):#画图
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)#矩阵转换为数组
    n=np.shape(dataMat)[0]#计算数据个数
    xcord1=[];ycord1=[]#正样本
    xcord2=[];ycord2=[]#负样本
    for i in range(n):
        if int(labelMat[i])==1:#正样本
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:#负样本
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=.5)#绘制正样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)#绘制负样本
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('FitFig')
    plt.xlabel('X1')
    plt.ylabel('x2')
    plt.show()

if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    weights=gradAscent(dataMat,labelMat)
    print(weights)
    plotFig(weights)
    

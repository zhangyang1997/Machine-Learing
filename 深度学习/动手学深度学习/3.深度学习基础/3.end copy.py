import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import torch.nn as nn
import IPython.display as display
import matplotlib.pyplot as plt
import torchsummary

'''画图'''
def use_svg_display():
    '''用矢量图显示'''
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    '''设置图的尺寸'''
    # 用矢量图显示
    use_svg_display()
    # 设置尺寸
    plt.rcParams['figure.figsize'] = figsize

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    '''作图函数'''
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

# 1.读取数据集
train_data = pd.read_csv('./Datasets/kaggle_house/train.csv')
print("train_data.shape",train_data.shape)
test_data = pd.read_csv('./Datasets/kaggle_house/test.csv')
print("test_data.shape",test_data.shape)
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print("all_features.shape",all_features.shape)

# 2.预处理数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print("numeric_features.shape",numeric_features.shape)
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
print("all_features[numeric_features].shape",all_features[numeric_features].shape)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
print("all_features[numeric_features].shape",all_features[numeric_features].shape)
all_features = pd.get_dummies(all_features, dummy_na=True)
print("all_features.shape",all_features.shape)

n_train = train_data.shape[0]
print("n_train",n_train)
print("all_features[:n_train].values.shape",all_features[:n_train].values.shape)
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
print("train_features.shape",train_features.shape)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
print("test_features.shape",test_features.shape)
print("train_data.SalePrice.values.shape",train_data.SalePrice.values.shape)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
print("train_labels.shape",train_labels.shape)

# 3.训练模型
loss = torch.nn.MSELoss()

def get_net(feature_num):
    # feature_num=331
    net = nn.Linear(feature_num, 1)
    # net=Linear(in_features=331, out_features=1, bias=True)
    # net.parameters()=<generator object Module.parameters at 0x00000198097837C8>
    for param in net.parameters():
        # torch.Size([1, 331])
        # torch.Size([1])
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def log_rmse(net, features, labels):
    #  evaluating 
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        #net(features).shape=torch.Size([730, 1])
        # clipped_preds.log().shape,labels.log().shape=torch.Size([730, 1]) torch.Size([730, 1])
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
        # rmse=torch.Size([])
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    print()
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    print(dataset)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    print(train_iter)
    optimizer = torch.optim.Adam(
        params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(optimizer)
    net = net.float()
    print(net)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    print("len(train_ls)",len(train_ls))
    print("len(test_ls)",len(test_ls))
    return train_ls, test_ls


# 4.K折交叉验证
def get_k_fold_data(k, i, X, y):
    print()
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    print("X.shape[0]",X.shape[0])
    fold_size = X.shape[0] // k
    print("fold_size",fold_size)
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        print("idx",idx)
        X_part, y_part = X[idx, :], y[idx]
        print("X_part.shape",X_part.shape)
        print("y_part.shape",y_part.shape)
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    print("X_train.shape",X_train.shape)
    print("y_train.shape",y_train.shape)
    print("X_valid.shape",X_valid.shape)
    print("y_valid.shape",y_valid.shape)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        print()
        print("len(data)",len(data))
        net = get_net(X_train.shape[1])
        print(net)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        print("len(train_ls)",len(train_ls))
        print("len(valid_ls)",len(valid_ls))
        train_l_sum += train_ls[-1]
        print("train_l_sum",train_l_sum)
        valid_l_sum += valid_ls[-1]
        print("valid_l_sum",valid_l_sum)
        # if i == 0:
            # semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',range(1, num_epochs + 1), valid_ls,['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

# 5.模型选择
k, num_epochs, lr, weight_decay, batch_size = 3, 200, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

'''
fold 0, train rmse 0.170178, valid rmse 0.156934
fold 1, train rmse 0.162093, valid rmse 0.191036
fold 2, train rmse 0.164167, valid rmse 0.168458
fold 3, train rmse 0.167910, valid rmse 0.154566
fold 4, train rmse 0.162863, valid rmse 0.183008
5-fold validation: avg train rmse 0.165442, avg valid rmse 0.170801
k折交叉验证是用来选择超参数的。
'''

# 完整训练集训练+预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    print()
    net = get_net(train_features.shape[1])
    print("net",net)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    # detach可以截断反向传播的梯度流。
    print("preds.shape",preds.shape)
    print("preds.reshape(1, -1).shape",preds.reshape(1, -1)[0])
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    print("test_data['SalePrice'].shape",test_data['SalePrice'].shape)
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    print(submission)
    submission.to_csv('./Datasets/kaggle_house/submission.csv', index=False)

# train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)


'''
train rmse 0.162827  
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 9:08
# @Author  : Jack Zhao
# @Site    : 
# @File    : focal_loss.py
# @Software: PyCharm

# #Desc:
import torch
import datetime
import pandas as pd
import torch.nn as nn
import torchmetrics
import warnings
import numpy as np

from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader,WeightedRandomSampler
from config import opt
from imblearn.metrics import geometric_mean_score
from autofeat import AutoFeatRegressor



warnings.filterwarnings("ignore")

class BCEFocalLoss(nn.Module):
    """注意这里和CE接受维度不一致，这里为B,1而CE为B"""
    def __init__(self, gamma=2, alpha=0.25, reduction='mean',device='cpu'): # 正文是这样效果会最好,试着调参失败
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        self.to(device)

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss).cuda() if opt.GPU_USED else torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss).cuda() if opt.GPU_USED else torch.mean(loss)
        return loss


class MLP(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_target):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_sz, hidden_sz),
            nn.BatchNorm1d(hidden_sz),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_sz, hidden_sz),
            nn.BatchNorm1d(hidden_sz),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_sz, output_target),
        )
        # self.apply(self.weight_init) # 不初始化的效果反而好点

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight) # kaiming用于relu[normal效果不好],正交xav用于Rnn.
            nn.init.constant_(m.bias, 0)
        # elif isinstance(m,nn.BatchNorm1d):
        #     nn.init.constant(m.weight, 1)
        #     nn.init.constant(m.bias, 0)

    def forward(self, X):
        out = self.mlp(X)
        return out


def train_step(model,features,labels):
    model.train()
    model.optimizer.zero_grad()
    predict = model(features)
    loss = model.criterion(predict,labels)
    labels =labels.view(-1,) # 这楼里需要维度变换
    metric =model.metric_func(predict.softmax(dim=-1),labels)
    loss.backward()
    model.optimizer.step()
    return loss.detach().item(),metric.item()

def valid_step(model,features,labels):
    model.eval()
    with torch.no_grad():
        predict = model(features)
        loss = model.criterion(predict,labels)
        labels =labels.view(-1,) # 这楼里需要维度变换
        predict = predict.softmax(dim=-1)
        metric = model.metric_func(predict,labels)
        metric2 =model.metric_func2(predict.softmax(dim=-1),labels)
        model.metric_func3.update(predict,labels)
        # metric3 = model.metric_func3.compute() # 需要在最后compute.这里需要注释
        metric4 = model.metric_func4(predict.softmax(dim=-1), labels)
        metric5 = model.metric_func5(torch.max(predict.softmax(dim=-1),dim=1)[1].cpu().numpy(), labels.cpu().numpy())
    return loss.detach().item(), metric.item(), metric2.item(),0,metric4.item(),metric5.item()

    # return loss.detach().item(), metric.item(), metric2.item(),metric3.item(),metric4.item(),metric5.item()


def get_feature_label(mode,dataset):
    # data = pd.read_csv(r'E:\Code\Pycharm\JOC\data\{}_{}.csv'.format(mode,dataset))
    data = pd.read_csv(r'/data/JOC/data/{}_{}.csv'.format(mode, dataset))
    if dataset != 'wdbc':
       data['TARGET'] = data['TARGET'].apply(lambda x: 0 if x==-1 else x) # 避免为负target出现

    feature_column = data.columns.tolist()
    feature_column.remove("TARGET")

    # 采样比率增高
    classcount = np.bincount(data['TARGET']).tolist()
    train_weights = 1. / torch.tensor(classcount, dtype=torch.float)
    train_sampleweights = train_weights[data['TARGET']]
    train_sampler = WeightedRandomSampler(weights=train_sampleweights, num_samples=len(train_sampleweights))

    # 直接平衡正负样本采样
    if mode =="train":
        features, labels = data[feature_column].values, data['TARGET'].values
        # oversampler = SMOTE(random_state=42)
        # features, labels = oversampler.fit_resample(data[feature_column].values, data['TARGET'].values)
    else:
        features, labels = data[feature_column].values, data['TARGET'].values
    return torch.Tensor(features), torch.Tensor(labels).view(-1,1).long(),train_sampler


# def data_read(dataset):
#     train_df = pd.read_csv(r'E:\Code\Pycharm\JOC\data\train_{}.csv'.format(dataset))
#     test_df = pd.read_csv(r'E:\Code\Pycharm\JOC\data\test_{}.csv'.format(dataset))
#     if dataset != 'wdbc':
#         train_df['TARGET'] = train_df['TARGET'].apply(lambda x: 0 if x == -1 else x)  # 避免为负target出现
#         test_df['TARGET'] = test_df['TARGET'].apply(lambda x: 0 if x == -1 else x)  # 避免为负target出现
#     feature_column = train_df.columns.tolist()
#     feature_column.remove("TARGET")
#     return train_df, test_df, feature_column

# def get_feature_label(mode,train_df,test_df,feature_column):
#     # 自动化特征工程
#     X_train = train_df[feature_column].values
#     y_train = train_df['TARGET'].values
#     X_test = test_df[feature_column].values
#     y_test = test_df['TARGET'].values
#     afreg = AutoFeatRegressor(verbose=1, feateng_steps=2)
#     X_train = afreg.fit_transform(X_train, y_train)
#     X_test = afreg.transform(X_test)
#     if mode == train:
#         return torch.Tensor(X_train), torch.Tensor(y_train).view(-1,1).long()
#     else:
#         return torch.Tensor(X_test), torch.Tensor(y_test).view(-1,1).long()
#     features, labels = torch.Tensor(X_train), torch.Tensor().view(-1,1).long()
#     return features, labels


def train(dataset_name,**kwargs):
    roc_lis, pos_f1_lis, weight_f1_lis, gmeans_lis = [],[],[],[]
    opt.parse(kwargs)
    for i in range(1,6):
        dataset = dataset_name + str(i)
        for j in range(10): # 这里是10次predict训练
            print("========现在处理的是{}第{}个数据集的第{}次训练!".format(dataset_name,str(i),str(j)))
            device = 'cpu'
            use_cuda = opt.GPU_USED
            if use_cuda and torch.cuda.is_available():
                print("cuda ready....")
                device = 'cuda:0'
            x, y,train_sampler = get_feature_label('train',dataset)
            val_x, val_y,_ = get_feature_label('test',dataset)
            # train_df,test_df,feature_column = data_read(dataset) # 自动化特征工程
            # x,y = get_feature_label('train',train_df,test_df,feature_column)
            # val_x,val_y = get_feature_label('valid',train_df,test_df,feature_column)

            input_sz = x.shape[1]
            model = MLP(input_sz, 32, 2).to(device)
            model.optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR, weight_decay=opt.WEIGHT_DECAY)
            model.criterion = BCEFocalLoss(device=device)
            # 定义指标
            model.metric_func = torchmetrics.Accuracy()
            model.metric_func2 = torchmetrics.F1(num_classes=2)
            model.metric_func3 = torchmetrics.AUROC(num_classes=2, pos_label=1)
            model.metric_func4 = torchmetrics.F1(num_classes=2,average='weighted')
            model.metric_func5 = geometric_mean_score # 代替G-mean

            model.metric_name = "ACC"
            model.metric_name2 = "F1_Score"
            model.metric_name3 = "AUC"
            model.metric_name4 = "Weighted_F1"
            model.metric_name5 = "Gmean"


            if use_cuda:
                model.metric_func = model.metric_func.cuda()
                model.metric_func2 = model.metric_func2.cuda()
                model.metric_func3 = model.metric_func3.cuda()
                model.metric_func4 = model.metric_func4.cuda()
                # model.metric_func5 = model.metric_func5.cuda()

            train_data = TensorDataset(x, y)
            train_loader = DataLoader(train_data, batch_size=opt.BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=6,
                                      drop_last=False)
            print("Train Loader Have Done {} Loaders!".format(len(train_loader)))
            val_data = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_data, batch_size=opt.BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=6,
                                    drop_last=False)
            print("Valid Loader Have Done {} Loaders!".format(len(val_loader)))
            print("All DataLoaders Have Done! Please Wait for Training!")
            metric_name, metric_name2, metric_name3,metric_name4, metric_name5 = model.metric_name, model.metric_name2, model.metric_name3,model.metric_name4,model.metric_name5
            dfhistory = pd.DataFrame(
                columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name, 'val_' + metric_name2,
                         'val_' + metric_name3, 'val_' + metric_name4,\
                         'val_' + metric_name5])
            print("Start Training !")

            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            print("=====" * 8 + "%s" % now_time)

            best_gmean = 0.0
            for epoch in range(1, opt.EPOCHES + 1):
                loss_sum = 0.
                metric_sum = 0.
                step = 1
                for step, (features, labels) in enumerate(train_loader, 1):
                    if use_cuda:
                        features = features.to(device)
                        labels = labels.to(device)

                    loss, metric = train_step(model, features, labels)
                    loss_sum += loss
                    metric_sum += metric

                    if step % opt.LOG_FREQ == 0:
                        print(
                            ('[step = %d] loss: %.3f, ' + metric_name + " %.3f,") % (step, loss_sum / step, metric_sum / step))
                val_loss_sum = 0.
                val_metric_sum = 0.
                val_metric_sum2 = 0.
                val_metric_sum3 = 0.
                val_metric_sum4 = 0.
                val_metric_sum5 = 0.

                val_step = 1
                for val_step, (features, labels) in enumerate(val_loader, 1):
                    if use_cuda:
                        features = features.to(device)
                        labels = labels.to(device)

                    loss, metric, metric2, metric3,metric4,metric5 = valid_step(model, features, labels)
                    val_loss_sum += loss
                    val_metric_sum += metric
                    val_metric_sum2 += metric2
                    val_metric_sum3 += metric3
                    val_metric_sum4 += metric4
                    val_metric_sum5 += metric5
                # info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step,
                #         val_metric_sum2 / val_step, val_metric_sum3 / val_step,val_metric_sum4 / val_step, val_metric_sum5 / val_step)
                info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step,
                        val_metric_sum2 / val_step, model.metric_func3.compute().item(),val_metric_sum4 / val_step, val_metric_sum5 / val_step) # 这种方式避免出现batch中全0的情况。
                dfhistory.loc[epoch - 1] = info
                if val_metric_sum5 / val_step > best_gmean:
                    torch.save(model.state_dict(), opt.WEIGHTS + "BEST_{}.pth".format(epoch))
                    best_gmean = val_metric_sum5 / val_step
                    # pos_f1,roc,weight_f1,gmeans = val_metric_sum2 / val_step,val_metric_sum3 / val_step,val_metric_sum4 / val_step,best_gmean
                print(("\nEPOCH = %d, loss = %.3f," + metric_name + " = %.3f," + \
                       "val_loss = %.3f," + "val_" + metric_name + " = %.3f," + "val_" + metric_name2 + " = %.3f," + "val_" + metric_name3 +
                       " = %.3f,"+ "val_" + metric_name4 + " = %.3f," + "val_" + metric_name5 + " = %.3f,") % info)
                now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("\n" + "==========" * 8 + "%s" % now_time)
                model.metric_func.reset()
                model.metric_func2.reset()
                model.metric_func3.reset()
                model.metric_func4.reset()
                # model.metric_func5.reset()

            # 下面仿照display的计算方式,这里暂时存疑
            pos_f1,roc,weight_f1,gmeans=dfhistory['val_F1_Score'].max(),dfhistory['val_AUC'].max(),\
                                        dfhistory['val_Weighted_F1'].max(),dfhistory['val_Gmean'].max()
            roc_lis.append(roc)
            pos_f1_lis.append(pos_f1)
            weight_f1_lis.append(weight_f1)
            gmeans_lis.append(gmeans)
            print(roc_lis)
    return roc_lis, np.mean(roc_lis), np.std(roc_lis),pos_f1_lis,np.mean(pos_f1_lis),np.std(pos_f1_lis),\
                weight_f1_lis, np.mean(weight_f1_lis), np.std(weight_f1_lis), gmeans_lis,np.mean(gmeans_lis),np.std(gmeans_lis)



if __name__ == '__main__':
    dataset_names = [ 'aba', 'bal', 'hab', 'hou', 'let', 'wdbc', 'wpbc', 'yea', 'pim', 'p1', 'p2', 'p3', 'cre']
    metrics = ['roc', 'avg_roc', 'std_roc', 'pos_f1', 'avg_pos_f1', 'std_pos_f1', 'weight_f1', 'avg_weight_f1',
     'std_weight_f1', 'gmean', 'avg_gmean', 'std_gmean']
    df = pd.DataFrame(columns=metrics, index=dataset_names)
    for data_name in dataset_names[9:12]: # 这些都是为了分布式的跑
        print("正在处理{}".format(data_name))
        # opt.Dataset = data_name
        metrics = train(data_name)
        df.loc[data_name] = metrics
    # df.to_csv('../reuslt/focal_loss{}.csv'.format("1"))
    df.to_csv('/data/JOC/result/focal_loss{}.csv'.format("4")) # 一开始命名错了，搞成reuslt

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 10:21
# @Author  : Jack Zhao
# @Site    : 
# @File    : hemc.py
# @Software: PyCharm

# #Desc: 这里是HEM_CLASSIFIER的重构

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score






class HemClass:
    def __init__(self,dataset):
        self.dataset_train = r'E:\Code\Pycharm\JOC\data\train_{}'.format(dataset)
        self.dataset_test = r'E:\Code\Pycharm\JOC\data\test_{}'.format(dataset)
        self.roc = []
        self.pos_f1 = []
        self.weight_f1 = []
        self.gmean = []

    def fit_base_est(self, model,n_est, feature, target):
        """
        :param model: 基学习器
        :param n_est: 估计器数量
        :param feature:
        :param target:
        :return:
        """
        if model=='Adaboost':
            model = AdaBoostClassifier(n_estimators=n_est)
            # print("Adaboost Training!")
        elif model == 'Bagging':
            model = BaggingClassifier(n_estimators=n_est, n_jobs=-1)
        base_est_fit = model.fit(feature, target)
        return base_est_fit


    def data_read(self,i):
        """
        # p2需要特殊处理
        :param i: 文件序号
        :return:
        """
        train_df = pd.read_csv(self.dataset_train + str(i) + '.csv')
        test_df = pd.read_csv(self.dataset_test + str(i) + '.csv')
        if self.dataset_train.split('_')[-1] =='wdbc': # 标签不一致
            train_df.loc[train_df.TARGET == 0, 'TARGET'] = -1
            test_df.loc[test_df.TARGET == 0, 'TARGET'] = -1
        feature_column = train_df.columns.tolist()
        feature_column.remove("TARGET")

        train_copy = train_df.copy()  # 每次测试base_est,S
        test_copy = test_df.copy()
        pos_num = train_df[train_df.TARGET == 1].shape[0]
        neg_num = train_df[train_df.TARGET == -1].shape[0]
        pos_train_df = train_df.loc[train_df.TARGET == 1]
        train_copy['sum'] = 0  # 每次预测结果之和

        # 开始筛选样本
        ratio = neg_num / pos_num
        return train_df, test_df, feature_column,train_copy,test_copy,pos_num,neg_num,pos_train_df,ratio

    def remove_hard_exam(self,pos_train_df,feature_column,train_copy,removed_neg_df):
        """
        :param pos_train_df: 少数类df
        :param feature_column: columns list
        :param train_copy: 完整训练集S
        :param removed_neg_df: 删除了多数类的训练集
        :return: removed_neg_df
        """
        neg_sample = removed_neg_df[removed_neg_df.TARGET==-1].sample(pos_train_df.shape[0])
        removed_neg_df = removed_neg_df.drop(neg_sample.index) # 删除负样本,Ni
        sub_sample = pd.concat((neg_sample,pos_train_df),axis=0)
        base_est_fit = self.fit_base_est('Adaboost',10,sub_sample[feature_column].values, sub_sample['TARGET'].values)
        preds = base_est_fit.predict_proba(train_copy[feature_column].values)[:,-1]
        iter_result = [1 if pred > 0.5 else -1 for pred in preds] # 这里显示两种写法
        train_copy['iter_result'] = iter_result
        train_copy['sum'] += train_copy['iter_result']

        print('训练集Shape:', removed_neg_df.shape)

        return removed_neg_df


    def whether_divide(self,ratio,train_df,feature_column,pos_train_df,train_copy,pos_num):
        """
        两种情况讨论
        :param ratio:neg/pos
        :param train_df:
        :param feature_column:
        :param pos_train_df:
        :param train_copy:==train_df，避免在train上修改
        :param pos_num:==pos_train_df.shape
        :return:
        """
        if round(ratio) > ratio:
            # print(ratio)
            # 多取样一次
            removed_neg_df = train_df.copy()
            for i in range(1, round(ratio)):
                removed_neg_df = self.remove_hard_exam(pos_train_df, feature_column, train_copy, removed_neg_df)

            add_sample = train_df.drop(removed_neg_df.index).sample(2 * pos_num - removed_neg_df.shape[0])  # 额外抽取 TODO: sample的随机种子
            last_subsample = pd.concat((add_sample, removed_neg_df), axis=0)
            base_est_fit = self.fit_base_est('Adaboost', 10, last_subsample[feature_column].values,
                                             last_subsample["TARGET"].values)
            preds = base_est_fit.predict(train_copy[feature_column].values)
            train_copy['iter_result'] = preds
            train_copy['sum'] += train_copy['iter_result']
        else:
            removed_neg_df = train_df.copy()
            for i in range(1, round(ratio) + 1):
                removed_neg_df = self.remove_hard_exam(pos_train_df, feature_column, train_copy, removed_neg_df)
        pos_hard_index = list(
            train_copy.loc[(train_copy['sum'] == (-round(ratio))) & (train_copy['TARGET'] == 1)].index)
        neg_hard_index = list(
            train_copy.loc[(train_copy['sum'] == round(ratio)) & (train_copy['TARGET'] == -1)].index)

        pos_hard_nums = len(pos_hard_index)
        neg_hard_nums = len(neg_hard_index)
        return pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index


    def apply_all(self):
        for i in range(1,6):
            # 读取数据
            train_df, test_df, feature_column, train_copy, test_copy, pos_num, neg_num,pos_train_df,ratio = self.data_read(i)
            # 删除样本
            pos_hard_nums, neg_hard_nums, pos_hard_index, neg_hard_index = self.whether_divide(ratio,train_df,feature_column,pos_train_df,train_copy,pos_num)
            train_df = train_df.drop(pos_hard_index)
            train_df = train_df.drop(neg_hard_index)
            print('删除样本量%d', len(pos_hard_index + neg_hard_index))

            self.predict(10,train_df,test_df,feature_column)




    def predict(self,nums,train_df,test_df,feature_column):
        """
        :param nums:验证次数
        :return:
        """
        for j in range(nums):
            # 训练10个分类器，检测10次
            base_est_fit = self.fit_base_est('Adaboost', 150, train_df[feature_column].values,
                                             train_df['TARGET'].values)
            test_preds = base_est_fit.predict(test_df[feature_column].values)
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))

            # new
            self.gmean.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))


    def display(self):
        print("roc : ", self.roc, ", pos_f1: ", self.pos_f1, ", weight_f1", self.weight_f1, ", gmean", self.gmean)
        print("avg_roc : ", np.mean(self.roc), ", avg_pos_f1: ", np.mean(self.pos_f1),
              ", avg_weight_f1", np.mean(self.weight_f1),
              ", avg_gmean", np.mean(self.gmean))
        print("std_roc : ", np.std(self.roc), ", std_pos_f1: ", np.std(self.pos_f1),
              ", std_weight_f1", np.std(self.weight_f1),
              ", std_gmean", np.std(self.gmean))
        return self.roc, np.mean(self.roc), np.std(self.roc), self.pos_f1, np.mean(self.pos_f1), np.std(self.pos_f1), \
               self.weight_f1, np.mean(self.weight_f1), np.std(self.weight_f1), self.gmean, np.mean(self.gmean), np.std(
            self.gmean)


if __name__ == '__main__':
    model = HemClass('p3')
    model.apply_all()
    model.display()


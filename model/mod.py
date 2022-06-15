#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 20:32
# @Author  : Jack Zhao
# @Site    : 
# @File    : mod.py
# @Software: PyCharm

# #Desc:MOD算法的实现

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from imblearn.metrics import geometric_mean_score

import smote_variants as sv
import pandas as pd
import numpy as np


class Mod:
    def __init__(self, dataset):
        self.roc = []
        self.pos_f1 = []
        self.weight_f1 = []
        self.gmean = [] # 新增metrics
        self.dataset_train = r'E:\Code\Pycharm\JOC\data\train_{}'.format(dataset) # 相对路径只能通过单元测试
        self.dataset_test = r'E:\Code\Pycharm\JOC\data\test_{}'.format(dataset)  # 相对路径只能通过单元测试

    def data_read(self,i):
        train_df = pd.read_csv(self.dataset_train + str(i) + '.csv')
        test_df = pd.read_csv(self.dataset_test + str(i) + '.csv')
        feature_column = train_df.columns.tolist()
        # 采样
        feature_column.remove("TARGET")
        X, y = train_df[feature_column].values, train_df['TARGET'].values
        oversampler = sv.SMOTE_D()
        X_samp, y_samp = oversampler.sample(X, y)
        return train_df,test_df,feature_column,X_samp,y_samp

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
            # print("Adaboost Training!") 为了美观，不打印了
        base_est_fit = model.fit(feature, target)
        return base_est_fit


    def apply_all(self):
        for i in range(1, 6):
            train_df,test_df,feature_column,X_samp,y_samp = self.data_read(str(i))
            self.predict(10,feature_column,X_samp,y_samp,test_df)

    def predict(self,num,feature_column,X_samp,y_samp,test_df):
        for j in range(0, num):
            base_est_fit = self.fit_base_est('Adaboost', 150, X_samp, y_samp)
            test_preds = base_est_fit.predict(test_df[feature_column].values)
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))
            # new
            self.gmean.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))


    def display(self):
        print("roc : ", self.roc, ", pos_f1: ", self.pos_f1, ", weight_f1", self.weight_f1,", gmean", self.gmean)
        print("avg_roc : ", np.mean(self.roc), ", avg_pos_f1: ", np.mean(self.pos_f1),
              ", avg_weight_f1",np.mean(self.weight_f1),
              ", avg_gmean",np.mean(self.gmean))
        print("std_roc : ", np.std(self.roc), ", std_pos_f1: ", np.std(self.pos_f1),
              ", std_weight_f1",np.std(self.weight_f1),
              ", std_gmean",np.std(self.gmean))
        return self.roc, np.mean(self.roc), np.std(self.roc),self.pos_f1,np.mean(self.pos_f1),np.std(self.pos_f1),\
                self.weight_f1, np.mean(self.weight_f1), np.std(self.weight_f1), self.gmean,np.mean(self.gmean),np.std(self.gmean)


if __name__ == '__main__':
    model = Mod('let')
    model.apply_all()
    model.display()


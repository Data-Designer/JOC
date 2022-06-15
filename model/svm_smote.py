#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 10:48
# @Author  : Jack Zhao
# @Site    : 
# @File    : svm_smote.py
# @Software: PyCharm

# #Desc: SVM+Smote
from model.mod import Mod
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score



class SvmS(Mod):
    def __init__(self,dataset):
        super(SvmS, self).__init__(dataset)

    def data_read(self,i):
        train_df = pd.read_csv(self.dataset_train + str(i) + '.csv')
        test_df = pd.read_csv(self.dataset_test + str(i) + '.csv')
        feature_column = train_df.columns.tolist()
        # 采样
        feature_column.remove("TARGET")
        X, y = train_df[feature_column].values, train_df['TARGET'].values
        oversampler = SMOTE(random_state=42)
        X_samp, y_samp = oversampler.fit_resample(X, y)

        return train_df,test_df,feature_column,X_samp,y_samp

    def fit_base_est(self, model, kernel, feature, target):
        """
        :param model: 基学习器
        :param n_est: 估计器数量
        :param feature:
        :param target:
        :return:
        """
        if model=='svm':
            model = svm.SVC(kernel=kernel,max_iter=1000)
            # print("Svm Training!")
        base_est_fit = model.fit(feature, target)
        return base_est_fit

    def predict(self,num,feature_column,X_samp,y_samp,test_df):
        for j in range(0, num):
            base_est_fit = self.fit_base_est('svm', 'rbf', X_samp, y_samp)
            test_preds = base_est_fit.predict(test_df[feature_column].values)
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))
            # new
            self.gmean.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))

if __name__ == '__main__':
    model = SvmS('let')
    model.apply_all()
    model.display()
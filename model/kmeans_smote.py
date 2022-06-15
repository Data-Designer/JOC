#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 10:48
# @Author  : Jack Zhao
# @Site    : 
# @File    : svm_smote.py
# @Software: PyCharm

# #Desc: Kmeans+Smote
from model.mod import Mod
import pandas as pd
from smote_variants import kmeans_SMOTE
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier


from sklearn.preprocessing import StandardScaler



class KS(Mod):
    def __init__(self, dataset):
        super(KS, self).__init__(dataset)
        imr_dic = {'aba':9.7,'bal':11.8,'hab':2.8,'hou':3.8,'let':24.3,'wdbc':1.7,'wpbc':3.2,
                   'yea':8.1,'pim':1.9,'p1':10,'p2':5,'p3':2,'cre':3.5}
        self.dst_value = imr_dic[dataset]

    def data_read(self,i):
        train_df = pd.read_csv(self.dataset_train + str(i) + '.csv')
        test_df = pd.read_csv(self.dataset_test + str(i) + '.csv')
        feature_column = train_df.columns.tolist()
        # 采样
        feature_column.remove("TARGET")
        X, y = train_df[feature_column].values, train_df['TARGET'].values
        # 归一化
        # scalar = StandardScaler()
        # X = scalar.fit_transform(X)
        # train_df[feature_column] = X
        # test_df[feature_column] = scalar.transform(test_df[feature_column].values)
        oversampler = kmeans_SMOTE(irt=self.dst_value,n_jobs= -1,n_clusters=5)
        X_samp, y_samp = oversampler.sample(X, y)

        return train_df,test_df,feature_column,X_samp,y_samp

    def fit_base_est(self, model, kernel, feature, target):
        """
        :param model: 基学习器
        :param n_est: 估计器数量
        :param feature:
        :param target:
        :return:
        """
        if model =='svm':
            model = svm.SVC(kernel=kernel,max_iter=1000)
            model = model.fit(feature, target)
            # print("Svm Training!")
        elif model == "adaboost":
            base_estimator = AdaBoostClassifier(n_estimators=150)
            model = EasyEnsembleClassifier(n_estimators=50, base_estimator=base_estimator, n_jobs=-1)
            model = model.fit(feature, target)
        elif model == 'bag':
            model = BaggingClassifier(n_estimators=100, n_jobs=-1)  # 150
            model = model.fit(feature, target)

        return model

    def predict(self,num,feature_column,X_samp,y_samp,test_df):
        for j in range(0, num):
            base_est_fit = self.fit_base_est('adaboost', 'rbf', X_samp, y_samp) # p1-p3重新选择
            test_preds = base_est_fit.predict(test_df[feature_column].values)
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))
            # new
            self.gmean.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))


if __name__ == '__main__':
    model = KS('p1')
    model.apply_all()
    model.display()
    model = KS('p2')
    model.apply_all()
    model.display()
    model = KS('p3')
    model.apply_all()
    model.display()

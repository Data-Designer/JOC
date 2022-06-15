#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 21:06
# @Author  : Jack Zhao
# @Site    : 
# @File    : other_bs.py
# @Software: PyCharm

# #Desc:
import pandas as pd
from model.mod import Mod
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
from imblearn.metrics import geometric_mean_score
from deslib.des import DESP





class OtherBaseline(Mod):
    def __init__(self,dataset):
        self.dataset = dataset
        super(OtherBaseline, self).__init__(dataset)

    def data_read(self,i):
        """读取数据"""
        train_df = pd.read_csv(self.dataset_train + str(i) + '.csv')
        test_df = pd.read_csv(self.dataset_test + str(i) + '.csv')
        feature_column = train_df.columns.tolist()
        # 采样
        feature_column.remove("TARGET")
        return train_df, test_df, feature_column

    def apply_all(self, model):
        """五折交叉验证，提前数据集已经分好"""
        for i in range(1, 6):
            train_df, test_df, feature_column = self.data_read(str(i))
            self.predict(10,model,feature_column, train_df, test_df)


    def fit_base_est(self, model,n_est,feature,target):
        """
        训练拟合
        :param model: 基学习器
        :param n_est: 估计器数量
        :param feature:
        :param target:
        :return:
        """
        if model=='Adaboost':
            model = AdaBoostClassifier(n_estimators=n_est) # 150
        elif model=='BaggingClassifier':
            model = BaggingClassifier(n_estimators=150, n_jobs=-1) # 150
        elif model=='RUSBoost':
            base_estimator = AdaBoostClassifier(n_estimators=10) # 10
            model = RUSBoostClassifier(n_estimators=15, base_estimator=base_estimator) # 15
        elif model=='EasyEnsemble':
            base_estimator = AdaBoostClassifier(n_estimators=10)
            model = EasyEnsembleClassifier(n_estimators=15, base_estimator=base_estimator, n_jobs=-1)
        elif model == 'SelfPacedEnsemble':
            base_estimator = AdaBoostClassifier(n_estimators=10)
            model = SelfPacedEnsembleClassifier(base_estimator,n_estimators=15)
        elif model == 'DES':
            # pool_classifiers = BaggingClassifier(RandomForestClassifier(n_estimators=10,
            #                               max_depth=10),n_estimators=150)
            # pool_classifiers = RandomForestClassifier(n_estimators=150,
            #                               max_depth=10)
            model = AdaBoostClassifier(n_estimators=150)
            pool_classifiers = EasyEnsembleClassifier(n_estimators=15, base_estimator=model, n_jobs=-1)
            model = pool_classifiers

        base_est_fit = model.fit(feature, target)
        return base_est_fit


    def predict(self,num,model,feature_column,train_df,test_df):
        """跑10次"""
        for j in range(num):
            base_est_fit = self.fit_base_est(model, 150, train_df[feature_column].values, train_df['TARGET'].values)
            if model == "DES":
                des = DESP(base_est_fit, DFP=True).fit(train_df[feature_column].values, train_df['TARGET'].values)
                test_preds = des.predict(test_df[feature_column].values)
            else:
                test_preds = base_est_fit.predict(test_df[feature_column].values)
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))

            # new
            self.gmean.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))



if __name__ == '__main__':
    model = OtherBaseline('bal')
    # model.apply_all('BaggingClassifier')
    model.apply_all("DES")
    model.display()




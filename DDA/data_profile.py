from datetime import date
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve

# 载入数据
drug_name = pd.read_csv('DDA/data/drug.csv')
z_means = pd.read_csv('DDA/data/z_mean_1.csv')
dgern_2= pd.read_csv('result/feature1_means.csv')
# 将DGERN-1按照DDA的药物顺序排序
z_means1 = z_means.loc[z_means['name'].str.lower().isin(drug_name['name'].str.lower())]
drug_name1 = drug_name.loc[drug_name['name'].str.lower().isin(z_means['name'].str.lower())]
z_means1.index = z_means1['name'].str.lower()
# 将DGERN-2按照DDA的药物顺序排列
dgern_2_1 = dgern_2.loc[dgern_2['iname'].str.lower().isin(drug_name['name'].str.lower())]
drug_name1 = drug_name.loc[drug_name['name'].str.lower().isin(dgern_2['iname'].str.lower())]
dgern_2_1.index = dgern_2_1['iname'].str.lower()
# DGERN_1获得要用的矩阵
z_means2 = z_means1.reindex(drug_name1['name'].str.lower())
z_means2 = z_means2.iloc[:,1:]

# 筛选后药物的DDA还剩的index
drug_index = drug_name1.index.values.tolist()

# 读取DDA的数据
data = scio.loadmat('DDA/data/SCMFDD_Dataset.mat')
# 获得键
data.keys()
data['chemical_list']
data['disease_list']
data['enzyme_feature_matrix'].shape
data['structure_feature_matrix'][drug_index].shape
data['normalized_dis_similairty_matrix']
data['target_sequence_list'].shape
data['target_feature_matrix'].shape
data['pathway_feature_matrix'].shape
data['drug_disease_association_matrix'][drug_index].shape
dda = data['drug_disease_association_matrix'][drug_index]
dda.sum()
# 将药物与疾病的向量加起来
def feature_combine(drug, disease):
    features = []
    for i in range(drug.shape[0]):
        for j in range(disease.shape[0]):
            f = drug[i].tolist() + disease[j].tolist() + [i] + [j]
            features.append(f)
    return np.array(features)
# 定义AUPR评价函数
def aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    return aupr
aupr_score = make_scorer(aupr,  greater_is_better=True)
# test
shape = [155, 598]
def feature_combine(shape):
    features = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            dda = data['drug_disease_association_matrix'][drug_index]
            dda[i][j] = 0
            f = dda[i].tolist() + dda[:,j].tolist()
            features.append(f)
    return np.array(features)
features = feature_combine(shape)
scoring = {'AUC':'roc_auc', 'AUPR':aupr_score, 'ACC':'accuracy', 'F1':'f1_macro'}
def logistic_f1():
    result = []
    clf = LogisticRegression()
    scores = cross_validate(clf, features, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    return result

result1 = logistic_f1()

clf = LogisticRegression()
scores = cross_validate(clf, expression_dis, label, cv=3, scoring=scoring)
# 获得说有药物的特征向量
drug_enzyme = data['enzyme_feature_matrix'][drug_index]
drug_target = data['target_feature_matrix'][drug_index]
drug_structure = data['structure_feature_matrix'][drug_index]
drug_pathway = data['pathway_feature_matrix'][drug_index]
drug_expression = z_means2.values
# 疾病的向量
disease = data['normalized_dis_similairty_matrix']
# 加和起来的向量
enzyme_dis = feature_combine(drug_enzyme, disease)
target_dis = feature_combine(drug_target, disease)
structure_dis = feature_combine(drug_structure, disease)
pathway_dis = feature_combine(drug_pathway, disease)
expression_dis = feature_combine(drug_expression, disease)

# dda矩阵
dda = data['drug_disease_association_matrix'][drug_index]
# 标签
label = data['drug_disease_association_matrix'][drug_index]
label = label.flatten()
# 评价指标
scoring = {'AUC':'roc_auc', 'AUPR':aupr_score, 'ACC':'accuracy', 'F1':'f1_macro'}
# 各机器学习的计算
def logistic_f1():
    result = []
    clf = LogisticRegression(max_iter=500)
    scores = cross_validate(clf, enzyme_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, pathway_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, structure_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, target_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, expression_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])

    return result

result1 = logistic_f1()
result1_df = pd.DataFrame(result1, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

def randomforest_f1():
    result = []
    clf = RandomForestClassifier()
    scores = cross_validate(clf, enzyme_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, pathway_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, structure_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, target_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, expression_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])

    return result

result2 = randomforest_f1()
result2_df = pd.DataFrame(result2, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

# def AdaBoostClassifier_f1():
#     result = []
#     clf = AdaBoostClassifier(n_estimators=100)
#     scores = cross_validate(clf, enzyme_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, pathway_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, structure_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, target_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, expression_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     return result

# result3 = AdaBoostClassifier_f1()

# from sklearn.naive_bayes import GaussianNB

# def GNB():
#     result = []
#     clf = GaussianNB()
#     scores = cross_validate(clf, enzyme_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, pathway_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, structure_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, target_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     scores = cross_validate(clf, expression_dis, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
#     result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
#     return result                                           

# result4 = GNB()

def XGB():
    result = []
    clf = XGBClassifier()
    scores = cross_validate(clf, enzyme_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, pathway_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, structure_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, target_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, expression_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])

    return result

result5 = XGB()
result5_df = pd.DataFrame(result5, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])


def MLP():
    result = []
    clf = MLPClassifier()
    scores = cross_validate(clf, enzyme_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, pathway_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, structure_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, target_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, expression_dis, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])

    return result

result6 = MLP()
result6_df = pd.DataFrame(result6, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

results = [result1, result2, result5, result6]
results_np = np.asarray(results)
results_np.shape
results_np_f = results_np.flatten()
results_np_f = pd.Series(results_np_f)
results_np_f.name = 'value'
# 一维标签
index1 = pd.Series(['roc_auc', 'average_precision', 'accuracy', 'f1'])
# 二维标签
index2 = pd.Series(['enzyme', 'pathway', 'structure', 'target', 'DGERN-1', 'DGERN-2'])
# 三维标签
index3 = pd.Series(['Logistic', 'Randomforest', 'XGBoost', 'Multi-layer Perceptron'])
# index的笛卡尔乘积。注意：高维在前，低维在后
prod = itertools.product(index3, index2, index1 )
# 转换为DataFrame
prod = pd.DataFrame([x for x in prod])
prod.columns = ['methods', 'data', 'scores']
result_all = pd.concat([prod, results_np_f], axis=1)
result_all.to_csv('DDA/dataresults_DGERN.csv')
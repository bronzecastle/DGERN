import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import itertools
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
sys.path.append(r"/home/DGERN/DDI")
import overlap
import pickle
# 获得标签
label_matrix = pd.read_csv('embeddings/label_matrix.csv', index_col=0, header=0)
label = []

for i in range(1,365):
    for j in range(i+1,366):
        label.append(label_matrix.iloc[i-1,j-1])
label = np.array(label)

# 将向量正确的排序
def re_index_vector(loc):
    id_drug = pd.read_csv('embeddings/id_drug.csv', header=None)
    dic_id = {}
    for i in range(365):
        dic_id[id_drug[1][i]] = id_drug[0][i]
    dv = pd.read_csv(loc, header=None)
    for i in range(365):
        dv = dv.replace(dv[0][i],dic_id[dv[0][i]])
    dv.index = dv[0]
    dv = dv.drop([0], axis=1)
    dv = dv.sort_index()
    return dv

# 组合成药物对数据    
def make_data(e_embeddings):
    e_list = []
    for i in range(1,365):
        for j in range(i+1,366):
            v_plus = e_embeddings.iloc[i-1].to_list()+e_embeddings.iloc[j-1].to_list()
            e_list.append(v_plus)
    e_array = np.array(e_list)
    return e_array

e_embeddings = pd.read_csv('data/embeddings/e_embeddings.csv', index_col=0, header=None)
e_array = make_data(e_embeddings)

p_embeddings = pd.read_csv('data/embeddings/p_embeddings.csv', index_col=0, header=None)
p_array = make_data(p_embeddings)

s_embeddings = pd.read_csv('data/embeddings/s_embeddings.csv', index_col=0, header=None)
s_array = make_data(s_embeddings)

t_embeddings = pd.read_csv('data/embeddings/t_embeddings.csv', index_col=0, header=None)
t_array = make_data(t_embeddings)

# 组合两个函数，获得expression数据的药物对向量
def make_dv(loc):
    dv = re_index_vector(loc)
    dv_array = make_data(dv)
    return dv_array
# DGERN-1模型的组合
dv_01 = make_dv('data/drugs_vector_0.1_2.csv')
dv_05 = make_dv('data/drugs_vector_0.5_2.csv')
dv_1 = make_dv('data/drugs_vector1.csv')
dv_2 = make_dv('data/drugs_vector_2_2.csv')
dv_4 = make_dv('data/drugs_vector_4_2.csv')
dv_6 = make_dv('data/drugs_vector_6_2.csv')
dv_7 = make_dv('data/drugs_vector_7_2.csv')
dv_9 = make_dv('data/drugs_vector_9_2.csv')
dv_0r8 = make_dv('data/drugs_vector_0r8_2.csv')
dv_0r9 = make_dv('data/drugs_vector_0r9_2.csv')
dv_1r1 = make_dv('data/drugs_vector_1r1_2.csv')
dv_1r2 = make_dv('data/drugs_vector_1r2_2.csv')
dv_1r3 = make_dv('data/drugs_vector_1r3_2.csv')

# 定义指标
def aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    return aupr
aupr_score = make_scorer(aupr,  greater_is_better=True)
scoring = {'AUC':'roc_auc', 'AUPR':aupr_score, 'ACC':'accuracy', 'F1':'f1_macro'}
# 机器学习方法
def svm_f1():
    result = []
    clf = svm.SVC(kernel='rbf', C=1)
    scores = cross_val_score(clf, e_array, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, p_array, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, s_array, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, t_array, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_1, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    return result                                                

results = svm_f1()
# 仅计算dv的巡游参数结果
def svm_f1():
    result = []
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, dv_01, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_05, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_1, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_2, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_4, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_6, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_7, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    scores = cross_val_score(clf, dv_9, label, cv=3, scoring='f1_macro')
    result.append(scores.mean())
    return result

results = svm_f1()

def logistic_f1():
    result = []
    clf = LogisticRegression(max_iter=500)
    scores = cross_validate(clf, e_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, p_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, s_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, t_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dv_1, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dgern_2, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    return result

result1 = logistic_f1()
result1_df = pd.DataFrame(result1, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

def randomforest_f1():
    result = []
    clf = RandomForestClassifier()
    scores = cross_validate(clf, e_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, p_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, s_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, t_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dv_1, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dgern_2, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    return result

result2 = randomforest_f1()
result2_df = pd.DataFrame(result2, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

def AdaBoostClassifier_f1():
    result = []
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_validate(clf, e_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, p_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, s_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, t_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, dv_1, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    return result

result3 = AdaBoostClassifier_f1()


def GNB():
    result = []
    clf = GaussianNB()
    scores = cross_validate(clf, e_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, p_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, s_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, t_array, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    scores = cross_validate(clf, dv_1, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    return result                                           

result4 = GNB()

def XGB():
    result = []
    clf = XGBClassifier()
    scores = cross_validate(clf, e_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, p_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, s_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, t_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dv_1, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dgern_2, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    return result

result5 = XGB()
result5_df = pd.DataFrame(result5, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

def MLP():
    result = []
    clf = MLPClassifier()
    scores = cross_validate(clf, e_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, p_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, s_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, t_array, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dv_1, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    scores = cross_validate(clf, dgern_2, label, cv=3, scoring=scoring)
    result.append([scores['test_AUC'].mean(), scores['test_AUPR'].mean(), scores['test_ACC'].mean(), scores['test_F1'].mean()])
    return result

result6 = MLP()
result6_df = pd.DataFrame(result6, index=['enzyme','pathway','structure','target','DGERN-1','DGERN-2'],columns=['AUC','AUPR','accuracy','f1'])

results = [result1, result2, result3, result4, result5, result6]
results_np = np.asarray(results)
results_np.shape
results_np_f = results_np.flatten()
results_np_f = pd.Series(results_np_f)
results_np_f.name = 'value'
# 一维标签
index1 = pd.Series(['roc_auc', 'average_precision', 'accuracy', 'f1'])
# 二维标签
index2 = pd.Series(['enzyme', 'pathway', 'structure', 'target', 'expression'])
# 三维标签
index3 = pd.Series(['Logistic', 'Randomforest', 'AdaBoost', 'GaussianNB', 'XGBoost', 'Multi-layer Perceptron'])
# index的笛卡尔乘积。注意：高维在前，低维在后
prod = itertools.product(index3, index2, index1 )
# 转换为DataFrame
prod = pd.DataFrame([x for x in prod])
prod.columns = ['methods', 'data', 'scores']
result_all = pd.concat([prod, results_np_f], axis=1)
result_all.to_csv('data/result/results_muti_scores.csv')

def randomforest_f1(dv):
    result = []
    clf = RandomForestClassifier()
    scores = cross_validate(clf, dv, label, cv=3, scoring=('roc_auc', 'average_precision', 'accuracy', 'f1_macro'))
    result.append([scores['test_roc_auc'].mean(), scores['test_average_precision'].mean(), scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()])
    return result

result_0r8 = randomforest_f1(dv_0r8)
result_0r9 = randomforest_f1(dv_0r9)
result_1r1 = randomforest_f1(dv_1r1)
result_1r2 = randomforest_f1(dv_1r2)
result_1r3 = randomforest_f1(dv_1r3)

results_r = [result_0r8[0], result_0r9[0], result_1r1[0], result_1r2[0], result_1r3[0]]
results_r = pd.DataFrame(results_r)
results_r.index = ['lambda0.8', 'lambda0.9', 'lambda1.1', 'lambda1.2', 'lambda1.3']
results_r.columns = ['roc_auc', 'average_precision', 'accuracy', 'f1']
results_r.to_csv('data/result/search_randomforest_para.csv')

# DGERN-1和DGERN-2模型的机器学习结果整合
results = [result1, result2, result5, result6]
results_np = np.asarray(results)
results_np.shape
results_np_f = results_np.flatten()
results_np_f = pd.Series(results_np_f)
results_np_f.name = 'value'
# 一维标签
index1 = pd.Series(['auc', 'aupr', 'accuracy', 'f1'])
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
result_all.to_csv('data/result/results_DGERN.csv')
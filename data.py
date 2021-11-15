#%%
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

# read data
gene_info = pd.read_csv("GSE70138_Broad_LINCS_gene_info_2017-03-06.txt",
                        sep="\t",
                        dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]
# landmark_only_gctoo = parse("GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx", rid = landmark_gene_row_ids)           

label = pd.read_csv('./outcome/drug_name2label_3.csv',index_col='b_id')
sig_info = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt",sep="\t")
MCF7_info = sig_info["sig_id"].loc[(sig_info["cell_id"] == "MCF7")&(sig_info["pert_id"].isin(label.index))]
PC3_info = sig_info["sig_id"].loc[(sig_info["cell_id"] == "PC3")&(sig_info["pert_id"].isin(label.index))]
A375_info = sig_info["sig_id"].loc[(sig_info["cell_id"] == "A375")&(sig_info["pert_id"].isin(label.index))]
train_data_info=pd.concat([MCF7_info,PC3_info,A375_info],axis=0, join='inner')
train_data=parse("GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx", 
                 rid = landmark_gene_row_ids, cid = train_data_info)

# info with label
MCF7_info = sig_info[["sig_id","pert_id"]].loc[(sig_info["cell_id"] == "MCF7")&sig_info["pert_id"].isin(label.index)]
PC3_info = sig_info[["sig_id","pert_id"]].loc[(sig_info["cell_id"] == "PC3")&sig_info["pert_id"].isin(label.index)]
A375_info = sig_info[["sig_id","pert_id"]].loc[(sig_info["cell_id"] == "A375")&sig_info["pert_id"].isin(label.index)]
train_data_info_1 = pd.concat([MCF7_info,PC3_info,A375_info],axis=0, join='inner')
train_data_info_2 = train_data_info_1.copy()
label['0'] = label['0'].astype('category')
label_index = pd.factorize(label['0'])
label['0_1'] = pd.factorize(label['0'])[0]
# label['0_1'].astype('category')
for i in label['0_1'].index:
    train_data_info_2['pert_id'] = train_data_info_2['pert_id'].replace(i,label['0_1'].loc[i])

label_1 = np.asarray(train_data.col_metadata_df.index)
train_data_info_2.index = train_data_info_2['sig_id']
label_init = []
for i in label_1:
    label_init.append(train_data_info_2['pert_id'].loc[i])

label_count=pd.Series(label_init,dtype='category')
label_count.value_counts().sort_index(ascending=True)

# change label to onehotencode
ohe = OneHotEncoder()
ohe.fit([[i] for i in range(0,18)])
label_array = np.asarray(label_init).reshape(-1,1)
label_ohe = ohe.transform(label_array.reshape(-1,1)).toarray()

#%%
# 保存数据
with open('./data/train_data.pkl','wb') as f:  
    pickle.dump(train_data.data_df, f)
with open('./data/label_init.pkl','wb') as f:  
    pickle.dump(label_init, f) 
with open('./data/label_ohe.pkl','wb') as f:  
    pickle.dump(label_ohe, f)

if __name__ == "__main__":
    
    # 将标签变为聚类可以用的分布
    max(label_init)
    vec = np.full(18,0.001)
    vec[label_init[0]] = .983
    vec = vec[np.newaxis, :]

    for i in label_init[1:]:
        vec_a = np.full(18,0.001)
        vec_a[i] = .983
        vec_a = vec_a[np.newaxis, :]
        vec = np.insert(vec, vec.shape[0], values = vec_a, axis = 0)
    

    label['drug_name'].drop_duplicates()



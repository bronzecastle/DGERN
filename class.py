#%%
import sys
sys.path.append(r"/home/cczhao/workspace/medicine/")
import pkg_resources
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
import dsn
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
import pickle
from keras.models import load_model
from keras.engine.topology import Layer ,InputSpec
import keras.backend as K

#%% data
f1 = open('./data/train_data.pkl','rb')
f2 = open('./data/label_init.pkl','rb')
f3 = open('./data/label_ohe.pkl','rb')

train_data = pickle.load(f1)
label_init = pickle.load(f2)
label_ohe = pickle.load(f3)
shape=[i for i in list(train_data.shape)[::-1]]
#%% get dims
dsn.dsn.getdims(shape)
adata=train_data.T
'''dsn.dsn.train_single(adata, dims=[adata.shape[1], 128, 32], tol=0.005, 
          batch_size=256, louvain_resolution=[0.8],n_clusters=3,
          save_dir="test", do_tsne=False, learning_rate=300,
          verbose=False,do_umap=False, save_encoder_weights=True)'''

dsn_run = dsn.network_1.DsnModel(dims=[adata.shape[1], 128, 32], x=adata, tol=0.005, 
          batch_size=256, louvain_resolution=[0.8], n_clusters=18,
          save_dir="test", label_init=label_init, label_ohe=label_ohe)
dsn_run.compile(optimizer=SGD(0.01,0.9), loss='kld')
Embeded_z,q_pred=dsn_run.fit(maxiter=1000)
#%%
if __name__ == '__main__':
    # 获得预测后的标签
    y_pred=pd.Series(np.argmax(q_pred,axis=1),dtype='category')
    y_pred.cat.categories=list(range(len(y_pred.unique())))
    y_pred.value_counts().sort_index(ascending=True)
    # 判断有多少标签变了
    change_label = np.asarray(y_pred)-np.asarray(label_init)
    change_label = pd.DataFrame(change_label)
    change_label[change_label[0] != 0]
    # 保存模型
    dsn_run.model.save('./outcome_fcf/dsn_2.h5')
    dsn_run.model.save_weights('./outcome_fcf/dsn_weights_1.h5')
    # 保存loss和标签
    dsn_run.hist_1.history['loss']
    dsn_run.hist_1.history['acc']
    dsn_run.hist
    dsn_run.hist_3.history['loss']
    dsn_run.hist_3.history['acc']
    dsn_run.hist['clustering'] = dsn_run.hist.pop('loss') 
    dsn_run.hist['class_1_loss'] = dsn_run.hist_1.history['loss']
    dsn_run.hist['class_1_acc'] = dsn_run.hist_1.history['acc']
    dsn_run.hist['class_2_loss'] = dsn_run.hist_3.history['loss']
    dsn_run.hist['class_2_acc'] = dsn_run.hist_3.history['acc']
    label_contain = {}
    label_contain['Embeded_z'] = Embeded_z
    label_contain['Embeded_class_1'] = dsn_run.features 
    label_contain['Embeded_class_2'] = dsn_run.features_1 

    with open('./outcome_fcf/loss_2.pkl','wb') as f:  
        pickle.dump(dsn_run.hist, f)  
    with open('./outcome_fcf/label_2.pkl','wb') as f:  
        pickle.dump(label_contain, f)  
   
    # 判断数据的正态性
    kstest(train_data.data_df['REP.A001_A375_24H:A03'],'norm')

    # 加载模型
    class ClusteringLayer(Layer):
        def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
            if 'input_shape' not in kwargs and 'input_dim' in kwargs:
                kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super(ClusteringLayer, self).__init__(**kwargs)
            self.n_clusters = n_clusters
            self.alpha = alpha
            self.initial_weights = weights
            self.input_spec = InputSpec(ndim=2)
        def build(self, input_shape):
            assert len(input_shape) == 2
            input_dim = input_shape[1]
            self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
            self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights
            self.built = True
        def call(self, inputs, **kwargs):
            q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
            q **= (self.alpha + 1.0) / 2.0
            q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
            return q 
        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            return input_shape[0], self.n_clusters
        def get_config(self):
            config = {'n_clusters': self.n_clusters}
            base_config = super(ClusteringLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))



    model = load_model('./outcome/dsn_1.h5', custom_objects={'ClusteringLayer': ClusteringLayer})
    from keras.utils import CustomObjectScope

    with CustomObjectScope({'ClusteringLayer': ClusteringLayer}):
        model = load_model('./outcome/dsn_1.h5')
    q_pred = model.predict(adata)
    Embeded_z,q_pred=model.fit_certain_dit(maxiter=1)
    y_pred=pd.Series(np.argmax(q_pred,axis=1),dtype='category')
    y_pred.cat.categories=list(range(len(y_pred.unique())))
    y_pred.value_counts().sort_index(ascending=True)
    y_pred_1 = pickle.load(open('./outcome/label_1.pkl','rb'))
    y_pred[y_pred!=y_pred_1]



    # 调试sae
    sae = dsn.SAE.SAE(dims=[978, 64, 32],act='tanh',drop_rate=0.2,batch_size=256,random_seed=201809,actincenter="tanh",init='glorot_uniform',use_earlyStop=True,save_dir="result_tmp")
    sae.fit(adata,label_ohe)

    result = sae.autoencoders.predict(adata)
    result[1,:]
    mse = np.mean(np.power(adata - result, 2), axis=1)
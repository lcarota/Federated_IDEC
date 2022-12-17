import os
#from time import time
import numpy as np
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
#from tensorflow.keras.layers import Dense, Input
import tensorflow.keras as keras
from tensorflow.keras import layers
#import tensorflow as tf
from tensorflow.keras import callbacks
import pandas as pd

from sklearn.cluster import KMeans,SpectralClustering
from sklearn import metrics


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    ind_to_change = linear_sum_assignment(w.max() - w)
    ind = np.vstack([ind_to_change[0],ind_to_change[1]]).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def autoencoder(dims):
    
    latent_dim = dims[-1]
        
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                      mean=0., stddev=0.5,seed=None)
        return z_mean + K.exp(z_log_var) * epsilon 
   
    #encoder
    input_seq = keras.Input(shape=(dims[0],))
    x=layers.Dense(500,activation="relu",name='e_0')(input_seq)
    x=layers.Dense(500,activation="relu",name='e_1')(x)
    x=layers.Dense(2000,activation="relu",name='e_2')(x)
    z_mean=layers.Dense(latent_dim,name='mean')(x)
    z_log_var=layers.Dense(latent_dim,name='std')(x)
    z = layers.Lambda(sampling,output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder=Model(input_seq,[z_mean,z_log_var,z],name='encoder')
    
    #decoder
    decoder_input=layers.Input(shape=(latent_dim,),name='z_sampling')
    x=layers.Dense(2000,activation="relu",name='d_2')(decoder_input)#was elu
    x=layers.Dense(500,activation="relu",name='d_1')(x)
    x=layers.Dense(500,activation="relu",name='d_0')(x)
    output=layers.Dense(dims[0],activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
    decoder=Model(decoder_input,output,name='decoder')
    
    #end-to-end vae
    output_seq = decoder(encoder(input_seq)[2])
    vae = Model(input_seq, output_seq, name='vae')
    
    reconstruction_loss = keras.losses.mean_squared_error(input_seq,output_seq)
    reconstruction_loss *= dims[0]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    #kl_loss *= 5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae

    
    


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

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
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
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




class IDECvar(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 batch_size=256,
                 out='out',
                 setting='centralized'):

        super(IDECvar, self).__init__()

        self.out = out
        self.setting = setting
        
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)

    
    def pretrain(self, x, y=None, 
                 optimizer='adam', 
                 patience=50, 
                 max_epochs=1000):
        
        #if not os.path.exists(self.out+'/pretrain'):
        #    os.makedirs(self.out+'/pretrain')
        
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer)#,metrics=["accuracy"])
        
        if self.setting == 'centralized':
            checkpointer=callbacks.ModelCheckpoint(
                      filepath=self.out+"/ae_weights.h5",
                      verbose=1,
                      save_weights_only=True, #controlla
                      best_only_model=True,
                      monitor="val_loss",
                      save_freq='epoch',
                      #period=1
                      )          
        
        earlystop=callbacks.EarlyStopping(monitor="val_loss",
                                                min_delta=0,
                                                patience=patience)

        reducelr=callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.5,
                                                   patience=int(patience/4),
                                                   verbose=1,
                                                   mode='auto',
                                                   min_delta=0,
                                                   cooldown=0,
                                                   min_lr=0)
        
        if self.setting == 'centralized':
            fit_callbacks = [checkpointer,earlystop,reducelr]
        if self.setting == 'federated':
            fit_callbacks = [earlystop,reducelr]
                       
        #training
        history=self.autoencoder.fit(x=x,
                        y=y,
                        shuffle=True,
                        epochs=max_epochs,
                        callbacks=fit_callbacks,
                        validation_data=(x,None),
                        batch_size=self.batch_size)
        
               
        #save training history
        h=pd.DataFrame(history.history)
        h.to_csv(self.out+"/ae_history.csv",mode='a')
                 
    
    def initialize_model(self, ae_weights=None, 
                         ae_loss_coeff=1, 
                         gamma=0.1, 
                         optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
        else:
            self.autoencoder.load_weights(self.out+"/ae_weights.h5")

        hidden = self.autoencoder.get_layer(name='encoder').output
        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden[0])
        self.model = Model(inputs=self.autoencoder.input,outputs=[clustering_layer, self.autoencoder.output])       
        
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        self.z_mean = self.model.get_layer('encoder').output[0]
        self.z_log_var = self.model.get_layer('encoder').output[1]
       
        def loss_modificata(z_mean,z_log_var):
            def loss_autoencoder(true, pred):
                # Reconstruction loss
                reconstruction_loss = keras.losses.mean_squared_error(true,pred)
                # KL divergence loss
                kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
                kl_loss = K.sum(kl_loss, axis=1)#axis? 1 o -1
                kl_loss *= -0.5
                #vae_loss = K.mean(reconstruction_loss + kl_loss)
                return reconstruction_loss#vae_loss 
            return loss_autoencoder
        
        if optimizer == 'sgd':
            from tensorflow.keras.optimizers import SGD
            optimizer = SGD(lr=0.001, momentum=0.9)
        
        self.model.compile(loss={'clustering':'kld','decoder':loss_modificata(self.z_mean, self.z_log_var)},#vae_loss(z_mean, z_log_var)},#loss_autoencoder},
                           loss_weights=[gamma,ae_loss_coeff],
                           optimizer=optimizer,
                           run_eagerly=True,
                           metrics=["accuracy"]
                           )
        
    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        encoder = Model(self.model.input, self.model.get_layer('encoder').output)
        return encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   cluster_method='kmeans'):
        
        if update_interval == None:
            # aggiorna p e q ogni epoca
            update_interval = int(x.shape[0] / self.batch_size)        
        print('Update interval', update_interval)
        epochs = 20
        save_interval = x.shape[0] / self.batch_size * epochs 
        print('Save interval', save_interval)
        
        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        features = self.encoder.predict(x)[0]
        if cluster_method=='kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features)
            y_pred_last = y_pred
            self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        if cluster_method=='SC':
            SC = SpectralClustering(n_clusters=self.n_clusters, n_init=20)
            y_pred = SC.fit_predict(features)
            y_pred_last = y_pred
            
            def compute_centroid(array):
                n_dim = len(array[0])
                length = array.shape[0]
                centroid = []
                for i in range(n_dim):
                    sum_i = np.sum(array[:, i]) / length
                    centroid = np.append(centroid, sum_i)
                return centroid
            centroids = []
            for i in np.unique(y_pred):
                idx = (y_pred == i)
                centroids.append(list(compute_centroid(features[idx, :])))
            self.model.get_layer(name='clustering').set_weights([np.array(centroids)])
        # logging file
        import csv
        logfile = open(self.out + '/idec_history.csv', 'a+')
        if y is not None:
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'ami', 'fms', 'L', 'Lc', 'Lr'])
            logwriter.writeheader()
        else:
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'L', 'Lc', 'Lr'])
            logwriter.writeheader()
            
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            print('ite: ',ite)
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    ami = np.round(metrics.adjusted_mutual_info_score(y, y_pred), 5)
                    fms = np.round(metrics.fowlkes_mallows_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, ami=ami, fms=fms, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, ', ami', ami, ', fms', fms, '; loss=', loss)
                else: 
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, '; loss=', loss)                    
                
                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break
                
            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=[p[index * self.batch_size::], x[index * self.batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1
      
            ite += 1
        # save the trained model
        logfile.close()
        if self.setting == 'centralized':
            print('saving model to:', self.out + '/idec_weights.h5')
            self.model.save_weights(self.out + '/idec_weights.h5')
        
        return y_pred    

 

### ci sarebbero pure i parametri del clustering layer (t-student distribution)

if __name__ == "__main__": 
    # setting the hyper parameters     
    
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--setting', default='centralized', choices=['centralized','federated'])
    parser.add_argument('--out', default='prova_idec')
    parser.add_argument('--dataset',default='mnist',choices=['mnist','eurodms'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=5, type=int)#64
    # pretraining parameters
    parser.add_argument('--ae_optimizer', default='adam')
    parser.add_argument('--patience', default=50)
    parser.add_argument('--max_ae_epochs', default=1)#000)
    # idec parameters
    parser.add_argument('--ae_weights', default=None)#'ae_weights.hdf5')#None)#'out/mnist/pretraining/mnist0/ae_weights.hdf5', help='This argument must be given')
    parser.add_argument('--idec_optimizer', default='adam')
    parser.add_argument('--ae_loss_coeff', default=1, type=float, help='coefficient of reconstruction loss')
    parser.add_argument('--gamma', default=1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--cluster_method',default='kmeans',choices=['kmeans','SC'])
    parser.add_argument('--idec_epochs',default=1,type=int)#50
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--maxiter', default=12e4, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    print(args)

    out = args.out
    setting = args.setting
    dataset = args.dataset
    n_clusters = args.n_clusters
    batch_size = args.batch_size
    
    ae_optimizer = args.ae_optimizer
    patience = args.patience
    max_ae_epochs = args.max_ae_epochs
    
    ae_weights = args.ae_weights
    idec_optimizer = args.idec_optimizer
    ae_loss_coeff = float(args.ae_loss_coeff)
    gamma = float(args.gamma)
    cluster_method = args.cluster_method
    idec_epochs = args.idec_epochs
    update_interval = args.update_interval
    maxiter = args.maxiter
    tol = args.tol
        
    if not os.path.exists(out):
        os.makedirs(out)
    
    import json
    with open(out+'/config.json', 'w') as file:
        json.dump(vars(args), file)
      
    if dataset=='euromds':
        x = json.load(open('data/euromds/euromds.json','r'))
        x = np.array(x)
        #if exclude_data_duplicates == True:
            #exclude duplicate rows:
            #x = np.unique(x,axis=0)
        y=None

    if dataset=='mnist':
        import dataset
        binary_threshold=0.5
        num_clients=8
        samples_per_cluster_per_client=100
        delete_dim=True
        x_tot,y_tot = dataset.load_mnist(binary_threshold)
        x,y,x_fed,y_fed = dataset.create_federated_dataset(x_tot,y_tot,
                                       num_clients=num_clients,
                                       samples_per_cluster_per_client=samples_per_cluster_per_client,
                                       delete_dim=delete_dim)
    input_dim = x.shape[1]

    # prepare the IDEC model
    model = IDECvar(dims=[input_dim, 500, 500, 2000, 10], 
                n_clusters=n_clusters, 
                batch_size=batch_size,
                out=out,
                setting=setting)
    
    # pretraining phase
    model.pretrain(x, 
                   y=y, 
                   optimizer=ae_optimizer, 
                   patience=patience, 
                   max_epochs=max_ae_epochs)
    

    # clustering phase
    model.initialize_model(ae_weights=ae_weights, 
                          ae_loss_coeff =ae_loss_coeff,
                          gamma=gamma, 
                          optimizer=idec_optimizer)
    model.model.summary()
    y_pred = model.clustering(x,y, 
                             epochs=idec_epochs, 
                             tol=tol, 
                             maxiter=maxiter,
                             update_interval=update_interval, 
                             cluster_method=cluster_method)
      
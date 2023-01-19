"""
Implementation for Improved Deep Embedded Clustering as described in paper:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.

Usage:
    Weights of Pretrained autoencoder for mnist are in './ae_weights/mnist_ae_weights.h5':
        python IDEC.py mnist --ae_weights ./ae_weights/mnist_ae_weights.h5
    for USPS and REUTERSIDF10K datasets
        python IDEC.py usps --update_interval 30 --ae_weights ./ae_weights/usps_ae_weights.h5
        python IDEC.py reutersidf10k --n_clusters 4 --update_interval 3 --ae_weights ./ae_weights/reutersidf10k_ae_weights.h5

Author:
    Xifeng Guo. 2017.4.30
    
THE ORIGINAL CODE HAS BEEN SLIGHTLY MODIFIED IN ORDER TO BE IMPORTED IN 
THE SCRIPT IDEC_federated.py
"""

from time import time
import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer,InputSpec,Dense,Input
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans
from sklearn import metrics
import json


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
   # y_new = y_true.copy()
    #for i in np.arange(0, len(y_true), 1):
     #   y_new[i] = np.argmax(y_true[i])
    #y_true = y_new[:,0]
    y_true = y_true.astype(np.int64)
    print('y_pred.size: ', y_pred.size)#
    print('y_true.size: ', y_true.size)#
    print('y_pred: ', y_pred)#
    print('y_true: ', y_true)#
    print('y_pred.shape: ', y_pred.shape)#
    print('y_true.shape: ', y_true.shape)#
    #assert y_pred.size == y_true.size
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


def autoencoder(dims, act='relu', init='glorot_uniform'): 
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE')#, Model(inputs=x, outputs=h, name='encoder')


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


class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=6,#10,
                 alpha=1.0,
                 batch_size=256,
                 out='out',
                 setting='centralized'):

        super(IDEC, self).__init__()
        
        self.out = out
        self.setting = setting

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims, init='glorot_uniform')
        #self.autoencoder, self.encoder = autoencoder(self.dims, init='glorot_uniform')
        

    def pretrain(self, x, y=None, 
                 optimizer='sgd', 
                 epochs=500):
        
        #if not os.path.exists(self.out+'/pretrain'):
        #    os.makedirs(self.out+'/pretrain')
        
        print('...Pretraining...')
        if optimizer == 'sgd':
            from tensorflow.keras.optimizers import SGD
            optimizer = SGD(learning_rate=1, momentum=0.9)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        
        csv_logger = callbacks.CSVLogger(self.out + '/ae_history.csv', append=True)
        
        cb = [csv_logger]

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=self.batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        if self.setting == 'centralized':
            self.autoencoder.save_weights(self.out + '/ae_weights.h5')
            print('Pretrained weights are saved to %s/ae_weights.h5' % self.out)
        self.pretrained = True


    def model_initialization(self, ae_weights=None, 
                         ae_loss_coeff=1, 
                         gamma=1, 
                         optimizer='adam'):
           
        # prepare DEC model
        
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)                
        else:
            self.autoencoder.load_weights(self.out+"/ae_weights.h5")
            
        if optimizer == 'sgd':
            from tensorflow.keras.optimizers import SGD
            optimizer = SGD(learning_rate=0.001, momentum=0.9)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')            
        
           
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('encoder_%d' % (self.n_stacks - 1)).output, name='encoder')
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)

        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])
        self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                           loss_weights=[gamma, ae_loss_coeff],
                           optimizer=optimizer,
                           metrics=["accuracy"])
        #print(self.encoder.summary())

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_features(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
    def centroid_initialization(self,x):
        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20,random_state=0)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        cluster_centers = kmeans.cluster_centers_
        self.model.get_layer(name='clustering').set_weights([cluster_centers])
        
        #if self.setting == 'centralized':
        with open(self.out+'/init_centroids.json', 'w') as f:
            f.write(json.dumps(cluster_centers.tolist()))
        with open(self.out+'/init_labels.json', 'w') as f:
            f.write(json.dumps(y_pred.tolist()))
        
        return cluster_centers,y_pred

       
    def clustering(self, x, y=None,
                   tol=1e-3,
                   maxiter=2e4,
                   update_interval=140):
        
        if update_interval == None:
            # aggiorna p e q ogni epoca
            update_interval = int(x.shape[0] / self.batch_size)
        print('Update interval', update_interval)
        # salva weights ogni 20 epoche
        epochs = 20 
        save_interval = x.shape[0] / self.batch_size * epochs
        print('Save interval', save_interval)
        
        #y_pred_last = self.y_pred
        # logging file
        import csv
        #if not os.path.exists(self.out+'/idec'):
        #    os.makedirs(self.out+'/idec')
        logfile = open(self.out + '/idec_history.csv', 'a+')
        if y is not None:
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'ami', 'fms', 'L', 'Lc', 'Lr'])
            logwriter.writeheader()
        else:
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'L', 'Lc', 'Lr'])
            logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        count = 0
        
        for ite in range(int(maxiter)):
            print('ite: ',ite)
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if ite > 0:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    ami = np.round(metrics.adjusted_mutual_info_score(y, y_pred), 5)
                    fms = np.round(metrics.fowlkes_mallows_score(y, y_pred), 5)
                    silh = np.round(metrics.silhouette_score(x, y_pred), 5) #silhouette score
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, ami=ami, fms=fms, silh=silh, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, ', ami', ami, ', fms', fms, ', silh', silh, '; loss=', loss)
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
            #print('loss: ',loss)
            '''
            if self.setting == 'centralized':
                # save intermediate model
                if ite % save_interval == 0: 
                    # save IDEC model checkpoints
                    print('saving model to:', self.out + '/idec/IDEC_model_' + str(count*epochs) + '.h5')
                    self.model.save_weights(self.out + '/idec/IDEC_model_' + str(count*epochs) + '.h5')
                    count += 1
            '''
            ite += 1
        # save the trained model
        logfile.close()
        if self.setting == 'centralized':
            print('saving model to:', self.out + '/idec_weights.h5')
            self.model.save_weights(self.out + '/idec_weights.h5')
        
        return y_pred

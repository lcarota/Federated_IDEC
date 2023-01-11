import os
#os.chdir('C:\\Users\\silvi\\Desktop\\Fisica\\TESI\\AAAAA_clustering_methods\\federated_clustering')
import IDEC
from dataset import load_mnist,load_euromds,create_federated_dataset,missing_data_distribution,from_fed_to_centr
from typing import List
import csv
import tensorflow.keras as keras
from sklearn.cluster import KMeans

import flwr as fl
import numpy as np 



# Make TensorFlow log less verbose 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disabling warning for ray dashboard missing dependencies
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
import argparse

parser = argparse.ArgumentParser(description='federated_idec',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--setting', default='centralized', choices=['centralized','federated'])
parser.add_argument('--out', default='results_sm')
parser.add_argument('--dataset',default='mnist',choices=['mnist','euromds'])
parser.add_argument('--n_clusters', default=10, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--samples_per_cluster_per_client', default=300, type=int)
parser.add_argument('--nan', default=False) #non introduce dati mancanti
parser.add_argument('--nan_fraction', default=0.375,type=float)
parser.add_argument('--nan_dim', default=0, type=int) #anche in una delle 784
parser.add_argument('--client_with_missing_data',default=8, help='if list, it is the list of clients with missing data; if int, it is the number of clients with missing data (randomly chosen)')
parser.add_argument('--nan_substitute',default='mean')#fa la media dove ci sono dati mancanti
parser.add_argument('--seed',default=0)
parser.add_argument('--client',default=None,help='only in centralized setting; if not None, it identifies in which client the centralized method has to be applied')

# server
parser.add_argument('--phase',default='complete',choices=['complete','ae','idec'])
parser.add_argument('--num_rounds_ae',default=20,type=int)
parser.add_argument('--num_rounds_idec',default=10,type=int)
parser.add_argument('--num_clients',default=1,type=int)
parser.add_argument('--strategy',default='fedavg',choices=['fedavg','fedadam','fedyogi'])#per gli ultimi due modificare il training specifico, anche le stesse funzioni
# pretraining parameters
parser.add_argument('--ae_optimizer', default='sgd')
parser.add_argument('--ae_epochs', default=150,type=int)
# idec parameters
parser.add_argument('--ae_weights', default='ae_weights_first.h5')#per la fase di clustering servono per forza i pesi iniziali
#'out/idec/federated/no_nan/750samples/server/ae_round_25_weights.h5')
parser.add_argument('--idec_optimizer', default='sgd')
parser.add_argument('--ae_loss_coeff', default=1, type=float, help='coefficient of reconstruction loss')
parser.add_argument('--gamma', default=1, type=float, help='coefficient of clustering loss')
parser.add_argument('--cluster_method',default='kmeans',choices=['kmeans','SC'])
parser.add_argument('--update_interval', default=None, type=int)
parser.add_argument('--maxiter', default=12e4, type=int)
parser.add_argument('--tol', default=0.000001, type=float)


args = parser.parse_args()
print(args)

setting = args.setting
out = args.out
dataset = args.dataset
n_clusters = args.n_clusters
batch_size = args.batch_size
samples_per_cluster_per_client = args.samples_per_cluster_per_client
nan = args.nan
nan_fraction = args.nan_fraction
nan_dim = args.nan_dim
client_with_missing_data = args.client_with_missing_data
seed = args.seed
client = args.client
if args.nan_substitute == 'mean':
    nan_substitute = args.nan_substitute
else:
    nan_substitute = float(args.nan_substitute)

phase = args.phase
num_rounds_ae = args.num_rounds_ae
num_rounds_idec = args.num_rounds_idec
num_clients = args.num_clients
    
ae_optimizer = args.ae_optimizer
ae_epochs = args.ae_epochs
 
ae_weights = args.ae_weights   
idec_optimizer = args.idec_optimizer
ae_loss_coeff = float(args.ae_loss_coeff)
gamma = float(args.gamma)
cluster_method = args.cluster_method
update_interval = args.update_interval
maxiter = args.maxiter
tol = args.tol

if args.strategy == 'fedavg':
    strategy = fl.server.strategy.FedAvg
if args.strategy == 'fedadam':
    strategy = fl.server.strategy.FedAdam
if args.strategy == 'fedyogi':
    strategy = fl.server.strategy.FedYogi


if idec_optimizer == 'adam':
    idec_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

#cid = args.cid

import os
if not os.path.exists(out):
    os.makedirs(out)
    
import json
with open(out+'/config.json', 'w') as file:
    json.dump(vars(args), file)
      
    
# load dataset
# load binary mnist
#binary_threshold=0.5
binary_threshold=None

#x,y = load_mnist(binary_threshold)
x,y = load_euromds()

# modifications for federated setting
x_fed,y_fed = create_federated_dataset(x,y,
                                       num_clients=num_clients,
                                       samples_per_cluster_per_client=samples_per_cluster_per_client)
input_dim = len(x_fed[0][0])


if setting == 'centralized':
    
    x_centr,y_centr = from_fed_to_centr(x_fed,y_fed)
    print('number of samples - centralized setting: ',len(x_centr))
    
    # tanto si usa solo in assenza di nan (sennò devi modificare)
    if client is not None:
        x_centr = x_fed[int(client)].copy()
        y_centr = y_fed[int(client)].copy()

    if nan == True:
        print('add missing data')
        x_centr_nan = missing_data_distribution(x_centr,
                                  nan_fraction=nan_fraction,
                                  nan_dims=[nan_dim], 
                                  client_with_missing_data=client_with_missing_data,
                                  nan_substitute = nan_substitute,
                                  seed=seed,
                                  setting=setting)
        x_centr = x_centr_nan.copy()
    #x_centr = x_fed[0].copy()
    #y_centr = y_fed[0].copy()
    
    

    model = IDEC.IDEC(dims=[input_dim, 500, 500, 2000, 10],
                 n_clusters=n_clusters,
                 alpha=1.0,
                 batch_size=batch_size,
                 out=out,
                 setting=setting)
    
    if phase == 'complete' or phase == 'ae':
        # pretraining phase   
        model.pretrain(x_centr, y_centr, 
                     optimizer=ae_optimizer, 
                     epochs=ae_epochs,)
    
    if phase == 'complete' or phase == 'idec':
        # clustering phase
        model.model_initialization(ae_weights=ae_weights, 
                              ae_loss_coeff =ae_loss_coeff,
                              gamma=gamma, 
                              optimizer=idec_optimizer)
        model.centroid_initialization(x_centr)
    
        y_pred = model.clustering(x_centr,y_centr, 
                                 tol=tol, 
                                 maxiter=maxiter,
                                 update_interval=update_interval)
        

if setting == 'federated':

    if nan == True:
        print('add missing data')
        x_fed_nan = missing_data_distribution(x_fed,
                                  nan_fraction=nan_fraction,
                                  nan_dims=[nan_dim], 
                                  client_with_missing_data=client_with_missing_data,
                                  nan_substitute = nan_substitute,
                                  seed=seed,
                                  setting=setting)
        x_fed = x_fed_nan.copy()
   
        
    if not os.path.exists(out+'/server'):
        os.makedirs(out+'/server')
        
    if phase=='ae' or phase=='complete':
        file_ae = open(out + '/server/ae_fit_history.csv', 'a+')
        writer_ae = csv.DictWriter(file_ae, fieldnames=['round', 'results', 'failures'])
        writer_ae.writeheader()

        
    ########### PRETRAINING
    
    # Define Flower client for pretraining phase
    class PretrainingClient(fl.client.NumPyClient):
        
        def __init__(self,model,x_train,y_train,x_test,y_test):
            self.model = model
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
        
        def get_parameters(self, config):
            return self.model.autoencoder.get_weights()
    
        def fit(self, parameters, config):
            self.model.autoencoder.set_weights(parameters)
            self.model.pretrain(self.x_train, 
                       y=self.y_train, 
                       optimizer=ae_optimizer, 
                       epochs=ae_epochs)
            return self.model.autoencoder.get_weights(), len(self.x_train), {}
    
        def evaluate(self, parameters, config):
            self.model.autoencoder.set_weights(parameters)
            loss,acc = self.model.autoencoder.evaluate(self.x_test, self.y_test)
            return loss, len(self.x_test), {"accuracy": acc}
    
    def client_fn_ae(cid):
        
        if not os.path.exists(out+'/client'+str(cid)):
            os.makedirs(out+'/client'+str(cid))        
            
        x_train = x_fed[int(cid)].copy()
        y_train = y_fed[int(cid)].copy()
        x_test = x_train.copy()
        y_test = y_train.copy()
                   
        # pretraining
        model = IDEC.IDEC(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters, 
                        alpha=1.0,
                        batch_size=batch_size,
                        out=out+'/client'+str(cid),
                        setting='federated',
                        )
        model.autoencoder.compile(metrics=["accuracy"])
            
        return PretrainingClient(model,x_train,y_train,x_test,y_test)
    
    
    class Strategy_ae(strategy):
        def aggregate_fit(
            self,
            rnd: int,
            results,
            failures,):

            weights = super().aggregate_fit(rnd, results, failures)
            weights_red,_ = weights 
                    
            dict_ae = dict(round=rnd, results=len(results), failures=len(failures))
            writer_ae.writerow(dict_ae)
            if weights is not None:
                # Save weights
                print(f"Saving round {rnd} weights...")
                aggregated_weights : List[np.ndarray] = fl.common.parameters_to_ndarrays(weights_red)
                
                model_for_saving_weights = IDEC.IDEC(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters, 
                        alpha=1.0,
                        batch_size=batch_size,
                        out=None,
                        setting='federated')
                model_for_saving_weights.autoencoder.set_weights(aggregated_weights)
                model_for_saving_weights.autoencoder.save_weights(out+f'/server/ae_weights.h5')#ae_round_{rnd}_weights.h5')
                if rnd == num_rounds_ae:
                    model_for_saving_weights.autoencoder.save_weights(out+f'/server/ae_round_{rnd}_weights.h5')
            '''
            if rnd == 1:
                for i in range(len(results)):
                    #if not os.path.exists(out+'/client'+str(i)):
                    #    os.makedirs(out+'/client'+str(i)) 
                    _,w = results[i]
                    weights_to_array = fl.common.parameters_to_ndarrays(w.parameters)
                    model_for_saving_weights.autoencoder.set_weights(weights_to_array)
                    model_for_saving_weights.autoencoder.save_weights(out+'/client'+str(i)+f'/ae_round_{rnd}_weights.h5')
             '''              
            return weights
          
    # pretraining
    strategy_ae = Strategy_ae(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        #initial_parameters=model.autoencoder.get_weights()
        )
    
    
    if phase == 'complete' or phase == 'ae':
         
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn_ae,
            num_clients=num_clients,
            client_resources={"num_cpus": num_clients},
            config=fl.server.ServerConfig(num_rounds=num_rounds_ae),  
            strategy=strategy_ae,
        )
        
        file_ae.close()
    
    
   
    ############# CLUSTERING

  
    if ae_weights is None:
        ae_weights = out+f'/server/ae_weights.h5'
    
    

    class KMeansClient(fl.client.NumPyClient):
        
        def __init__(self,model,x,y):
            self.model = model
            self.x = x
            self.y = y
            self.cluster_centers = []
            
        def get_parameters(self,config):
            return self.cluster_centers
        
        def fit(self, parameters, config):
            print('...starting fit...')
            #self.model.model.set_weights(parameters)
            self.cluster_centers,self.y_pred = self.model.centroid_initialization(self.x)
            print('cluster_centers: ',self.cluster_centers)#
            #print('y_pred: ',self.y_pred)#
            couples = []
            for i in np.unique(self.y_pred):
                idx = (self.y_pred == i)
                couples.append(self.cluster_centers[i])
                couples.append(np.sum(idx))
            #print('couples: ',couples)
            return couples, len(self.x), {}
        
        def evaluate(self, parameters, config):
            print('...starting evaluate...')
            #self.model.model.set_weights(parameters)
            self.cluster_cen, self.y_pred = self.model.centroid_initialization(self.x)
            print('cluster_cen: ', self.cluster_cen)
            #print('y_pred: ', self.y_pred)
            acc = IDEC.cluster_acc(self.y,self.y_pred)
            return acc, len(self.x), {}
                            
            
    def client_fn_kmeans(cid): 
    
        if not os.path.exists(out+'/client'+str(cid)):
            os.makedirs(out+'/client'+str(cid))
            
        # KMeans application on latent space
        model = IDEC.IDEC(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters,
                        alpha=1.0,
                        batch_size=batch_size,
                        out=out+'/client'+str(cid),
                        setting='federated',)
        model.model_initialization(ae_weights=ae_weights, 
                                  ae_loss_coeff =ae_loss_coeff,
                                  gamma=gamma, 
                                  optimizer=idec_optimizer)
        #model.model.compile(metrics=["accuracy"])
            
        x = x_fed[int(cid)].copy()
        y = y_fed[int(cid)].copy()
            
        return KMeansClient(model,x,y)
    
    def distance_from_centroids(centroids_array, vector):
        # distances array
        distances = []
        # loop on the array/list of centroids
        for centroid in centroids_array:
            # distance between the current centroid and the given vector
            d = np.linalg.norm(centroid-vector)
            # append the distance
            distances = np.append(distances, d)
        # returning the lower distance between all the centroids
        return min(distances)
    
    class Strategy_KMeans(fl.server.strategy.FedAvg):
        '''
        def __init__(self,
                     method='max_min',
                     seed=0):
            self.method = method
            self.seed = seed
        '''
        
        def aggregate_fit(
            self,
            rnd: int,
            results,
            failures,):
            method='max_min'
            seed=0
            rng = np.random.default_rng(seed)
            '''
            centr_list = []
            n_samples_list = []
            for i in range(num_clients): 
                _,fit_res = results[i]
                fit_res_array = fl.common.parameters_to_ndarrays(fit_res.parameters)
                centr = []
                n_samples = []
                for j in range(n_clusters):
                    centr.append(fit_res_array[2*j])
                    n_samples.append(int(fit_res_array[2*j+1]))
                centr_list.append(centr)
                n_samples_list.append(n_samples)
            print('centr_list: ',centr_list)
            print('n_samples_list: ',n_samples_list)
            '''
            all_centroids = []
            all_centroids_multi = []
            n_samples = []
            for _, fit_res in results:
                f_r = fl.common.parameters_to_ndarrays(fit_res.parameters)
                for i in range(n_clusters):
                    all_centroids.append(f_r[2*i])
                    for _ in range(f_r[int(2*i+1)]):
                        all_centroids_multi.append(f_r[2*i])
                    n_samples.append(f_r[int(2*i+1)])
            all_centroids = np.array(all_centroids)
            all_centroids_multi = np.array(all_centroids_multi)
            n_samples = np.array(n_samples)
            #print('All centroids\' multi shape: {}'.format(all_centroids_multi.shape))
            #print('All centroids\' shape: {}'.format(all_centroids.shape))
            #print('N samples shape: {}'.format(n_samples.shape))
            #pd.DataFrame(all_centroids_multi).to_csv(out/'centroids_multi.csv')
            
            if method == 'double_kmeans':
                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                predicted = kmeans.fit_predict(all_centroids_multi)
                base_centroids = kmeans.cluster_centers_
            
            if method == 'max_min':
                # pick, randomly, one client's first centroids
                idx = rng.integers(0, all_centroids.shape[0], 1)
                # basis to be completed
                base_centroids = np.array(all_centroids[idx])
                print('Basis centroids\' starting shape: {}'.format(base_centroids.shape))
                # basis initial length
                basis_length = 1
                # loop for completing the basis
                while basis_length < n_clusters:
                    # all distances from the basis of centroids
                    distances = [distance_from_centroids(
                        base_centroids, c) for c in all_centroids]
                    # get the index of the maximum distance
                    idx = np.argmax(distances)
                    # add the new centroid --> (n_centroids, n_dimensions)
                    base_centroids = np.concatenate(
                        (base_centroids, [all_centroids[idx]]), axis=0)
                    basis_length = base_centroids.shape[0]
                    
            
            if method == 'random':            
                ## weight by n samples the centroids set!!!
                rng.shuffle(all_centroids)
                base_centroids = all_centroids[:n_clusters]
            
            if method == 'random_weighted':            
                ## weight by n samples the centroids set!!!
                rng.shuffle(all_centroids_multi)
                base_centroids = all_centroids_multi[:n_clusters]

            # Save base_centroids
            print(f"Saving base centroids...")
            with open(out+'/server/aggr_centroids.json', 'w') as f:
                f.write(json.dumps(base_centroids.tolist()))
            return fl.common.ndarrays_to_parameters(base_centroids), {}
    
    # clustering
    strategy_kmeans = Strategy_KMeans(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients
        )
    
    
    if phase == 'complete' or phase == 'idec':
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn_kmeans,
            num_clients=num_clients,
            client_resources={"num_cpus": num_clients},
            config=fl.server.ServerConfig(num_rounds=1),  
            strategy=strategy_kmeans,
            )
 
    
    
    
    if phase=='idec' or phase=='complete':
        file_idec = open(out + '/server/idec_fit_history.csv', 'a+')
        writer_idec = csv.DictWriter(file_idec, fieldnames=['round', 'results', 'failures'])
        writer_idec.writeheader()
  
    # Define Flower client for clustering phase
    class ClusteringClient(fl.client.NumPyClient):
        def __init__(self,model,x_train,y_train,x_test,y_test):
            self.model = model
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            
        def get_parameters(self, config):
            return self.model.model.get_weights()
    
        def fit(self, parameters, config):
            print('x_train: ', self.x_train)#
            print('y_train: ', self.y_train)#
            self.model.model.set_weights(parameters)
            self.model.clustering(x=self.x_train, y=self.y_train, 
                                     tol=tol, 
                                     maxiter=maxiter,
                                     update_interval=update_interval)
            return self.model.model.get_weights(), len(self.x_train), {}
    
        def evaluate(self, parameters, config):
            self.model.model.set_weights(parameters)
            loss,cl_loss,dec_loss,cl_acc,dec_acc = self.model.model.evaluate(self.x_test, self.y_test)
            return loss, len(self.x_test), {"clustering_accuracy": cl_acc,"decoder_accuracy":dec_acc}
    
    
    def client_fn_idec(cid): 
    
        if not os.path.exists(out+'/client'+str(cid)):
            os.makedirs(out+'/client'+str(cid))
            
        # clustering
        model = IDEC.IDEC(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters,
                        alpha=1.0,
                        batch_size=batch_size,
                        out=out+'/client'+str(cid),
                        setting='federated',)
        model.model_initialization(ae_weights=ae_weights, 
                                  ae_loss_coeff =ae_loss_coeff,
                                  gamma=gamma, 
                                  optimizer=idec_optimizer)
        cluster_centers = json.load(open(out+'/server/aggr_centroids.json', 'r'))
        cluster_centers = np.array(cluster_centers)
        model.model.get_layer(name='clustering').set_weights([cluster_centers])
        #model.model.compile(metrics=["accuracy"])
            
        x_train = x_fed[int(cid)].copy()
        y_train = y_fed[int(cid)].copy()
        x_test = x_train.copy()
        y_test = y_train.copy()
            
        return ClusteringClient(model,x_train,y_train,x_test,y_test)
    
    
    
    class Strategy_idec(strategy):
        def aggregate_fit(
            self,
            rnd: int,
            results,
            failures,):

            weights = super().aggregate_fit(rnd, results, failures)
            weights_red,_ = weights
            
            dict_idec = dict(round=rnd, results=len(results), failures=len(failures))
            writer_idec.writerow(dict_idec)            
            if weights is not None:
                # Save weights
                print(f"Saving round {rnd} weights...")
                aggregated_weights : List[np.ndarray] = fl.common.parameters_to_ndarrays(weights_red)
                
                model_for_saving_weights = IDEC.IDEC(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters,
                        alpha=1.0,
                        batch_size=batch_size,
                        out=None,
                        setting='federated')
                model_for_saving_weights.model_initialization(ae_weights=ae_weights, 
                                  ae_loss_coeff =ae_loss_coeff,
                                  gamma=gamma, 
                                  optimizer=idec_optimizer)
                model_for_saving_weights.model.set_weights(aggregated_weights)
                model_for_saving_weights.model.save_weights(out+f'/server/idec_weights.h5')
                #if rnd == num_rounds_idec:
                model_for_saving_weights.model.save_weights(out+f'/server/idec_round_{rnd}_weights.h5')
            #print('len(results):',len(results))
            #print('len(results[0]): ',len(results[0]))
            '''
            if rnd == 1:
                for i in range(len(results)):
                    _,w = results[i]
                    weights_to_array = fl.common.parameters_to_ndarrays(w.parameters)
                    model_for_saving_weights.model.set_weights(weights_to_array)
                    model_for_saving_weights.model.save_weights(out+'/client'+str(i)+f'/idec/idec_round_{rnd}_weights.h5')
            '''    
            return weights
            
    # clustering
    strategy_idec = Strategy_idec(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients)
    
    
    if phase == 'complete' or phase == 'idec':
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn_idec,
            num_clients=num_clients,
            client_resources={"num_cpus": num_clients},
            config=fl.server.ServerConfig(num_rounds=num_rounds_idec),  
            strategy=strategy_idec,
        )
        file_idec.close()

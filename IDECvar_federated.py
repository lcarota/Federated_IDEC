import os
#os.chdir('C:\\Users\\silvi\\Desktop\\Fisica\\TESI\\AAAAA_clustering_methods\\federated_clustering')
import IDECvar 
from dataset import load_mnist,create_federated_dataset,missing_data_distribution,from_fed_to_centr
from typing import List
import csv 

import flwr as fl
import numpy as np



# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disabling warning for ray dashboard missing dependencies
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 
import argparse

parser = argparse.ArgumentParser(description='fed_idec_var',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--setting', default='centralized',choices=['centralized','federated'])
parser.add_argument('--out', default='out/prova')
parser.add_argument('--dataset',default='mnist',choices=['mnist','eurodms'])
parser.add_argument('--n_clusters', default=10, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--samples_per_cluster_per_client',default=300,type=int)#100
parser.add_argument('--nan', default=False)
parser.add_argument('--nan_fraction', default=0.375,type=float)
parser.add_argument('--nan_dim', default=0, type=int) 
parser.add_argument('--client_with_missing_data',default=8, help='if list, it is the list of clients with missing data; if int, it is the number of clients with missing data (randomly chosen)')
parser.add_argument('--seed', default=0)

# server
parser.add_argument('--phase',default='complete',choices=['complete','ae','idec'])
parser.add_argument('--num_rounds_ae',default=25,type=int)#25
parser.add_argument('--num_rounds_idec',default=25,type=int)
parser.add_argument('--num_clients',default=8,type=int)
# pretraining parameters
parser.add_argument('--ae_optimizer', default='sgd')
parser.add_argument('--patience', default=50)
parser.add_argument('--max_ae_epochs', default=800,type=int)#1000
# idec parameters
parser.add_argument('--ae_weights', default=None)
parser.add_argument('--idec_optimizer', default='sgd')
parser.add_argument('--ae_loss_coeff', default=1, type=float, help='coefficient of reconstruction loss')
parser.add_argument('--gamma', default=1, type=float, help='coefficient of clustering loss')
parser.add_argument('--cluster_method',default='kmeans',choices=['kmeans','SC'])
parser.add_argument('--update_interval', default=None, type=int)
parser.add_argument('--maxiter', default=12e4, type=int)
parser.add_argument('--tol', default=0.0001, type=float)
 

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


phase = args.phase
num_rounds_ae = args.num_rounds_ae
num_rounds_idec = args.num_rounds_idec
num_clients = args.num_clients
    
ae_optimizer = args.ae_optimizer
patience = args.patience
max_ae_epochs = args.max_ae_epochs
 
ae_weights = args.ae_weights   
idec_optimizer = args.idec_optimizer
ae_loss_coeff = float(args.ae_loss_coeff)
gamma = float(args.gamma)
cluster_method = args.cluster_method
update_interval = args.update_interval
maxiter = args.maxiter
tol = args.tol


import os
if not os.path.exists(out):
    os.makedirs(out)
    
import json
with open(out+'/config.json', 'w') as file:
    json.dump(vars(args), file)
      
    
# load dataset
# load binary mnist
binary_threshold=0.5
x,y = load_mnist(binary_threshold)

# modifications for federated setting
x_fed,y_fed = create_federated_dataset(x,y,
                                       num_clients=num_clients,
                                       samples_per_cluster_per_client=samples_per_cluster_per_client)
input_dim = len(x_fed[0][0])


if setting == 'centralized':
    
    x_centr,y_centr = from_fed_to_centr(x_fed,y_fed)
    print('number of samples - centralized setting: ',len(x_centr))

    if nan == True:
        print('introduction of missing data')
        x_centr_nan = missing_data_distribution(x_centr,
                                  nan_fraction=nan_fraction,
                                  nan_dims=[nan_dim], 
                                  client_with_missing_data=client_with_missing_data,
                                  nan_substitute = 'mean',
                                  seed=seed,
                                  setting=setting)
        x_centr = x_centr_nan.copy()
        
    model = IDECvar.IDECvar(dims=[input_dim, 500, 500, 2000, 10],
                 n_clusters=n_clusters,
                 batch_size=batch_size,
                 out=out,
                 setting=setting)
   
    if phase == 'complete' or phase == 'ae':
        # pretraining phase   
        model.pretrain(x_centr, y_centr,
                       optimizer=ae_optimizer, 
                       patience=patience, 
                       max_epochs=max_ae_epochs)
    
    if phase == 'complete' or phase == 'idec':
        # clustering phase
        model.initialize_model(ae_weights=ae_weights, 
                              ae_loss_coeff =ae_loss_coeff,
                              gamma=gamma, 
                              optimizer=idec_optimizer)
        y_pred = model.clustering(x_centr,y_centr, 
                                 tol=tol, 
                                 update_interval=update_interval,
                                 maxiter=maxiter,
                                 cluster_method=cluster_method)

 

if setting == 'federated':

    if nan == True:
        print('introduction of missing data')
        x_fed_nan = missing_data_distribution(x_fed,
                                  nan_fraction=nan_fraction,
                                  nan_dims=[nan_dim], 
                                  client_with_missing_data=client_with_missing_data,
                                  nan_substitute = 'mean',
                                  seed=seed,
                                  setting=setting)
        x_fed = x_fed_nan.copy()
    
        
    if not os.path.exists(out+'/server'):
        os.makedirs(out+'/server')
        
    if phase=='ae' or phase=='complete':
        file_ae = open(out + '/server/ae_fit_history.csv', 'a+')
        writer_ae = csv.DictWriter(file_ae, fieldnames=['round', 'results', 'failures'])
        writer_ae.writeheader()

    if phase=='idec' or phase=='complete':
        file_idec = open(out + '/server/idec_fit_history.csv', 'a+')
        writer_idec = csv.DictWriter(file_idec, fieldnames=['round', 'results', 'failures'])
        writer_idec.writeheader()
        
        
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
                       patience=patience, 
                       max_epochs=max_ae_epochs)
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
        model = IDECvar.IDECvar(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters, 
                        batch_size=batch_size,
                        out=out+'/client'+str(cid),
                        setting='federated',
                        )
        model.autoencoder.compile(metrics=["accuracy"])
            
        return PretrainingClient(model,x_train,y_train,x_test,y_test)
    



    class Strategy_ae(fl.server.strategy.FedAvg):
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
                
                model_for_saving_weights = IDECvar.IDECvar(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters, 
                        batch_size=batch_size,
                        out=None,
                        setting='federated')
                model_for_saving_weights.autoencoder.set_weights(aggregated_weights)
                model_for_saving_weights.autoencoder.save_weights(out+f'/server/ae_weights.h5')
                if rnd == num_rounds_ae:
                    model_for_saving_weights.autoencoder.save_weights(out+f'/server/ae_round_{rnd}_weights.h5')
    
            if rnd == 1:
                for i in range(len(results)):
                    _,w = results[i]
                    weights_to_array = fl.common.parameters_to_ndarrays(w.parameters)
                    model_for_saving_weights.autoencoder.set_weights(weights_to_array)
                    model_for_saving_weights.autoencoder.save_weights(out+'/client'+str(i)+f'/ae_round_{rnd}_weights.h5')
                                      
            return weights
      
    # pretraining
    strategy_ae = Strategy_ae(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients)
    
    
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
    
    ae_weights = out+f'/server/ae_weights.h5'


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
            self.model.model.set_weights(parameters)
            self.model.clustering(x=self.x_train, y=self.y_train, 
                                     tol=tol, 
                                     maxiter=maxiter,
                                     update_interval=update_interval, 
                                     cluster_method=cluster_method)
            return self.model.model.get_weights(), len(self.x_train), {}
    
        def evaluate(self, parameters, config):
            self.model.model.set_weights(parameters)
            loss,cl_loss,dec_loss,cl_acc,dec_acc = self.model.model.evaluate(self.x_test, self.y_test)
            return loss, len(self.x_test), {"clustering_accuracy": cl_acc,"decoder_accuracy":dec_acc}
    

    
    def client_fn_idec(cid): 
    
        if not os.path.exists(out+'/client'+str(cid)):
            os.makedirs(out+'/client'+str(cid))
            
        # clustering
        model = IDECvar.IDECvar(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters, 
                        batch_size=batch_size,
                        out=out+'/client'+str(cid),
                        setting='federated',)
        model.initialize_model(ae_weights=ae_weights, 
                                  ae_loss_coeff =ae_loss_coeff,
                                  gamma=gamma, 
                                  optimizer=idec_optimizer)
        #model.model.compile(metrics=["accuracy"])
            
        x_train = x_fed[int(cid)].copy()
        y_train = y_fed[int(cid)].copy()
        x_test = x_train.copy()
        y_test = y_train.copy()
            
        return ClusteringClient(model,x_train,y_train,x_test,y_test)
    
    

    class Strategy_idec(fl.server.strategy.FedAvg):
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
                
                model_for_saving_weights = IDECvar.IDECvar(dims=[input_dim, 500, 500, 2000, 10], 
                        n_clusters=n_clusters, 
                        batch_size=batch_size,
                        out=None,
                        setting='federated')
                model_for_saving_weights.initialize_model(ae_weights=ae_weights, 
                                  ae_loss_coeff =ae_loss_coeff,
                                  gamma=gamma, 
                                  optimizer=idec_optimizer)
                model_for_saving_weights.model.set_weights(aggregated_weights)
                model_for_saving_weights.model.save_weights(out+f'/server/idec_weights.h5')
                if rnd == num_rounds_idec:
                    model_for_saving_weights.model.save_weights(out+f'/server/idec_round_{rnd}_weights.h5')
 
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
    

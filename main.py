import dataset
import os
import argparse

parser = argparse.ArgumentParser(description='missing_data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--setting', default='federated', choices=['centralized','federated'])
parser.add_argument('--nan_distribution', default='localized', choices=['uniform','localized'])
args = parser.parse_args()


    
# load dataset
binary_threshold=0.5
x,y = dataset.load_mnist(binary_threshold)
x_fed,y_fed = dataset.create_federated_dataset(x,y,
                                       num_clients=8,
                                       samples_per_cluster_per_client=300)
x_centr,y_centr = dataset.from_fed_to_centr(x_fed,y_fed)

index_dim_nan,mean_list = dataset.dim_choice(x_centr,threshold=12700)
#con 750 samples per cluster per client la threshold era 31300


if args.nan_distribution == 'uniform':
    folder = 'unif'
    cwmd = 8
if args.nan_distribution == 'localized':
    folder = 'loc'
    cwmd = 3
    
if args.setting == 'centralized':
    setting = 'centr'
if args.setting == 'federated':
    setting = 'fed'

for i in range(len(index_dim_nan)):
    if index_dim_nan[i] not in [211,212,407,435,436,601,602]:
        continue
    print('python IDEC_federated.py --out out/'+setting+'_nan_'+folder+'_'+str(index_dim_nan[i])+' --setting '+args.setting+' --client_with_missing_data '+str(cwmd)+' --nan True --nan_dim '+str(index_dim_nan[i]))
    os.system('tmux new -d -s '+setting+'_nan_'+folder+'_'+str(index_dim_nan[i]))
    os.system('tmux send-keys -t '+setting+'_nan_'+folder+'_'+str(index_dim_nan[i])+'.0 "python IDEC_federated.py --out out/'+setting+'_nan_'+folder+'_'+str(index_dim_nan[i])+' --setting '+args.setting+' --client_with_missing_data '+str(cwmd)+' --nan True --nan_dim '+str(index_dim_nan[i])+' --nan_substitute '+str(mean_list[i])+'" ENTER')# --ae_weights out/'+setting+'_nan_'+folder+'_'+str(index_dim_nan[i])+'/ae_round_25_weights.h5" ENTER')

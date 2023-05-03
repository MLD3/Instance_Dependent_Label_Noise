import numpy as np
import torch

import random
import pickle
from datetime import datetime
import os
import sys
import argparse

import get_data
import util
from hyperparams import all_hyperparams, hp_ranges
from data_settings import all_settings 
from exp_settings import all_exp_settings
import process_results

from directories import results_dir

seed = 123456789

###################################################################################################
'''
run an experiment with a specific approach on one dataset
'''
def run_exp(dataset_name, data_package, approach, tune, val_gt, settings, split_seed=0, date='0'):
    #random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
   
    if not os.path.isdir(results_dir + dataset_name + '/'):
        os.mkdir(results_dir + dataset_name + '/')
    
    data_params = all_settings[dataset_name]
    data_params['n_feats'] = data_package[0][0].shape[1]
    
    approach_name = approach
    if 'proposed' in approach:
        approach_name = 'proposed1'
    hyperparams = all_hyperparams[dataset_name][approach_name]
    hyperparam_ranges = hp_ranges[dataset_name][approach]
    
    if tune:
        mod, hyperparams, res = util.tune_hyperparams(data_package, approach, data_params, \
                                hyperparam_ranges, val_gt, results_dir, date, dataset_name)
        print(dataset_name, approach, hyperparams, hyperparam_ranges)
        print(res)
        return res
        
    else:
        mod, _, _ = util.get_model(dataset_name, data_package, approach, data_params, hyperparams, val_gt)
        print(util.eval_overall(mod, data_package[1], data_params, use_gt=True))
   

'''
run an full set of experiments from a list of approaches on one dataset
'''
def run_bulk_exp(dataset_name, approaches, val_gt, split_seed, date, setting, tune):
    time_now = datetime.now()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_params = all_settings[dataset_name]
    data_package = get_data.get_dataset(dataset_name, setting, params=data_params, split_seed=split_seed)

    for i in range(len(approaches)):
        run_exp(dataset_name, data_package, approaches[i], tune, val_gt, setting, split_seed=split_seed, date=date)

    print('################################################################')
    print('done!')
    print(dataset_name)

    end_time = datetime.now()
    print('time elapsed: ', end_time - time_now)


'''
run bulk experiments on different versions of a dataset (example - varying the noise rate on the synthetic data)
'''
def run_bulk_and_vary_setting(dataset_name, approaches, val_gt, split_seed, date, settings, tune, vary=False):
    num_settings = len(settings)

    for i in range(num_settings):
        setting = settings[i]
        run_bulk_exp(dataset_name, approaches, val_gt, split_seed, date + '-' + str(i), setting, tune)
    print('done with varying the setting')

    for i in range(num_settings):
        setting = settings[i]
        print('setting ', setting)
        process_results.postprocess_results(dataset_name, approaches, date + '-' + str(i), val_gt)
        print('************************************************************************')


###################################################################################################
'''
main block

dataset names: synth, MIMIC-ARF, MIMIC-Shock, adult, compas

main approach names: 'baseline_plain_clean', 'baseline_plain', 'baseline_sln_filt', 'baseline_transition', 'baseline_transit_conf', 'baseline_fair_gpl', 'baseline_js_loss', 'proposed1'

proposed ablations names: 'proposed1_anchor', 'proposed1_noisy_p2', 'proposed1_no_p2_theta', 'proposed1_no_p2_beta', 'proposed1_no_ss_beta'

proposed hyperparameter sensitivity analysis names:
    vary alpha 1: 'proposed1_110.01', 'proposed1_110.1', 'proposed1_111', 'proposed1_1110', 'proposed1_11100'
    vary alpha 2: 'proposed1_10.011', 'proposed1_10.11', 'proposed1_111', 'proposed1_1101', 'proposed1_11001'
    vary gamma: 'proposed1_0.0111', 'proposed1_0.111', 'proposed1_111', 'proposed1_1011', 'proposed1_10011'

experiment names: baseline, sa_size, sa_bias, sa_noise_rate, sa_noise_disp
'''
if __name__ == '__main__':
    #random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    time_now = datetime.now()
    date = 'mmdd' 

    parser = argparse.ArgumentParser(description='Datasets and experimental conditions for noisy labels experiments')
    parser.add_argument('--dataset') #use one of the dataset names for the --dataset argument
    parser.add_argument('--experiment', default='baseline') #use one of the experiment names for the --experiment argument
    args = parser.parse_args()

    dataset_name = args.dataset
    tune = True
    val_gt = True #our experiments always used the ground truth labels in the validation set
    
    #approaches to test, change as needed (see above for list of approach names)
    approaches = ['baseline_plain', 'baseline_sln_filt', 'baseline_transition', 'baseline_fair_gpl', \
                  'baseline_transit_conf', 'baseline_js_loss', 'proposed1', 'baseline_plain_clean']
                 
    vary_setting = 'sa' in args.experiment
    exp_setting = all_exp_settings[dataset_name]
    seeds = [0, 1,2,3,4,5,6,7,8,9] #change as needed
    if not vary_setting: #experiment name baseline
        exp_setting = [exp_setting[24]] #evaluate all approaches at 1 setting
    else:
        if args.experiment == 'sa_size': #vary size of alignment set
            exp_setting = exp_setting[0:10]
        elif args.experiment == 'sa_bias': #vary bias in alignment set
            exp_setting = exp_setting[10:20]
        elif args.experiment == 'sa_noise_rate': #vary overall noise rate
            exp_setting = exp_setting[20:30] 
        elif args.experiment == 'sa_noise_disp': #vary difference in noise rate between groups
            exp_setting = exp_setting[30:40] 
    
    for i in range(len(seeds)):
        split_seed = seeds[i]
        date = date + 's' + str(split_seed) + args.experiment
        run_bulk_and_vary_setting(dataset_name, approaches, val_gt, split_seed, date, exp_setting, tune)
        date = date[:4] #remove the seed from the date (assuming the original date is in mmdd form)
        
    end_time = datetime.now()
    print('time elapsed: ', end_time - time_now)


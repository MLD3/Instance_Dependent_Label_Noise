'''
code to load/generate and split data lives here
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import copy
import pandas as pd
import scipy.sparse
import torch
import sparse

import random
import pickle

from directories import mimic_root, adult_root, compas_root

################################################################################################################
'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[1:]
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    f.close()
    return c


'''
normalize data
'''
def normalize(data):
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    
    dims = data.shape
    mins = np.tile(mins, (dims[0], 1))
    maxs = np.tile(maxs, (dims[0], 1))
    
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    data[:, :] = (data[:,:] - mins[:, :]) / (ranges)
    return data


################################################################################################################
'''
random noise
'''
def flip_random(labels, noise_rate, class_cond=False):
    num_samples = labels.shape[0]
    n_labs = copy.deepcopy(labels)

    if not class_cond:
        flip = np.random.choice(num_samples, int(noise_rate * num_samples), replace=False)
        n_labs[flip] = 1 - n_labs[flip]

    elif class_cond:
        for i in range(len(noise_rate)):
            class_i = np.where(labels == i)[0]
            num_i = class_i.shape[0]
            flip = np.random.choice(num_i, int(noise_rate[i] * num_i), replace=False)
            n_labs[class_i[flip]] = 1 - n_labs[class_i[flip]]

    return n_labs


'''
examples flipped according to features
'''
def flip_from_feats(labels, feats, noise_rate, sens_attr, distr_params, min_in=0):
    num_samples = labels.shape[0]
    temp = distr_params[2]
    exp_max = distr_params[3]

    #noise function
    coef_sd, sa_coef = distr_params[0], distr_params[1]
    coef2 = np.random.normal(0, coef_sd, size=(feats.shape[1],)) 
    coef2[sens_attr] = 1*sa_coef #sensitive attribute makes minorities more likely to be mislabeled

    outcome_raw = np.matmul(feats, coef2).reshape(-1)
    
    num_groups = np.unique(min_in).shape[0]
    lab_flip = np.zeros((0,))
    for i in range(num_groups):
        group_i = np.where(min_in == i)[0]
        mislab_prob = (1 / (1 + np.exp(-outcome_raw)))[group_i]
        thresh = np.percentile(mislab_prob, noise_rate[i] * 100) #mislabeling theshold
        print(group_i.shape)
        print(thresh, noise_rate[i] * 100, mislab_prob.shape, np.where(mislab_prob < thresh)[0].shape)
        raw_i = np.where(mislab_prob < thresh)[0]
        lab_flip = np.concatenate((lab_flip, group_i[raw_i]))
    lab_flip = lab_flip.astype(int)
    
    #flip labels
    n_labs = copy.deepcopy(labels)
    n_labs[lab_flip] = 1 - n_labs[lab_flip] 
    
    return n_labs


################################################################################################################
'''
select anchor/alignment points, but this time not necessarily from the points whose observed label matches the ground truth
this will include potentially mislabeled points
this is only biased if prop_known_grp is biased
'''
def select_anchor_biased(prop_known_grp, pop, num_samples, labs, obs_labs, grps, split_seed):
    known_correct = np.zeros((num_samples,))
    total_known = 0

    num_grp = np.unique(grps).shape[0]
    for i in range(num_grp):
        in_grp = np.where(grps == i)[0]
        num_known = int(prop_known_grp[i] * in_grp.shape[0])
        known_samples = np.random.choice(pop[in_grp], num_known, replace=False)
        known_gt = labs[known_samples]
        known_obs = obs_labs[known_samples]
        known_mislab = np.where(np.equal(known_gt, known_obs) == False)[0]

        known_correct[known_samples] = 1 #sample is known to be correctly labeled
        known_correct[known_samples[known_mislab]] = 2 #sample is known to be mislabeled
        total_known += num_known
    print(np.unique(known_correct, return_counts=True))
    print('grp props in anchor points:', np.unique(grps[known_correct > 0], return_counts=True)[1]/total_known)
    print('group breakdown, raw counts: ', np.unique(grps[known_correct > 0], return_counts=True)[1])
    return known_correct


################################################################################################################
'''
find noise rates
'''
def find_noise_rates(n_labs, labs):
    print('noise rates') 
    lab_flip = np.where(n_labs != labs)[0]
    pos = np.where(labs == 1)[0]
    neg = np.where(labs == 0)[0]
    fn = np.intersect1d(pos, lab_flip).shape[0] / pos.shape[0]
    fp = np.intersect1d(neg, lab_flip).shape[0] / neg.shape[0]
    print('fp, fn:  ', fp, fn)
    print('average: ', lab_flip.shape[0] / labs.shape[0]) 
    print('pos rate: ', np.sum(labs) / labs.shape[0])
    

################################################################################################################
'''
synthetic data, can chose noise type
'''
def make_synth(params, settings, dataset_name):
    num_samples = params['n_data']
    num_pos = int(num_samples * params['prop_pos'])
    num_feat = 30
    sens_attr = 0 #feature at index 0
    
    covariates = np.random.normal(0, 1, size=((num_samples, num_feat)))

    #assign majority/minority based on a synthetic sensitive attribute
    def_min = 0.2 
    min_thresh = np.percentile(covariates[:, sens_attr], def_min*100)
    min_samples = np.where(covariates[:, sens_attr] < min_thresh)[0]
    min_in = np.zeros((num_samples,))
    min_in[min_samples] = 1
    majority = np.setdiff1d(np.arange(num_samples), min_samples)
    
    #make feature distributions between majority/minority groups different
    num_diff = 20
    covariates[np.where(min_in == 0)[0], -num_diff:-num_diff//2] = 0
    covariates[np.where(min_in == 1)[0], -num_diff//2:] = 0
    
    #ground truth label
    coef = np.random.normal(0, 1, size=(num_feat, 1)) 
    expon = np.random.randint(1, 3, size=(num_feat,))  

    outcome_raw = np.matmul(np.power(covariates, expon), coef).reshape(-1)
    outcome_prob = 1 / (1 + np.exp(-outcome_raw))
    pos_thresh = np.percentile(outcome_prob, params['prop_pos'] * 100)
    pos_gt = np.where(outcome_prob > pos_thresh)[0]
    labs = np.zeros((num_samples,))
    labs[pos_gt] = 1

    #introduce noise
    noise_rate = settings['noise_rate']
    noise_params = settings['distr_params']
    n_labs = flip_from_feats(labs, covariates, noise_rate, sens_attr, noise_params, min_in)
    
    print('overall')
    find_noise_rates(n_labs, labs) 
    print('minority')
    find_noise_rates(n_labs[np.where(min_in == 1)[0]], labs[np.where(min_in == 1)[0]]) 
    print('majority')
    find_noise_rates(n_labs[np.where(min_in == 0)[0]], labs[np.where(min_in == 0)[0]]) 
    
    return covariates, n_labs, labs, min_in


'''
real data
gets mimic data and formats it for preprocessing - noise from race features

columns of population: ID, names[event]_ONSET_HOUR, names[event]_LABEL
'''
def get_mimic(event, params, dataset_name, settings):
    names = {'arf': 'ARF', 'shock': 'Shock'}
    num_hr = str(4) #length of prediction horizon, 4 hours in this case
    
    population = mimic_root + 'population/' + names[event] + '_' + num_hr + '.0h.csv'
    labs = pd.read_csv(population)[names[event] + '_LABEL'].to_numpy()
    
    time_var = mimic_root + 'features/outcome=' + names[event] + ',T=' + num_hr + '.0,dt=1.0/X.npz'
    feats_var = np.load(time_var)
    feats_var = sparse.COO(feats_var['coords'], feats_var['data'], tuple(feats_var['shape']))
    
    time_invar = mimic_root + 'features/outcome=' + names[event] + ',T=' + num_hr + '.0,dt=1.0/s.npz'
    feats_invar = np.load(time_invar)
    feats_invar = sparse.COO(feats_invar['coords'], feats_invar['data'], tuple(feats_invar['shape']))
    
    feats = feats_invar
    for i in range(feats_var.shape[1]):
        feats = np.concatenate((feats, feats_var[:, i, :]), axis=1)   
    num_feat = feats.shape[1]
    feats = feats.todense()
    
    min_in = np.zeros((feats.shape[0],))
    white, non_white = np.where(feats[:, 34] == 1)[0], np.where(feats[:, 34] != 1)[0]
    min_in[non_white] = 1
    print('race/eth breakdown: ', np.sum(feats[:, 14:38], axis=0), np.sum(np.sum(feats[:, 14:38], axis=0)))
    
    #introduce label noise
    sens_attr = 34
    noise_rate = settings['noise_rate']
    noise_params = settings['distr_params']
    print(noise_params, noise_rate)
    n_labs = flip_from_feats(labs, feats, noise_rate, sens_attr, noise_params, min_in)
        
    print('num encounters and feats', feats.shape[0], feats.shape[1])
    print('min prop', np.sum(min_in) / min_in.shape[0])
    
    print('overall')
    find_noise_rates(n_labs, labs) 
    print('minority non white')
    find_noise_rates(n_labs[np.where(min_in == 1)[0]], labs[np.where(min_in == 1)[0]]) 
    print('majority white')
    find_noise_rates(n_labs[np.where(min_in == 0)[0]], labs[np.where(min_in == 0)[0]]) 

    return feats, n_labs, labs, min_in


'''
real data from fairness literature
part - train or test (this dataset comes presplit)
for feature names see https://github.com/AissatouPaye/Fairness-in-Classification-and-Representation-Learning/blob/master/adult/adult_headers.txt
'''
def get_adult(params, dataset_name, part, settings):
    covariates = np.load(adult_root + part + '/x.npy')
    covariates = np.take(covariates, np.concatenate((np.arange(55), [66], np.arange(71,112))), axis=1)

    labs = np.load(adult_root + part + '/y.npy').reshape(-1)
    print(part, covariates.shape, labs.shape, np.unique(labs))
    num_samples = covariates.shape[0]
    covariates = normalize(covariates)

    #split into groups based on sensitive attribute
    sens_attr = 55
    min_samples = np.where(covariates[:, sens_attr] != 0)[0]
    min_in = np.zeros((num_samples,))
    min_in[min_samples] = 1
    majority = np.setdiff1d(np.arange(num_samples), min_samples)
    print('pos rate within groups: ', np.sum(labs[min_samples]) / min_samples.shape[0], np.sum(labs[majority]) / majority.shape[0])
    print('groups counts: ', np.sum(labs[min_samples]), min_samples.shape[0], np.sum(labs[majority]), majority.shape[0])
    print(np.unique(covariates[:, sens_attr]))

    #introducce label noise
    noise_rate = settings['noise_rate']
    noise_params = settings['distr_params']
    print(noise_params, noise_rate)
    n_labs = flip_from_feats(labs, covariates, noise_rate, sens_attr, noise_params, min_in)

    print('num x and feats', covariates.shape[0], covariates.shape[1])
    print('min prop', np.sum(min_in) / min_in.shape[0])  
    
    print(part, 'overall')
    find_noise_rates(n_labs, labs) 
    print(part, 'minority not male')
    find_noise_rates(n_labs[np.where(min_in == 1)[0]], labs[np.where(min_in == 1)[0]]) 
    print(part, 'majority male')
    find_noise_rates(n_labs[np.where(min_in == 0)[0]], labs[np.where(min_in == 0)[0]]) 
    
    if part == 'train':  
        #randomly pick 1000 peopole to keep
        keep = np.random.choice(covariates.shape[0], size=(1000,), replace=False)
        return covariates[keep, :], n_labs[keep], labs[keep], min_in[keep]
    return covariates, n_labs, labs, min_in


'''
fairness dataset
pre-filtered features:  Index(['Two_yr_Recidivism', 'Number_of_Priors', 'score_factor',
       'Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'African_American',
       'Asian', 'Hispanic', 'Native_American', 'Other', 'Female',
       'Misdemeanor'],
      dtype='object')
filtered features:  Index([0'Number_of_Priors',
       1'Age_Above_FourtyFive', 2'Age_Below_TwentyFive', 3'African_American',
       4'Asian', 5'Hispanic', 6'Native_American', 7'Other', (8,white), 9'Female',
       10'Misdemeanor' (11'Age_25-45')],
      dtype='object')

'''
def get_compas(params, dataset_name, settings):
    data_file = compas_root + 'propublica_data_for_fairml.csv'
    dataset = pd.read_csv(data_file)
    print('features: ', dataset.columns)
    dataset = dataset.to_numpy()
    num_samples = dataset.shape[0]

    labs = dataset[:, 0]
    race_start = 4
    covariates = dataset[:, 1:] #remove the label from feature vector
    covariates = normalize(covariates)
    covariates, race_start = np.delete(covariates, [1], axis=1), 3 #remove propublica's score from features

    #split into groups based on sensitive attribute
    min_in = np.sum(covariates[:, race_start:race_start+5], axis=1)
    min_in[min_in > 0] = 1
    print(np.unique(labs, return_counts=True))
    min_samples = np.where(min_in != 0)[0]
    print(np.unique(min_in[min_samples], return_counts=True))
    majority = np.setdiff1d(np.arange(num_samples), min_samples)
    print('pos rate within groups: ', np.sum(labs[min_samples]) / min_samples.shape[0], np.sum(labs[majority]) / majority.shape[0])
    print('groups counts: ', np.sum(labs[min_samples]), min_samples.shape[0], np.sum(labs[majority]), majority.shape[0])
    #add 1 more feature for white race
    covariates = np.concatenate((covariates[:, :race_start+5], 1-min_in.reshape(-1, 1), covariates[:, race_start+5:]), axis=1)
    sens_attr = race_start + 5
    
    #add 1 more feature for age 25-45 to make it 1 hot
    age_sum = np.sum(covariates[:, [1,2]], axis=1)
    missing_feat = np.zeros((covariates.shape[0], ))
    missing_feat[age_sum == 0] = 1
    covariates = np.concatenate((covariates, missing_feat.reshape(-1, 1)), axis=1)

    #introduce label noise
    noise_rate = settings['noise_rate']
    noise_params = settings['distr_params']
    print(noise_params, noise_rate)
    n_labs = flip_from_feats(labs, covariates, noise_rate, sens_attr, noise_params, min_in)
  
    print('num x and feats', covariates.shape[0], covariates.shape[1])
    print('min prop', np.sum(min_in) / min_in.shape[0])  

    print('overall')
    find_noise_rates(n_labs, labs) 
    print('minority not white')
    find_noise_rates(n_labs[np.where(min_in == 1)[0]], labs[np.where(min_in == 1)[0]]) 
    print('majority white')
    find_noise_rates(n_labs[np.where(min_in == 0)[0]], labs[np.where(min_in == 0)[0]]) 
    print(n_labs == labs)
    return covariates, n_labs, labs, min_in


################################################################################################################
'''
return a subset of the data at the specified indexes
'''
def get_subset(raw_data, labs, labs_gt, min_in, indexes):
    subset_data = torch.Tensor(raw_data[indexes, :])
    subset_labs = torch.Tensor(labs[indexes]).type(torch.LongTensor)
    subset_labs_gt = torch.Tensor(labs_gt[indexes]).type(torch.LongTensor)
    subset_min = torch.Tensor(min_in[indexes]).type(torch.LongTensor)

    return [subset_data, subset_labs, subset_labs_gt, subset_min]

'''
split data into training, validation, and test sets
correctly labeled data can go to training set (correct_where='train') or validation (correct_where='val')
    might want to look at putting a proportion into both...
'''
def split_data(raw_data, labs, labs_gt, min_in, settings, split_seed, mimic=False):  
    np.random.seed(123456789 + split_seed)
    random.seed(123456789 + split_seed)
    
    num_min = np.where(min_in == 1)[0].shape[0]
    overall_anchor = (num_min*settings['anchor_props'][0] + (min_in.shape[0] - num_min)*settings['anchor_props'][1])/raw_data.shape[0]
    print('overall anchor', overall_anchor)
    prop_test = 0.2
    prop_not_train = prop_test + (overall_anchor*0.4) #0.4=0.8*0.5

    #split into training/not training
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=prop_not_train)
    train_i, test_i = next(splitter.split(raw_data, labs))
    train_package = get_subset(raw_data, labs, labs_gt, min_in, train_i)

    pretest_data = raw_data[test_i, :]
    pretest_labs = labs[test_i]
    pretest_labs_gt = labs_gt[test_i]
    pretest_min = min_in[test_i]
    
    #further split test set into test/validation
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=(prop_not_train-prop_test)/prop_not_train) #0.5
    test_i, val_i = next(splitter.split(pretest_data, pretest_labs))
    print(test_i.shape, val_i.shape)
 
    test_package = get_subset(pretest_data, pretest_labs, pretest_labs_gt, pretest_min, test_i)
    val_package = get_subset(pretest_data, pretest_labs, pretest_labs_gt, pretest_min, val_i)

    anchor_props = np.array(settings['anchor_props'])/2 #other half in validation
    anchor = select_anchor_biased(anchor_props, np.arange(train_i.shape[0]), train_i.shape[0], \
                                  labs_gt[train_i], labs[train_i], min_in[train_i], 123456789)
    train_package.append(torch.Tensor(anchor).type(torch.LongTensor))
    
    return train_package, test_package, val_package


'''
this one has its own splitting function because test set comes pre-separated
'''
def split_data_adult(raw_data, labs, labs_gt, min_in, settings, split_seed):
    np.random.seed(123456789 + split_seed)
    random.seed(123456789 + split_seed)
    torch.manual_seed(123456789 + split_seed)
    
    num_min = np.where(min_in == 1)[0].shape[0]
    overall_anchor = (num_min*settings['anchor_props'][0] + (min_in.shape[0] - num_min)*settings['anchor_props'][1])/raw_data.shape[0]
    prop_not_train = overall_anchor/2
    unknown_lab = np.arange(raw_data.shape[0])

    #split into training/not training
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=prop_not_train)
    train_i, val_i = next(splitter.split(raw_data, labs))
    train_package = get_subset(raw_data, labs, labs_gt, min_in, train_i)

    val_data = raw_data[val_i, :]
    val_labs = labs[val_i]
    val_labs_gt = labs_gt[val_i]
    val_min = min_in[val_i]
    val_package = get_subset(val_data, val_labs, val_labs_gt, val_min, np.arange(val_data.shape[0]))

    anchor_props = np.array(settings['anchor_props'])/2
    anchor = select_anchor_biased(anchor_props, np.arange(train_i.shape[0]), train_i.shape[0], \
                                  labs_gt[train_i], labs[train_i], min_in[train_i], 123456789)
    train_package.append(torch.Tensor(anchor).type(torch.LongTensor))

    return train_package, val_package


################################################################################################################
'''
get and preprocess dataset by name
'''
def get_dataset(dataset_name, settings, params=[], split_seed=0):
    print(dataset_name)
    
    if 'synth' in dataset_name:
        dataset = make_synth(params, settings, dataset_name)
        return split_data(dataset[0], dataset[1], dataset[2], dataset[3], settings, split_seed)

    elif 'MIMIC-ARF' in dataset_name:
        arf_data = get_mimic('arf', params, dataset_name, settings)
        return split_data(arf_data[0], arf_data[1], arf_data[2], arf_data[3], settings, split_seed)

    elif 'MIMIC-Shock' in dataset_name:
        shock_data = get_mimic('shock', params, dataset_name, settings)
        return split_data(shock_data[0], shock_data[1], shock_data[2], shock_data[3], settings, split_seed, mimic=True)

    elif 'adult' in dataset_name:
        test_data = get_adult(params, dataset_name, 'test', settings)
        test_data = get_subset(test_data[0], test_data[1], test_data[2], test_data[3], np.arange(test_data[0].shape[0]))
        train_data = get_adult(params, dataset_name, 'train', settings)
        trainval_data = split_data(train_data[0], train_data[1],train_data[2], train_data[3], settings, split_seed)
        return trainval_data[0], test_data, trainval_data[1]

    elif 'compas' in dataset_name:
        dataset = get_compas(params, dataset_name, settings)
        return split_data(dataset[0], dataset[1], dataset[2], dataset[3], settings, split_seed)


################################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

'''
functions related to training and evaluation live here
'''

import copy
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
import torch.nn as nn

import base_feed_for
import base_transition
import base_transition_conf
import base_fair
import base_robust_loss

import proposed

################################################################################################
'''
put things on gpu
'''
def to_gpu(for_gpu):
    dev = 'cpu'
    gpus = [6 ,2, 3, 4, 5, 6, 7, 0] #numbered gpus, change as needed
   
    for i in range(len(gpus)):
        if torch.cuda.is_available():
            dev = 'cuda:' + str(gpus[i]) 
        device = torch.device(dev)

        if not isinstance(for_gpu, list):
            for_gpu = for_gpu.to(device)
            if for_gpu.device.type != 'cpu':
                return for_gpu
        else:
            gpu_items = []
            for item in for_gpu:
                gpu_items.append(item.to(device))
            if gpu_items[0].device.type != 'cpu':
                return gpu_items


################################################################################################
'''
train a model
'''
def train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, use_val_gt, dataset_name, pretraining=False):
    #unpack
    train_cov, train_labs, train_labs_gt, train_min_in, train_gt_in \
        = train_data[0], train_data[1], train_data[2], train_data[3], train_data[4]
    train_dummy = np.zeros((train_cov.shape[0], 1))
    train_weights = np.zeros((np.unique(train_min_in).shape[0], 2)) 
    
    val_cov, val_labs, val_labs_gt, val_min_in = val_data[0], val_data[1], val_data[2], val_data[3]
    val_package = [val_cov, val_labs, val_labs_gt, val_min_in]
    
    #setup
    l_rate, l2_const, num_batch = hyperparams['l_rate'], hyperparams['l2'], hyperparams['batch']
    mod_params = model.get_parameters()
    optimizer = optim.Adam(mod_params, lr=l_rate, weight_decay=l2_const) 
    min_epochs = data_params['min_ep']
    max_epochs = data_params['max_ep']
    patience = 10
    if 'proposed' in approach:
        patience /= 2

    #use gpu
    on_gpu = to_gpu([train_cov, train_labs, train_labs_gt, model])
    train_cov, train_labs, train_labs_gt, model = on_gpu[0], on_gpu[1], on_gpu[2], on_gpu[3]
    
    #initial "evaluation"
    val_loss = 100000
    loss_diff = 100000
    loss_prev = 100000
    loss_tol = 1e-4
    val_avg = 0
    
    #train model 
    i = 1
    prev_mod = copy.deepcopy(model)
    while (loss_diff > loss_tol or i < min_epochs) and i < max_epochs:
        train_loss = 0     
        splitter = KFold(num_batch, shuffle=True)
        batch_split = splitter.split(train_dummy) 
        for j in range(num_batch):
            _, batch_ind = next(batch_split)
            forward_args = {'labs': train_labs[batch_ind], 'new_ep': j == num_batch-1, 'gt_in': train_gt_in[batch_ind], \
                          'gt_labs': train_labs_gt[batch_ind], 'pretrain': pretraining}
            loss_args = {'gt_in': train_gt_in[batch_ind], 'weights': train_weights, \
                         'min_in': train_min_in[batch_ind], 'pretrain': pretraining, 'epoch': i, \
                         'gt_labs': train_labs_gt[batch_ind]}
        
            train_out = model(train_cov[batch_ind, :], True, forward_args)
            batch_loss = loss_fx(train_out, train_labs[batch_ind], loss_args)
                
            if batch_loss != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += (batch_loss.detach() / num_batch)
                
        val_loss = -eval_overall(model, val_package, data_params, use_gt=use_val_gt)['hm']
        val_avg += val_loss / patience

        #evaluate every 10 epochs on validation data 
        if i % patience == 0 and i != 0:   
            forward_args['labs'] = train_labs
            forward_args['gt_labs'] = train_labs_gt
            train_out = model(train_cov, True, forward_args) 
            loss_diff = loss_prev - val_avg
            if loss_diff > 0:
                prev_mod = copy.deepcopy(model)
                loss_prev = copy.deepcopy(val_avg)
            val_avg = 0
            val_eval = eval_overall(model, val_package, data_params, use_gt=use_val_gt)['hm']
            print('new training evaluation')
            print(i, val_loss, loss_diff)
            print(val_eval)
            print('training loss: ', train_loss)   
        i += 1

    print('done training', loss_prev)   
    return prev_mod, loss_prev, i


################################################################################################
'''
equalized odds
'''
def eval_eqod(model, eval_data, data_params):
    inp, labs, gt_labs, min_in = eval_data[0], eval_data[1], eval_data[2], eval_data[3]

    on_gpu = to_gpu([inp, labs])
    inp, labs = on_gpu[0], on_gpu[1]

    preds = (model.forward(inp)).detach().cpu().numpy()
    fp_overall, tp_overall, thresh = roc_curve(gt_labs, preds[:, 1])
    num_thresh = thresh.shape[0]

    minority = np.where(min_in == 1)[0]
    majority = np.where(min_in == 0)[0]
    groups = [majority, minority]
    
    has_pos_preds = []

    tps, fps = np.zeros((num_thresh, len(groups))), np.zeros((num_thresh, len(groups)))
    for i in range(num_thresh): #iterate through thresholds and calculate eq odds
        thresh_i = thresh[i]
        has_pos = True
        for j in range(len(groups)):
            group = groups[j]
            group_preds = preds[group, 1]
            group_labs = gt_labs[group]
            group_pos = np.where(group_labs == 1)[0]
            group_neg = np.where(group_labs == 0)[0]
            pred_pos = np.where(group_preds >= thresh_i)[0]
            if pred_pos.shape[0] > 0:
                #measure true positive
                group_tp = np.intersect1d(group_pos, pred_pos)
                tps[i, j] = group_tp.shape[0] / pred_pos.shape[0]
                #measure false positive
                group_fp = np.intersect1d(group_neg, pred_pos)
                fps[i, j] = group_fp.shape[0] / pred_pos.shape[0]
            else:
                has_pos = False
        if has_pos:
            has_pos_preds.append(i)
    
    #take difference
    tp_diff = np.absolute(tps[has_pos_preds, 0] - tps[has_pos_preds, 1])
    fp_diff = np.absolute(fps[has_pos_preds, 0] - fps[has_pos_preds, 1])
    #apply eq odds definition
    eq_odd = (2 - (tp_diff + fp_diff)) / 2
    
    #take the area under the curve
    results = {'aueo': np.trapz(eq_odd, fp_overall[has_pos_preds])}

    return results


'''
measure model performance (discriminative)
'''
def eval_disc(model, eval_data, use_gt=False):
    cov, labs = eval_data[0], eval_data[1]
    if use_gt and eval_data[2] is not None:
        labs = eval_data[2]

    on_gpu = to_gpu([cov, labs])
    cov, labs = on_gpu[0], on_gpu[1]

    results = {}
    preds = model.forward(cov).detach().cpu().numpy()
    eval_labs = labs.detach().cpu().numpy()
    results['auroc'] = roc_auc_score(eval_labs, preds[:, 1])
    results['aupr'] = average_precision_score(eval_labs, preds[:, 1])
    
    return results
    
'''
overall performance
'''
def eval_overall(model, eval_data, data_params, use_gt=False):
    results = eval_disc(model, eval_data, use_gt)
    eq_od_res = eval_eqod(model, eval_data, data_params)
    results.update(eq_od_res)
    
    if results['auroc'] != 0 and results['aueo'] != 0:
        results['hm'] = 2 / (1/results['auroc'] + 1/results['aueo'])
    else:
        results['hm'] = np.nan
        
    return results


################################################################################################
'''
overall wrapper - train/test/validate a model given the dataset, approach, parameters
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    if approach in ['baseline_plain', 'baseline_sln_filt', 'baseline_plain_clean', 'anchor']:
        model, val_loss, ep = base_feed_for.get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val)
    
    elif approach == 'baseline_transition':
        model, val_loss, ep = base_transition.get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val)

    elif approach == 'baseline_transit_conf':
        model, val_loss, ep = base_transition_conf.get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val)
        
    elif approach == 'baseline_fair_gpl':
        model, val_loss, ep = base_fair.get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val)
        
    elif approach == 'baseline_js_loss':
        model, val_loss, ep = base_robust_loss.get_model(dataset_name, dataset_package, approach, data_params, hyperparams,gt_val)
    
    elif 'proposed' in approach:
        model, val_loss, ep = proposed.get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val)
    
    return model, val_loss, ep


'''
hyperparameter tuning
'''
def tune_hyperparams(dataset_package, approach, data_params, hyperparam_ranges, val_gt, results_dir, date, dataset_name):
    budget = 20
    keys = list(hyperparam_ranges.keys())
    num_hyperparams = len(keys)
    
    test_data = dataset_package[1]
    val_results = np.ones((budget,)) * 1000
    
    best_hyperparams = 1
    best_mod = 1
    num_ep, best_i = 0, -1
    
    #hyperparameter tuning
    for i in range(budget):
        print('hyperparam selection iter ', i)
        hyperparams = {}
        for j in range(num_hyperparams):
            bound = hyperparam_ranges[keys[j]]
            if bound[0] < bound[1] and (not 'fil' in keys[j] and not 'num_parts' in keys[j]):
                hyperparams[keys[j]] = loguniform.rvs(bound[0], bound[1])
            elif bound[0] < bound[1]:
                hyperparams[keys[j]] = np.random.uniform(bound[0], bound[1])
                if keys[j] == 'num_parts':
                    hyperparams[keys[j]] = np.random.randint(bound[0], bound[1] + 1)
            else:
                hyperparams[keys[j]] = bound[0]
        
        print(hyperparams)
        mod, val_loss, ep = get_model(dataset_name, dataset_package, approach, data_params, hyperparams, val_gt)
        val_results[i] = val_loss
        
        #update the best model
        if val_results[i] == np.min(val_results):
            best_mod = mod
            best_hyperparams = hyperparams
            num_ep, best_i = ep, i

    print('using ground truth validation: ', val_gt)
    print('num_epochs, best iter: ', num_ep, best_i, np.min(val_results), val_results)

    #save results
    results = eval_overall(best_mod, test_data, data_params, use_gt=True)
    pickle.dump(results, open(results_dir + dataset_name + '/' + date + '_' + approach + '_' + str(val_gt) + ".pkl", "wb"))

    return best_mod, best_hyperparams, results


################################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

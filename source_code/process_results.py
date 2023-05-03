import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle

from exp_settings import all_exp_settings
from data_settings import all_settings

matplotlib.rcParams.update({'font.size': 20})

from directories import results_dir, plots_dir

approach_names = {'baseline_plain_clean': 'Clean Labels', 'baseline_plain': 'Standard', \
                  'baseline_filt': 'Filter', 'baseline_sln': 'SLN', 'baseline_sln_filt': 'SLN + Filter', \
                  'baseline_transition': 'Transition', 'baseline_fair_reweight': 'Fair Reweight', \
                  'baseline_js_loss': 'JS Loss', 'baseline_transit_conf': 'CSIDN', \
                  'baseline_fair_gpl': 'Fair GPL', 'anchor': 'Anchor Only', 'proposed1': 'Proposed'}
data_names_official = {'synth': 'Synthetic', 'MIMIC-ARF': 'MIMIC-ARF', 'MIMIC-Shock': 'MIMIC-Shock', \
                       'adult': 'Adult', 'compas': 'COMPAS'}


#########################################################################################
'''
print out results nicely
'''
def postprocess_results(dataset, approaches, date, val_gt, show=True):
    if show:
        print(dataset)

    results = []
    boot_keys = ['auroc', 'aupr', 'aueo', 'hm']
    for i in range(len(boot_keys)):
        results.append(np.zeros((len(approaches), 3)))
        
    #non-bootstrapped results
    for i in range(len(approaches)):
        approach = approaches[i]
        file_name = results_dir + dataset + '/' + date + '_' + approach + '_' + str(val_gt) + '.pkl'
        file_handle = open(file_name, 'rb')
        res = pickle.load(file_handle)
        file_handle.close()
        if show:
            print(approach, 'auroc', res['auroc'], 'aupr', res['aupr'], 'aueo', res['aueo'], 'hm', res['hm'])
                
    return results
    

#########################################################################################
'''
plot performance over different conditions (overall noise rate)
'''
def plot_noise_rate(dataset, approaches, date, conditions, cond_lab, results, offset, plot_num, exp_name, metric):
    evals = []
    for i in range(len(approaches)):
        evals.append(np.zeros((len(conditions), 3)))
     #x axis values
    xpoints = np.zeros((len(conditions),))
    for i in range(len(conditions)):
        cond = conditions[i]
        for j in range(len(approaches)):
            approach = approaches[j]
            evals[j][i, 0] = np.mean(results[j][i, :])
            evals[j][i, 1:] = np.std(results[j][i, :])
        xpoints[i] = all_exp_settings[dataset][cond+offset][cond_lab][1]
        if 'rate' in exp_name:
            xpoints[i] = all_exp_settings[dataset][cond+offset][cond_lab][1]

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lines = ['-']
    markers = ['o', 'v', '^', '>', '<', 's', 'x', '*', 'D']
    matplotlib.rcParams.update({'font.size': 16})
    ax = plt.subplot(1, plot_num[1], plot_num[0]) 
    
    for i in range(len(approaches)):
        lab = approach_names[approaches[i]]
        plt.errorbar(xpoints, evals[i][:, 0], yerr=evals[i][:, 1:].T, marker=markers[i], markersize=10, linestyle=lines[0], color=colors[i], label=lab)
    if plot_num[0] == plot_num[1] and plot_num[1] > 1:
        plt.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.65, 0.1)) 
    elif plot_num[1] == 1:
        plt.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.51, 0.1)) 
    matplotlib.rcParams.update({'font.size': 20})
    if plot_num[0] == (plot_num[1] // 2) + 1:
        if exp_name == 'noise_rate':
            plt.xlabel('Minority Noise Rate (Majority Noise Rate +0.2)')
        else:
            plt.xlabel('Minority Noise Rate (Majority Rate Fixed at 20%)')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if plot_num[0] == 1:
        if metric == 'hm':
            plt.ylabel('Harmonic Mean of AUROC and AUEOC')
        elif metric == 'auroc':
            plt.ylabel('AUROC')
        elif metric == 'aueo':
            plt.ylabel('AUEOC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    if plot_num[1] > 1:
        plt.subplots_adjust(right=0.9, top=0.92, left=0.05, bottom=0.15, wspace=0.2)
        plt.title(data_names_official[dataset])
    else:
        plt.subplots_adjust(right=0.71, top=0.92, left=0.13, bottom=0.15)
    
    #save plot    
    if exp_name == 'noise_rate':
        plt.savefig(plots_dir + 'noise_rate_' + metric + '.png', dpi=300)
    else:
        plt.savefig(plots_dir + 'noise_disp_' + metric + '.png', dpi=300)
    
    
'''
plot performance over different conditions (distribution within anchor set)
'''
def plot_anc_distr(dataset, approaches, date, conditions, cond_lab, results, offset, plot_num, metric):
    evals = []
    for i in range(len(approaches)):
        evals.append(np.zeros((len(conditions), 3)))

    xpoints = np.zeros((len(conditions),))
    for i in range(len(conditions)):
        cond = conditions[i]
        for j in range(len(approaches)):
            approach = approaches[j]
            evals[j][i, 0] = np.mean(results[j][i, :])
            evals[j][i, 1:] = np.std(results[j][i, :])
        xpoints[i] = min(all_exp_settings[dataset][cond+offset][cond_lab][1]*all_settings[dataset]['min_prop']*10, 1)
        if 'MIMIC' in dataset:
            xpoints[i] = min(all_exp_settings[dataset][cond+offset][cond_lab][1]*all_settings[dataset]['min_prop']*50, 1)
        if 'adult' in dataset:
            xpoints[i] = min(all_exp_settings[dataset][cond+offset][cond_lab][1]*all_settings[dataset]['min_prop']*20, 1)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lines = ['--', '-']
    markers = ['o', 'v', '^', '>', '<', 's', 'x', '*', 'D']
    matplotlib.rcParams.update({'font.size': 20})
    ax = plt.subplot(1, plot_num[1], plot_num[0]) 
    approaches2 = ['baseline_plain', 'proposed1', 'baseline_plain_clean']
    for i in range(len(approaches)):
        #if not approaches[i] in approaches2: #if want to plot all approaches, comment out this line and the next one
        #    continue
        lab = approach_names[approaches[i]]
        plt.errorbar(xpoints, evals[i][:, 0], yerr=evals[i][:, 1:].T, marker=markers[i], markersize=10, linestyle=lines[0], color=colors[i], label=lab)
    if plot_num[0] == plot_num[1]:
        plt.legend(loc='lower center', frameon=False, bbox_to_anchor=(-2, -0.31), ncol=3)
    matplotlib.rcParams.update({'font.size': 20})
    plt.axvline(all_settings[dataset]['min_prop'], linestyle='dotted', color='k')
    if plot_num[0] == (plot_num[1] // 2) + 1:
        plt.xlabel('Proportion Minority')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if plot_num[0] == 1:
        if metric == 'hm':
            plt.ylabel('Harmonic Mean of AUROC and AUEOC')
        elif metric == 'auroc':
            plt.ylabel('AUROC')
        elif metric == 'aueo':
            plt.ylabel('AUEOC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(data_names_official[dataset])
    plt.subplots_adjust(right=0.98, top=0.92, left=0.05, bottom=0.2, wspace=0.2)
    plt.savefig(plots_dir + 'anc_distr_' + metric + '.png', dpi=300)


'''
plot performance over different conditions (size of anchor set)
'''
def plot_anc_size(dataset, approaches, date, conditions, cond_lab, res, plot_num, metric):
    evals = []
    for i in range(len(approaches)):
        evals.append(np.zeros((len(conditions), 3)))

    xpoints = np.zeros((len(conditions),))
    for i in range(len(conditions)):
        cond = conditions[i]
        for j in range(len(approaches)):
            approach = approaches[j]
            evals[j][i, 0] = np.mean(res[j][i, :])
            evals[j][i, 1:] = np.std(res[j][i, :])
        xpoints[i] = all_exp_settings[dataset][cond][cond_lab][1]

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lines = ['--', '-']
    markers = ['o', 'v', '^', '>', '<', 's', 'x', '*', 'D']
    matplotlib.rcParams.update({'font.size': 20})
    ax = plt.subplot(1, plot_num[1], plot_num[0])
    approaches2 = ['baseline_plain', 'proposed1', 'baseline_plain_clean']
    for i in range(len(approaches)):
        #if not approaches[i] in approaches2: #if want to plot all approaches, comment out this line and the next one
        #    continue
        lab = approach_names[approaches[i]] 
        plt.errorbar(xpoints, evals[i][:, 0], yerr=evals[i][:, 1:].T, marker=markers[i], markersize=10, linestyle=lines[0], color=colors[i], label=lab)
        
    matplotlib.rcParams.update({'font.size': 20})
    if plot_num[0] == (plot_num[1] // 2) + 1:
        plt.xlabel('Proportion of Training Data in Alignment Set')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if plot_num[0] == 1:
        if metric == 'hm':
            plt.ylabel('Harmonic Mean of AUROC and AUEOC')
        elif metric == 'auroc':
            plt.ylabel('AUROC')
        elif metric == 'aueo':
            plt.ylabel('AUEOC')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(data_names_official[dataset])
    plt.subplots_adjust(right=0.98, top=0.92, left=0.05, bottom=0.15, wspace=0.2)
    plt.savefig(plots_dir + 'anc_size_' + metric + '.png', dpi=300)


'''
process experiemnts over several replications
'''
def postprocess_repl(dataset_name, approaches, date, seeds, conditions, cond_lab, val_gt, exp_name, plot_num, metric='hm'):
    num_cond, num_seeds = len(conditions), len(seeds)
    res_all = []  

    #get the results
    for i in range(len(approaches)):
        approach = approaches[i]
        seed_res = np.zeros((num_cond, num_seeds))
        for j in range(len(conditions)):
            cond = conditions[j]
            for k in range(len(seeds)):
                seed = seeds[k]
                new_date = date + 's' + str(seed) + 'sa_' + exp_name + '-' + str(cond)
                if cond == -3:
                    new_date = date + 's' + str(seed) + 'sa_' + 'size' + '-7'       
                file_name = results_dir + dataset_name + '/' + new_date + '_' + approach + '_' + str(val_gt) + '.pkl'
                file_handle = open(file_name, 'rb')
                res = pickle.load(file_handle)
                file_handle.close()
                seed_res[j, k] = res[metric]
        res_all.append(seed_res) 

    #plots the results
    height = 7
    if plot_num[0] == 1:
        plt.clf()
        width = 25
        height = 6
        if plot_num[1] == 1:
            width = 8
        plt.figure(figsize=(width, height)) 
        
    if exp_name == 'size':
        plot_anc_size(dataset_name, approaches, date, conditions, cond_lab, res_all, plot_num, metric)
    elif exp_name == 'bias':
        plot_anc_distr(dataset_name, approaches, date, conditions, cond_lab, res_all, 10, plot_num, metric)
    elif 'noise_disp' in exp_name:
        plot_noise_rate(dataset_name, approaches, date, conditions, cond_lab, res_all, 30, plot_num, exp_name, metric)
    elif 'noise' in exp_name:
        plot_noise_rate(dataset_name, approaches, date, conditions, cond_lab, res_all, 20, plot_num, exp_name, metric)


#########################################################################################
'''
plot multiple datasets at once when varying anchor set
'''
def plot_exp(dataset_names, approaches, date, seeds, conditions, cond_lab, val_gt, exp_name):
    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        plot_num = [i+1, len(dataset_names)]
        postprocess_repl(dataset_name, approaches, date, seeds, conditions, cond_lab, val_gt, exp_name, plot_num, metric='hm')
        
    plt.clf()
    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        plot_num = [i+1, len(dataset_names)]
        postprocess_repl(dataset_name, approaches, date, seeds, conditions, cond_lab, val_gt, exp_name, plot_num, metric='auroc')
    
    plt.clf()    
    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        plot_num = [i+1, len(dataset_names)]
        postprocess_repl(dataset_name, approaches, date, seeds, conditions, cond_lab, val_gt, exp_name, plot_num, metric='aueo')


######################################################################################
'''
main block
'''

if __name__ == '__main__':
    print(':)')
    date = '0520'
    val_gt = True
    approaches = ['baseline_plain', 'baseline_sln_filt', 'baseline_js_loss', 'baseline_transition', \
                  'baseline_transit_conf', 'baseline_fair_gpl', 'proposed1', 'baseline_plain_clean']
    dataset_names = ['synth', 'MIMIC-ARF', 'MIMIC-Shock', 'adult', 'compas']
    seeds = np.arange(10)
    cond_lab = 'anchor_props'

    size_cond = np.arange(1, 10, 1)
    distr_cond = np.arange(0, 10)

    print('varying size of anchor set')
    plot_exp(dataset_names, approaches, date, seeds, size_cond, cond_lab, val_gt, 'size')
    print('varying bias in anchor set')
    plot_exp(dataset_names, approaches, date, seeds, distr_cond, cond_lab, val_gt, 'bias')
    
    cond_lab = 'noise_rate'
    dataset_names = ['synth']
    print('varying noise rate')
    rate_cond = np.arange(0, 10, 1)
    plot_exp(dataset_names, approaches, date, seeds, rate_cond, cond_lab, val_gt, 'noise_rate')
    print('varying noise disparity')
    disp_cond = np.arange(0, 10, 1)
    plot_exp(dataset_names, approaches, date, seeds, disp_cond, cond_lab, val_gt, 'noise_disp')
    

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
make bar plots
'''
def make_bar_plots(approaches, res, labels, title, save_name):
    plot_bars = []
    error_bars = []
    for i in range(len(approaches)):
        appr_res = res[i]
        num_cond = appr_res.shape[0]
        for j in range(num_cond):
            res_j = appr_res[j, :]
            plot_bars.append(np.mean(res_j))
            error_bars.append(np.std(res_j))
    
    plt.figure(figsize=(10, 8)) 
    
    x_points = np.arange(len(approaches))
    plt.xticks([], [])
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i in range(len(approaches)):
        plt.bar(x_points[i], plot_bars[i], yerr=error_bars[i], color=colors[i], label=labels[i])
   
    plt.title(title[0]) 
    plt.ylim(0.5, 1)
    plt.ylabel('Harmonic Mean of AUROC and AUEOC')
    plt.xlabel(title[1])
    plt.legend(loc='upper left')
    
    plt.savefig(plots_dir + save_name + '.png')
    
    
'''
make bar plots
'''
def make_line_plots(approaches, res, labels, title, save_name):
    plot_bars = []
    error_bars = []
    for i in range(len(approaches)):
        appr_res = res[i]
        num_cond = appr_res.shape[0]
        for j in range(num_cond):
            res_j = appr_res[j, :]
            plot_bars.append(np.mean(res_j))
            error_bars.append(np.std(res_j))
    
    plt.figure(figsize=(10, 8)) 
    
    x_points = [float(labels[i]) for i in range(len(labels))]
    plt.errorbar(x_points, plot_bars, yerr=error_bars)
   
    plt.title(title[0]) 
    plt.ylim(0.75, 1)
    plt.ylabel('Harmonic Mean of AUROC and AUEOC')
    plt.xlabel(title[1])
    plt.xscale('log')
    
    plt.savefig(plots_dir + save_name + '.png')
    
'''
def collect data
'''
def collect_data(dataset_name, approaches, conditions, seeds, date):
    num_cond, num_seeds = len(conditions), len(seeds)
    res_all = []  

    for i in range(len(approaches)):
        approach = approaches[i]
        seed_res = np.zeros((num_cond, num_seeds))
        for j in range(len(conditions)):
            cond = conditions[j]
            for k in range(len(seeds)):
                seed = seeds[k]
                new_date = date + 's' + str(seed) + 'baseline-' + str(cond)      
                file_name = results_dir + dataset_name + '/' + new_date + '_' + approach + '_True.pkl'
                file_handle = open(file_name, 'rb')
                res = pickle.load(file_handle)
                file_handle.close()
                seed_res[j, k] = res['hm']
        res_all.append(seed_res)
        
    return res_all 
    
    
#########################################################################################
'''
plot the ablations and hyperparameter sensitivity experiments
'''
if __name__ == '__main__':
    date = '1108'
    
    approaches0 = ['proposed1_anchor', 'proposed1_noisy_p2','proposed1_no_p2_beta', \
                  'proposed1_no_p2_theta',  'proposed1'] 
    approaches1 = ['proposed1_110.01', 'proposed1_110.1', 'proposed1_111', 'proposed1_1110', 'proposed1_11100']
    approaches2 = ['proposed1_10.011', 'proposed1_10.11', 'proposed1_111', 'proposed1_1101', 'proposed1_11001']
    approaches3 = ['proposed1_0.0111', 'proposed1_0.111', 'proposed1_111', 'proposed1_1011', 'proposed1_10011']
    
    labels0 = ['Alignment Only', '+' + r'$\mathcal{L}_{\theta}$' + '\'', '+' + r'$\mathcal{L}_{\theta}$' + '\'+' + r'$\gamma\mathcal{L}_{\theta}$', \
               '+' + r'$\mathcal{L}_{\theta}$' + '\'+' + r'$\alpha_{2}\mathcal{L}_{\beta}$', 'Proposed']
    labels1 = ['0.01', '0.1', '1', '10', '100']
    
    title0 = ['Ablation', 'Approach']
    title1 = ['Hyperparamter Analysis: ' +  r'$\alpha_1$', r'$\alpha_1$']
    title2 = ['Hyperparamter Analysis: ' +  r'$\alpha_2$', r'$\alpha_2$']
    title3 = ['Hyperparamter Analysis: ' +  r'$\gamma$', r'$\gamma$']
    
    conditions = [0]
    
    seeds = np.arange(10)
    
    res0 = collect_data('synth', approaches0, conditions, seeds, date)
    res1 = collect_data('synth', approaches1, conditions, seeds, date)
    res2 = collect_data('synth', approaches2, conditions, seeds, date)
    res3 = collect_data('synth', approaches3, conditions, seeds, date)
    
    make_bar_plots(approaches0, res0, labels0, title0, 'ablat')   
    make_line_plots(approaches1, res1, labels1, title1, 'alpha1')
    make_line_plots(approaches2, res2, labels1, title2, 'alpha2')
    make_line_plots(approaches3, res3, labels1, title3, 'gamma')

'''
record of all hyperparameters
'''

###################################################################################################
'''
synthetic data 
'''

synth1_hyperparams = \
{
    'baseline_plain_clean': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_sln_filt': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'transit': False, 'a1': 0.1, 'add_noise': True}, \
    'baseline_transition': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20, \
                  'fil_thr': 0.8, 'num_parts': 3, 'a1': 0.25, 'a2': 1}, \
    'baseline_transit_conf': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20}, \
    'baseline_fair_gpl': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'weight_lr': 0.001, 'a1': 0.1}, \
    'baseline_js_loss': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, 'pi': 0.5}, \
    'proposed1': {'l_rate': 0.002821181655865712, 'l2': 0.011839034735233178, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'a1': 0.7738530472331953, 'b1': 1}, \
}

#for the proposed approach: alpha1 in the paper is c1, alpha2 is b1, and gamma is a1
synth1_ranges = \
{
    'baseline_plain_clean': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'filter': [False, False], 'fil_thr':[100, 100], 'fil_opp':[100, 100], 'add_noise': [False, False], 
                  'noise_amnt': [0.1, 0.1]}, \
    'baseline_plain': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'filter': [False, False], 'fil_thr':[80, 80], 'fil_opp':[90, 90], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_sln_filt': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'filter': [True, True], 'fil_thr':[60, 90], 'fil_opp':[100, 100], 'add_noise': [True, True], \
                   'noise_amnt': [0.0001, 0.01]}, \
    'baseline_transition': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'fil_thr':[50, 50], 'fil_thr2':[50, 50], 'num_parts': [1, 10], 'a1': [0.1, 10], 'a2': [0.1, 10], 'a3':[0.1, 10]}, \
    'baseline_transit_conf': {'l_rate': [0.0001, 0.1], 'l2': [0.001, 0.1], 'batch':[5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100]},\
    'baseline_fair_gpl': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'weight_lr': [0.0001, 0.0001], 'a1': [0.01, 0.1]}, \
    'baseline_js_loss': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'pi':[0.001, 0.1]}, \
    'proposed1': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_no_pretrain': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_anchor': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_noisy_p2': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_no_p2_theta': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_no_p2_beta': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_no_ss_beta': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
    'proposed1_111': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[1, 1], 'c1': [1, 1]}, \
    'proposed1_110.1': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[1, 1], 'c1': [0.1, 0.1]}, \
    'proposed1_110.01': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[1, 1], 'c1': [0.01, 0.01]}, \
    'proposed1_1110': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[1, 1], 'c1': [10, 10]}, \
    'proposed1_11100': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[1, 1], 'c1': [100, 100]}, \
    'proposed1_10.11': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[0.1, 0.1], 'c1': [1, 1]}, \
    'proposed1_10.011': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[0.01, 0.01], 'c1': [1, 1]}, \
    'proposed1_1101': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[10, 10], 'c1': [1, 1]}, \
    'proposed1_11001': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [1, 1], 'b1':[100, 100], 'c1': [1, 1]}, \
    'proposed1_0.111': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.1, 0.1], 'b1':[1, 1], 'c1': [1, 1]}, \
    'proposed1_0.0111': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [0.01, 0.01], 'b1':[1, 1], 'c1': [1, 1]}, \
    'proposed1_1011': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [10, 10], 'b1':[1, 1], 'c1': [1, 1]}, \
    'proposed1_10011': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                  'a1': [100, 100], 'b1':[1, 1], 'c1': [1, 1]}, \
} 


###################################################################################################
'''
mimic data 
'''
mimic_arf_hyperparams = \
{
    'baseline_plain_clean': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_sln_filt': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'transit': False, 'a1': 0.1, 'add_noise': True}, \
    'baseline_transition': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20, \
                  'fil_thr': 0.8, 'num_parts': 3, 'a1': 0.25, 'a2': 1}, \
    'baseline_transit_conf': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20}, \
    'baseline_fair_gpl': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'weight_lr': 0.001, 'a1': 0.1}, \
    'baseline_js_loss': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, 'pi': 0.5}, \
    'proposed1': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'a1': 1, 'b1': 1}, \
    'anchor': {'l_rate': 0.002821181655865712, 'l2': 0.011839034735233178, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'a1': 0.7738530472331953, 'b1': 1}, \
} 

mimic_arf_ranges = \
{
    'baseline_plain_clean': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'filter': [False, False], 'fil_thr':[100, 100], 'fil_opp':[100, 100], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_plain': {'l_rate':[0.00001, 0.001], 'l2':[0.00001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'filter': [False, False], 'fil_thr':[80, 80], 'fil_opp':[90, 90], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_sln_filt': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'filter': [True, True], 'fil_thr':[50, 90], 'fil_opp':[100, 100], 'a1': [0.1, 0.10], 'add_noise': [True, True], \
                   'noise_amnt': [0.00001, 0.001]}, \
    'baseline_transition': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'fil_thr':[50, 50], 'fil_thr2':[100, 100], 'num_parts': [1, 10], 'a1':[0.1, 10], 'a2':[0.1, 10], 'a3':[0.1, 10]}, \
    'baseline_transit_conf': {'l_rate':[0.0001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2],'layer_s':[500, 500]},\
    'baseline_fair_gpl': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'weight_lr': [0.0001, 0.0001], 'a1': [0.001, 0.1]}, \
    'baseline_js_loss': {'l_rate':[0.00001, 0.001], 'l2':[0.00001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'pi':[0.01, 0.1]}, \
    'proposed1': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
}


mimic_shock_hyperparams = \
{
    'baseline_plain_clean': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_sln_filt': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'transit': False, 'a1': 0.1, 'add_noise': True}, \
    'baseline_transition': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20, \
                  'fil_thr': 0.8, 'num_parts': 3, 'a1': 0.25, 'a2': 1}, \
    'baseline_transit_conf': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20}, \
    'baseline_fair_gpl': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'weight_lr': 0.001, 'a1': 0.1}, \
    'baseline_js_loss': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, 'pi': 0.5}, \
    'proposed1': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'a1': 1, 'b1': 1}, \
    'anchor': {'l_rate': 0.002821181655865712, 'l2': 0.011839034735233178, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'a1': 0.7738530472331953, 'b1': 1}, \
} 

mimic_shock_ranges = \
{
    'baseline_plain_clean': {'l_rate':[0.000001, 0.0001],'l2':[0.000001, 0.01],'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'filter': [False, False], 'fil_thr':[100, 100], 'fil_opp':[100, 100], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_plain': {'l_rate':[0.00001, 0.001], 'l2':[0.00001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'filter': [False, False], 'fil_thr':[80, 80], 'fil_opp':[90, 90], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_sln_filt': {'l_rate':[0.00001, 0.001], 'l2':[0.0001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'filter': [True, True], 'fil_thr':[50, 90], 'fil_opp':[100, 100], 'a1': [0.1, 0.10], 'add_noise': [True, True], \
                   'noise_amnt': [0.00001, 0.001]}, \
    'baseline_transition': {'l_rate':[0.00001, 0.001], 'l2':[0.0000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'fil_thr':[50, 50],'fil_thr2':[100, 100],'num_parts':[1, 10], 'a1':[0.1, 10], 'a2':[0.1, 10], 'a3':[0.1, 10]}, \
    'baseline_transit_conf': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2],'layer_s':[500, 500]},\
    'baseline_fair_gpl': {'l_rate':[0.00001, 0.01], 'l2':[0.00001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'weight_lr': [0.0001, 0.0001], 'a1': [0.001, 0.01]}, \
    'baseline_js_loss': {'l_rate':[0.0001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                  'pi':[0.001, 0.1]}, \
    'proposed1': {'l_rate':[0.00001, 0.001], 'l2':[0.000001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[500, 500], \
                 'a1': [0.01, 10], 'b1':[0.01, 10], 'c1': [0.01, 10]}, \
}

###################################################################################################
'''
adult dataset from fairness literature
'''
adult_hyperparams = \
{
    'baseline_plain_clean': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_sln_filt': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'transit': False, 'a1': 0.1, 'add_noise': True}, \
    'baseline_transition': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20, \
                  'fil_thr': 0.8, 'num_parts': 3, 'a1': 0.25, 'a2': 1}, \
    'baseline_transit_conf': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20}, \
    'baseline_fair_gpl': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'weight_lr': 0.001, 'a1': 0.1}, \
    'baseline_js_loss': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, 'pi': 0.5}, \
    'proposed1': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'a1': 1, 'b1': 1}, \
    'anchor': {'l_rate': 0.002821181655865712, 'l2': 0.011839034735233178, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'a1': 0.7738530472331953, 'b1': 1}, \
} 

adult_ranges = \
{
    'baseline_plain_clean': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'filter': [False, False], 'fil_thr':[100, 100], 'fil_opp':[100, 100], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_plain': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'filter': [False, False], 'fil_thr':[80, 80], 'fil_opp':[90, 90], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_sln_filt': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'filter': [True, True], 'fil_thr':[50, 100], 'fil_opp':[100, 100], 'a1': [0.1, 0.10], 'add_noise': [True, True], \
                   'noise_amnt': [0.001, 0.1]}, \
    'baseline_transition': {'l_rate': [0.001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'fil_thr':[50, 50], 'fil_thr2':[90, 90], 'num_parts': [1, 10], 'a1': [0.1, 10], 'a2': [0.1, 10], 'a3': [0.1, 10]}, \
    'baseline_transit_conf': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch':[5, 5], 'n_layer':[2, 2],'layer_s':[10, 10]},\
    'baseline_fair_gpl': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'weight_lr': [0.0001, 0.0001], 'a1': [0.001, 0.01]}, \
    'baseline_js_loss': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[10, 10], \
                  'pi':[0.01, 0.1]}, \
    'proposed1': {'l_rate': [0.001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'a1': [0.1, 10], 'b1':[0.1, 10], 'c1': [0.1, 10]}, \
}


'''
compas dataset from fairness literature
'''
compas_hyperparams = \
{
    'baseline_plain_clean': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': False, 'fil_thr': 0.8, 'fil_opp': 0.9, 'transit': False, 'a1': 0.1, 'add_noise': False}, \
    'baseline_sln_filt': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'transit': False, 'a1': 0.1, 'add_noise': True}, \
    'baseline_transition': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20, \
                  'fil_thr': 0.8, 'num_parts': 3, 'a1': 0.25, 'a2': 1}, \
    'baseline_transit_conf': {'l_rate': 0.01, 'l2': 0.8, 'batch': 5, 'n_layer': 2, 'layer_s': 20}, \
    'baseline_fair_gpl': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'weight_lr': 0.001, 'a1': 0.1}, \
    'baseline_js_loss': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, 'pi': 0.5}, \
    'proposed1': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'filter': True, 'fil_thr': 90, 'fil_opp': 95, 'a1': 1, 'b1': 1}, \
    'anchor': {'l_rate': 0.002821181655865712, 'l2': 0.011839034735233178, 'batch': 5, 'n_layer': 2, 'layer_s': 100, \
                  'a1': 0.7738530472331953, 'b1': 1}, \
} 

compas_ranges = \
{
    'baseline_plain_clean': {'l_rate': [0.0001, 0.01], 'l2':[0.001, 0.01], 'batch':[5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'filter': [False, False], 'fil_thr':[100, 100], 'fil_opp':[100, 100], 'a1': [1, 1], 'add_noise': [False, False], \
                   'noise_amnt': [0.1, 0.1]}, \
    'baseline_plain': {'l_rate': [0.0001, 0.01], 'l2': [0.0001, 0.01], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'filter': [False, False], 'fil_thr':[80, 80], 'fil_opp':[90, 90], 'a1': [1, 1], 'add_noise': [False, False], \
                  'noise_amnt': [0.1, 0.1]}, \
    'baseline_sln_filt': {'l_rate': [0.00001, 0.01], 'l2': [0.0001, 0.001], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'filter': [True, True], 'fil_thr':[50, 90], 'fil_opp':[100, 100], 'a1': [0.1, 0.10], 'add_noise': [True, True], \
                   'noise_amnt': [0.0001, 0.01]}, \
    'baseline_transition': {'l_rate': [0.00001, 0.01], 'l2':[0.0001, 0.01], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'fil_thr':[50, 50], 'fil_thr2':[90, 90], 'num_parts': [1, 10], 'a1': [0.01, 1], 'a2': [0.01, 1], 'a3': [0.1, 10]}, \
    'baseline_transit_conf': {'l_rate': [0.0001, 0.01], 'l2': [0.0001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2],'layer_s':[10, 10]},\
    'baseline_fair_gpl': {'l_rate': [0.0001, 0.01], 'l2': [0.00001, 0.01], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'weight_lr': [0.0001, 0.0001], 'a1': [0.01, 1]}, \
    'baseline_js_loss': {'l_rate': [0.0001, 0.01], 'l2': [0.0001, 0.01], 'batch':[5, 5], 'n_layer':[2, 2], 'layer_s':[10, 10], \
                  'pi':[0.01, 0.1]}, \
    'proposed1': {'l_rate': [0.001, 0.01], 'l2': [0.001, 0.01], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [10, 10], \
                  'a1': [0.01, 10], 'b1':[0.01, 10], 'c1': [0.01, 10]}, \
}

###################################################################################################
'''
putting everything together
'''

all_hyperparams = \
{
    'synth': synth1_hyperparams, \
    'MIMIC-ARF': mimic_arf_hyperparams, \
    'MIMIC-Shock': mimic_shock_hyperparams, \
    'adult': adult_hyperparams, \
    'compas': compas_hyperparams, \
}

hp_ranges = \
{
    'synth': synth1_ranges, \
    'MIMIC-ARF': mimic_arf_ranges, \
    'MIMIC-Shock': mimic_shock_ranges, \
    'adult': adult_ranges, \
    'compas': compas_ranges, \
}

###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

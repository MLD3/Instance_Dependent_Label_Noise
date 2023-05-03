'''
record of all settings for datasets
'''

synth = \
{
    'name': 'synthetic', \
    'n_data': 5000, \
    'n_class': 2, \
    'prop_pos': 0.5, \
    'weights': [1, 1], \
    'min_ep': 11, \
    'max_ep': 200, \
    'noise_rate': 0.20, \
    'attr_nrate': [0.40, 0.20, 'feat'], \
    'distr_params': [0.5, -0.75, 0.2, 2], \
    'min_prop': 0.2, \
    'min_name': 'Minority', \
}

mimic_arf = \
{
    'name': 'MIMIC-ARF', \
    'n_class': 2, \
    'weights': [1, 1], \
    'noise_feats': {}, \
    'noise_feat_discr': [True, True], \
    'temp': 0.05, \
    'noise_offset': 0, \
    'nw_noise_prop': 0.25, \
    'noise_feat_overlap': -1, \
    'prop_w': -1, \
    'min_ep': 11, \
    'max_ep': 200, \
    'noise_rate': 0.35, \
    'attr_nrate': [0.40, 0.20, 'feat'], \
    'distr_params': [0.10, -0.33, 0.5, 3], \
    'min_prop': 0.278, \
    'min_name': 'Non-White', \
} 

mimic_shock = \
{
    'name': 'MIMIC-Shock', \
    'n_class': 2, \
    'prop_pos': 0.5, \
    'weights': [1, 1], \
    'noise_feats': {}, \
    'noise_feat_discr': [False, False], \
    'noise_offset': 0, \
    'nw_noise_prop': 0.25, \
    'noise_feat_overlap': -1, \
    'prop_w': -1, \
    'temp': 0.25, \
    'min_ep': 11, \
    'max_ep': 200, \
    'noise_rate': 0.35, \
    'attr_nrate': [0.40, 0.20, 'feat'], \
    'distr_params': [0.10, -0.33, 0.5, 3], \
    'min_prop': 0.287, \
    'min_name': 'Non-White', \
} #0.1, -0.4, 0.3, 3, 0.8, 0.8

adult = \
{
    'name': 'adult', \
    'n_data': 5000, \
    'n_class': 2, \
    'prop_pos': 0.5, \
    'weights': [1, 1], \
    'min_ep': 11, \
    'max_ep': 200, \
    'noise_rate': 0.4, \
    'attr_nrate': [0.40, 0.20, 'feat'], \
    'distr_params': [0.8, -0.25, 1.5, 2], \
    'min_prop': 0.325, \
    'min_name': 'Female', \
}

compas = \
{
    'name': 'compas', \
    'n_data': 5000, \
    'n_class': 2, \
    'prop_pos': 0.5, \
    'weights': [1, 1], \
    'min_ep': 11, \
    'max_ep': 200, \
    'noise_rate': 0.45, \
    'attr_nrate': [0.45, 0.35, 'feat'], \
    'distr_params': [1, 1, 0.09, 2], \
    'min_prop': 0.346, \
    'min_name': 'Non-White', \
}
##########################################################################################################
all_settings = \
{
    'synth': synth, \
    'MIMIC-ARF': mimic_arf, \
    'MIMIC-Shock': mimic_shock, \
    'adult': adult, \
    'compas': compas, \
}

##########################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

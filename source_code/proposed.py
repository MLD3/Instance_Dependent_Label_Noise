'''
proposed method
'''

import numpy as np
from scipy import stats
import itertools
import torch
import torch.nn as nn
import util
import copy
import matplotlib.pyplot as plt


###################################################################################################
'''
the overall network
'''
class proposed1_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(proposed1_net, self).__init__()

        self.num_class = data_params['n_class']
        self.num_feats = data_params['n_feats']
        self.num_layers = hyperparams['n_layer']
        self.layer_size = np.floor(hyperparams['layer_s']).astype(int)
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.hidden = []
        self.hidden.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden.append(nn.Linear(self.layer_size, self.layer_size))
        self.output = nn.Linear(self.layer_size, self.num_class)
        self.layers = self.hidden + [self.output]

        self.beta = nn.Linear(self.layer_size + 1, 2)
        self.layers += [self.beta]

        self.layers = nn.ModuleList(self.layers)
           
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.layers[i].weight)
    
    def forward(self, inp, return_extra=False, extra_args=None):
        out = self.hidden[0](inp)
        for i in range(1, len(self.hidden)):
            out = self.activation(out)
            out = self.hidden[i](out)

        if extra_args is not None: #at training time use obs lab to get beta
            beta_inp = torch.cat((out, extra_args['labs'].view(-1, 1)), dim=1)
            beta = self.beta(beta_inp)

        out = self.output(out)
        preds = self.softmax(out)
        
        if return_extra and extra_args is not None:
            return {'input': inp, 'preds': preds, 'emb': out, 'beta': beta}
        elif return_extra:
            return {'input': inp, 'preds': preds, 'emb': out, 'beta': 1}
        
        return preds
    
    def get_parameters(self):
        params = []
        for i in range(len(self.layers)):
            params.append(self.layers[i].parameters())
        
        params = itertools.chain.from_iterable(params)        
        return params
    

###################################################################################################
'''
pretty much cross entropy, making a separate class for this so that the proposed loss can be customized
'''
class proposed1_loss(nn.Module):
    def __init__(self, hyperparams, data_params, approach):
        super(proposed1_loss, self).__init__()
        self.weights = torch.Tensor(data_params['weights'])
        self.weights = util.to_gpu(self.weights)
        
        self.loss = nn.CrossEntropyLoss(weight=self.weights)
        self.a1 = hyperparams['a1']
        self.b1 = hyperparams['b1']
        self.c1 = hyperparams['c1'] 
        
        self.approach = approach

    def forward(self, outputs, labs, extra_args=None):
        preds = outputs['preds']
        raw_out = outputs['emb']
        alt_beta = outputs['beta']

        pretrain = extra_args['pretrain']
        gt_in = extra_args['gt_in']
        gt_labs = extra_args['gt_labs']

        known_correct = np.where(gt_in > 0)[0]
        self.gt_preds = preds[known_correct, :].detach().cpu().numpy()
        noisy = np.where(gt_in == 0)[0]

        #theta loss (l_theta)
        loss = util.to_gpu(torch.Tensor([0]))
        if known_correct.shape[0] > 0:
            loss += self.loss(raw_out[known_correct, :], gt_labs[known_correct]) 

        #beta loss (l_beta)
        obs_corr = np.zeros((known_correct.shape[0],))
        obs_corr[np.where(gt_in[known_correct] == 1)[0]] = 1 #labels match
        obs_corr = util.to_gpu(torch.Tensor(obs_corr).type(torch.LongTensor))
        alt_beta_loss = util.to_gpu(torch.Tensor([0]))
        if known_correct.shape[0] > 0:
            alt_beta_loss = nn.CrossEntropyLoss()(alt_beta[known_correct, :], obs_corr) 
        
        if pretrain:
            return loss + self.c1*alt_beta_loss
        
        #theta prime loss (l_theta prime)   
        nloss = util.to_gpu(torch.Tensor([0])) 
        min_in = extra_args['min_in']  
        num_groups = np.unique(min_in).shape[0]
        for i in range(num_groups):
            noisy = np.intersect1d(np.where(gt_in == 0)[0], np.where(min_in == i)[0])
            if noisy.shape[0] > 0:
                noisy_ce = nn.CrossEntropyLoss(weight=self.weights, reduction='none')(raw_out[noisy, :], labs[noisy])
                alt_beta_weights = nn.Softmax(dim=1)(alt_beta)[noisy, 1]
                if extra_args['epoch'] % 2 == 0:
                    alt_beta_weights = util.to_gpu(torch.Tensor(alt_beta_weights.detach().cpu().numpy()))
                else:
                    noisy_ce = noisy_ce.detach()
                    if 'no_ss_beta' in self.approach:
                        alt_beta_weights = util.to_gpu(torch.Tensor(alt_beta_weights.detach().cpu().numpy()))
                nloss += torch.mean(alt_beta_weights*noisy_ce) / (torch.mean(alt_beta_weights))
    
        #ablations
        if 'noisy_p2' in self.approach:
            return nloss
        elif 'no_p2_beta' in self.approach:
            return nloss + self.a1*loss 
        elif 'no_p2_theta' in self.approach:
            return nloss + self.b1*alt_beta_loss 
        
        #alternation between steps 2a and 2b    
        if extra_args['epoch'] % 2 == 0:
            return nloss + self.a1*loss 
        else:
            return nloss + self.b1*alt_beta_loss 


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    
    model = proposed1_net(hyperparams, data_params)
    loss_fx = proposed1_loss(hyperparams, data_params, approach)
    min_ep = data_params['min_ep']

    model = util.to_gpu([torch.tensor([0]), model])[1]
    
    #pretrain
    if not 'no_pretrain' in approach:  
        known_correct = np.where(train_data[3] > 0)[0]
        gt_data = [train_data[0][known_correct, :]] + [train_data[i][known_correct] for i in range(1, 5)]
        model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, gt_data, val_data, data_params, approach, gt_val, dataset_name, pretraining=True)
  
    #train
    if not 'anchor' in approach:
        data_params['min_ep'] = min_ep
        model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, gt_val, dataset_name)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

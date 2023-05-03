'''
baseline23 
    estimate transition matrices using method from http://proceedings.mlr.press/v139/berthon21a/berthon21a.pdf
    uses estimated confidence scores from predictions since our datasets don't come with them
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
class baseline23_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline23_net, self).__init__()

        self.num_class = data_params['n_class']
        self.num_feats = data_params['n_feats']
        self.num_layers = hyperparams['n_layer']
        self.layer_size = np.floor(hyperparams['layer_s']).astype(int)
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.alpha = np.ones((self.num_class, self.num_class))
        
        #for the actual predictions
        self.hidden = []
        self.hidden.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden.append(nn.Linear(self.layer_size, self.layer_size))
        self.output = nn.Linear(self.layer_size, self.num_class)
        self.layers = self.hidden + [self.output]
        
        #for the pretraining step where the intial predictions/confidence scores are generated
        self.hidden_c = []
        self.hidden_c.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden_c.append(nn.Linear(self.layer_size, self.layer_size))
        self.output_c = nn.Linear(self.layer_size, self.num_class)
        self.layers += self.hidden_c + [self.output_c]
        
        #for the other pretraining step where a model to predict noisy labels is learned
        self.hidden_n = []
        self.hidden_n.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden_n.append(nn.Linear(self.layer_size, self.layer_size))
        self.output_n = nn.Linear(self.layer_size, self.num_class)
        self.layers += self.hidden_n + [self.output_n]

        self.layers = nn.ModuleList(self.layers)
           
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.layers[i].weight)
    
    def forward(self, inp, return_extra=False, extra_args=None):
        out_c = self.hidden_c[0](inp)
        for i in range(1, len(self.hidden_c)):
            out_c = self.activation(out_c)
            out_c = self.hidden_c[i](out_c)
        out_c = self.output_c(out_c)
        preds_c = self.softmax(out_c) #confidence score
        
        out_n = self.hidden_n[0](inp)
        for i in range(1, len(self.hidden_n)):
            out_n = self.activation(out_n)
            out_n = self.hidden_n[i](out_n)
        out_n = self.output_n(out_n)
        preds_n = self.softmax(out_n) #noisy label prediction
        
        out = self.hidden[0](inp)
        for i in range(1, len(self.hidden)):
            out = self.activation(out)
            out = self.hidden[i](out)
        out = self.output(out)
        preds = self.softmax(out) 
        if extra_args is not None and extra_args['pretrain'] == 2:
            preds = self.adjust_preds(preds, preds_c[:, 1], preds_n, extra_args['labs']) #actual (noisy) class predictions
            
        if return_extra:
            return {'input': inp, 'preds': preds, 'conf_emb': out_c, 'noisy_emb': out_n, \
                    'conf_score': preds_c, 'noisy_preds': preds_n}
        
        return preds
        
    def adjust_preds(self, preds, conf_scores, noisy_preds, labs):
        betas = self.compute_beta(preds, noisy_preds)
        transits = self.compute_T(betas, conf_scores, labs)
        unsquee_preds = torch.unsqueeze(preds, dim=2)
        adj_preds = torch.sum(unsquee_preds * transits, dim=2)
        return adj_preds
        
    def compute_beta(self, preds, noisy_preds):
        detached_npred = torch.Tensor(noisy_preds.detach().cpu().numpy())
        detached_npred = util.to_gpu(noisy_preds)
        return (preds / detached_npred)[:, 1]
    
    def compute_T(self, betas, conf_scores, labs):
        transits = np.zeros((betas.shape[0], self.num_class, self.num_class))
                
        #populate diagonal entries
        for i in range(self.num_class):
            has_obs = np.where(labs.detach().cpu().numpy() == i)[0]
            not_obs = np.where(labs.detach().cpu().numpy() != i)[0]
            diag_entries = (conf_scores[has_obs] * betas[has_obs]).detach().cpu().numpy()
            transits[has_obs, i, i] = diag_entries
            transits[not_obs, i, i] = np.mean(diag_entries)
        
        #populate non-diagonal entries using diagonal entries and alphas
        #rows correspond to ground truth label, columns correspond to observed labels
        for i in range(self.num_class):
            for j in range(self.num_class):
                if i == j:
                    continue
                transits[:, i, j] = (1 - transits[:, i, i]) * self.alpha[i, j].detach().cpu().numpy()
        
        return util.to_gpu(torch.Tensor(transits))
        
    def compute_alpha(self, noisy_preds, conf_scores, labs):
        for i in range(self.num_class):
            has_lab = np.where(labs == i)[0]
            preds_i = noisy_preds[has_lab, i]
            scores_i = conf_scores[has_lab, i]
            denom = 1 - torch.mean(preds_i * scores_i)
            for j in range(self.num_class):
                if i == j:
                    continue
                preds_ij = noisy_preds[has_lab, j]
                numer = torch.mean(preds_ij)
                self.alpha[i, j] = numer / denom
        self.alpha = util.to_gpu(torch.Tensor(self.alpha))
    
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
class baseline23_loss(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline23_loss, self).__init__()
        self.weights = torch.Tensor(data_params['weights'])
        self.weights = util.to_gpu(self.weights)
        
        self.loss = nn.CrossEntropyLoss(weight=self.weights) 

    def forward(self, outputs, labs, extra_args=None):
        preds = outputs['preds']
        conf_raw = outputs['conf_emb']
        noisy_raw = outputs['noisy_emb']

        pretrain = extra_args['pretrain']
        gt_in = extra_args['gt_in']
        gt_labs = extra_args['gt_labs']

        known_correct = np.where(gt_in > 0)[0]
        labs[known_correct] = gt_labs[known_correct]

        if pretrain == 0:
            loss = self.loss(noisy_raw, labs)
        elif pretrain == 1:
            loss = self.loss(conf_raw, labs)
        else:
            loss = nn.NLLLoss()(preds, labs) 
            
        return loss 


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    known_correct = np.where(train_data[3] > 0)[0]
    gt_data = [train_data[0][known_correct, :]] + [train_data[i][known_correct] for i in range(1, 5)]
    
    model = baseline23_net(hyperparams, data_params)
    loss_fx = baseline23_loss(hyperparams, data_params)
    
    #pretrain step 0
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams,train_data, val_data, data_params, approach, gt_val, dataset_name, pretraining=0)
    
    #pretrain step 1
    model = util.to_gpu([torch.tensor([0]), model])[1]
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, gt_data, val_data, data_params, approach, gt_val, dataset_name, pretraining=1)
    
    mod_out = model(util.to_gpu(gt_data[0]), True, {'pretrain': 1})
    model.compute_alpha(mod_out['conf_score'], mod_out['noisy_preds'], gt_data[2])
    
    #train
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, gt_val, dataset_name, pretraining=2)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

'''
baseline, type 4 version 2
follows this paper: https://arxiv.org/pdf/2011.00379.pdf
using the group peer loss approach since that seemed to work better in general in the paper
'''

import copy
import numpy as np
import itertools
import torch
import torch.nn as nn

import util


###################################################################################################
'''
the overall network
'''
class baseline42_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline42_net, self).__init__()

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
        out = self.output(out)
        preds = self.softmax(out)
        
        if return_extra:
            return {'input': inp, 'preds': preds, 'emb': out}
        
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
class baseline42_loss(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline42_loss, self).__init__()
        self.weights = torch.Tensor(data_params['weights'])
        self.weights = util.to_gpu(self.weights)
        

        self.loss = nn.CrossEntropyLoss(weight=self.weights)
        self.a1 = hyperparams['a1']

    def forward(self, outputs, labs, extra_args=None):
        preds = outputs['preds']
        raw_out = outputs['emb']
        loss = util.to_gpu(torch.Tensor([0]))
        train_weights = extra_args['weights']
        min_in = extra_args['min_in']
        gt_in = extra_args['gt_in']

        #setup 
        num_groups = np.unique(min_in).shape[0]
        num_classes = preds.shape[1]

        #find peers
        peers = np.zeros((raw_out.shape[0], 2))
        for i in range(num_groups):
            in_group = np.where(min_in == i)[0]
            peers[in_group, 0] = np.random.permutation(in_group)
            peers[in_group, 1] = np.concatenate((peers[in_group[1:], 0], np.array([peers[0, 0]])))
        peers = util.to_gpu(torch.Tensor(peers).type(torch.LongTensor))

        #get class specific loss
        class_losses = self.loss(raw_out, labs)

        #standard loss + transition matrix (on loss, not predictions), also do peer loss
        peer_losses = self.loss(raw_out[peers[:, 0], :], labs[peers[:, 1]])

        loss = class_losses - peer_losses*self.a1
        return loss


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    
    model = baseline42_net(hyperparams, data_params)
    loss_fx = baseline42_loss(hyperparams, data_params)
        
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, gt_val, dataset_name)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

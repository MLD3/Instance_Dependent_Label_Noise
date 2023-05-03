'''
baseline5, implements Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels
'''

import numpy as np
import itertools
import torch
import torch.nn as nn

import util


###################################################################################################
'''
the overall network
'''
class baseline5_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline5_net, self).__init__()

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
        
        self.num_perturbs = 3 #the original paper for this method used 3
        if 'MIMIC' in data_params['name']:
            self.num_perturbs = 1 #3 was really slow on mimic
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.layers[i].weight)
    
    def forward(self, inp, return_extra=False, extra_args=None):
        preds = self.forward_pass(inp)
        
        if not return_extra:
            return preds
        
        perturbs = []
        for i in range(self.num_perturbs):
            perturb = util.to_gpu(torch.randn(inp.shape[0], inp.shape[1])*0.01)
            perturbs.append(self.forward_pass(inp + perturb))
         
        return {'input': inp, 'preds': preds, 'perturbs': perturbs}
    
    def forward_pass(self, inp):
        out = self.hidden[0](inp)
        for i in range(1, len(self.hidden)):
            out = self.activation(out)
            out = self.hidden[i](out)
        out = self.output(out)
        preds = self.softmax(out)
        
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
class baseline5_loss(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline5_loss, self).__init__()
        self.pi = hyperparams['pi']

    def forward(self, outputs, labs, extra_args=None):
        preds = outputs['preds']
        perturbs = outputs['perturbs']
        perturbs = [preds] + perturbs
        
        gt_in = extra_args['gt_in']
        gt_labs = extra_args['gt_labs']
        known_correct = np.where(gt_in > 0)[0]
        labs[known_correct] = gt_labs[known_correct]

        lab_vec = np.ones((labs.shape[0], 2))*0.00001
        labs_np = labs.detach().cpu().numpy()
        lab_vec[np.where(labs_np == 0)[0], 0] = 1
        lab_vec[np.where(labs_np == 1)[0], 1] = 1
        lab_vec = util.to_gpu(torch.Tensor(lab_vec))
        
        scale = -(1 - self.pi) * np.log(1 - self.pi)
        weight = (1 - self.pi) / len(perturbs)
        average = self.pi * lab_vec
        for i in range(len(perturbs)):
            average += weight * perturbs[i]
            
        loss = torch.sum(torch.log(average), dim=1)
        for i in range(len(perturbs)):
            loss -= weight * torch.sum(torch.log(perturbs[i]), dim=1)
        return torch.mean(loss) * scale


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    
    model = baseline5_net(hyperparams, data_params)
    loss_fx = baseline5_loss(hyperparams, data_params)
        
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, gt_val, dataset_name)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

'''
baseline1, is able to 
    train as usual
    apply basic filtering
    use stochastic label noise
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
class baseline1_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline1_net, self).__init__()

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
class baseline1_loss(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline1_loss, self).__init__()
        self.weights = torch.Tensor(data_params['weights'])
        self.weights = util.to_gpu(self.weights)
        
        self.filter = hyperparams['filter']
        self.filter_thresh = hyperparams['fil_thr']
        self.filt_opp = hyperparams['fil_opp']
        
        self.loss = nn.CrossEntropyLoss(weight=self.weights)

        self.add_noise = hyperparams['add_noise']
        self.noise_amnt = hyperparams['noise_amnt']

    def forward(self, outputs, labs, extra_args=None):
        preds = outputs['preds']
        raw_out = outputs['emb']
        loss = util.to_gpu(torch.Tensor([0]))
        gt_in = extra_args['gt_in']
        min_in = extra_args['min_in']
        gt_labs = extra_args['gt_labs']

        diffs = torch.abs(labs - preds[:, 1]).detach().cpu().numpy()
        thresh = self.filter_thresh/100
        keep = np.where(diffs < thresh)[0]
        known_correct = np.where(gt_in > 0)[0]
        keep = np.union1d(keep, known_correct)      

        #give ground truth labels for alignment poitns
        labs[known_correct] = gt_labs[known_correct]

        if (not self.filter and not self.add_noise) or extra_args['epoch'] < 20: #for standard baseline or for burn in
            loss += self.loss(raw_out, labs)
            
        elif self.filter and not self.add_noise:
            loss += self.loss(raw_out[keep, :], labs[keep])

        elif self.add_noise:
            num_class = preds.shape[1]
            class_loss = util.to_gpu(torch.Tensor(np.zeros((num_class,))))
            dummies = [0 for i in range(num_class)]
            for i in range(num_class):
                probs = torch.Tensor(np.random.normal(0, 1, size=labs.shape) * self.noise_amnt)
                probs = util.to_gpu(probs)
                probs[labs == i] += 1      
                if not (self.filter and keep.shape[0] > 0):
                    keep = np.arange(labs.shape[0])
                keep = keep.astype(int)
                pred_probs = preds[keep, i]
                pred_probs[pred_probs < 1e-6] += 1e-6
                dummies[i] = -torch.log(pred_probs)
                dummies[i] = dummies[i] * probs[keep]
                class_loss[i] = torch.mean(dummies[i])
            loss += torch.sum(class_loss)
        
        return loss


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    if 'clean' in approach:
        for dataset in [train_data, test_data, val_data]:
            dataset[1] = dataset[2]
    
    model = baseline1_net(hyperparams, data_params)
    loss_fx = baseline1_loss(hyperparams, data_params)
    
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, gt_val, dataset_name)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

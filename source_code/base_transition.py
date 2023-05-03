'''
baseline22 (2-2), is able to 
    estimate individual level transition matrices "from parts"
    uses method from https://papers.nips.cc/paper/2020/file/5607fe8879e4fd269e88387e8cb30b7e-Paper.pdf
'''

import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable

import copy

import util


###################################################################################################
'''
the overall network
'''
class baseline22_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline22_net, self).__init__()

        self.num_class = data_params['n_class']
        self.num_feats = data_params['n_feats']
        self.num_layers = hyperparams['n_layer']
        self.layer_size = hyperparams['layer_s']
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.hidden = []
        self.hidden.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden.append(nn.Linear(self.layer_size, self.layer_size))
        self.output = nn.Linear(self.layer_size, self.num_class)
        self.layers = self.hidden + [self.output]
           
        self.init_weights()
 
        self.num_parts = hyperparams['num_parts']
        self.autoenc_hid = nn.Linear(self.num_feats, self.num_parts)
        self.autoenc_out = nn.Linear(self.num_parts, self.num_feats)
        self.layers = self.layers + [self.autoenc_hid, self.autoenc_out]
        
        self.part_layers = []
        for _ in range(self.num_parts):
            self.part_layers.append(nn.Linear(1, self.num_class**2))
        self.layers = self.layers + self.part_layers
        self.layers = nn.ModuleList(self.layers)
        self.part_layers = nn.ModuleList(self.part_layers)
    
    def init_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.layers[i].weight)
    
    #predicts ground truth then observed
    def forward(self, inp, return_extra=False, extra_args=None):
        inp = Variable(inp).requires_grad_(True)     
        reconst_inp = Variable(inp).requires_grad_(True)
        reconst_emb = self.autoenc_hid(reconst_inp)
        reconst2 = self.autoenc_out(self.activation(reconst_emb))

        emb = self.autoenc_hid(inp)
        reconst = self.autoenc_out(self.activation(emb))
        emb = self.softmax(emb)
        
        part_matrs = []
        for i in range(self.num_parts):
            dummy = util.to_gpu(torch.Tensor([[1]]))#.to(self.device)
            matr = self.part_layers[i](dummy).view(self.num_class, self.num_class)
            matr = self.softmax(matr)
            part_matrs.append(matr[[0, 1], [1, 0]])
        
        out = self.hidden[0](inp)
        for i in range(1, len(self.hidden)):
            out = self.activation(out)
            out = self.hidden[i](out)
        out = self.output(out)        

        preds = self.softmax(out)
        preds2, fp, fn = self.recover_preds(preds, part_matrs, emb)
        
        if return_extra:
            return {'inp': inp, 'obs_preds': preds2, 'preds': preds, 'reconst': reconst2, \
                    'parts': part_matrs, 'fp': fp, 'fn': fn, 'emb': out, 'reconst_inp': reconst_inp}
        
        return preds
            
    
    def recover_preds(self, preds, parts, embs):
        fp = util.to_gpu(torch.zeros(preds.shape[0],))
        fn = util.to_gpu(torch.zeros(preds.shape[0],))
        for i in range(self.num_parts):
            fp += parts[i][0] * embs[:, i]
            fn += parts[i][1] * embs[:, i]
        
        recovered_preds = util.to_gpu(torch.zeros(preds.shape))
        recovered_preds[:, 0] += preds[:, 0] * (1 - fn) + preds[:, 1] * fn 
        recovered_preds[:, 1] += preds[:, 0] * fp + preds[:, 1] * (1 - fp)    
        
        return recovered_preds, fp, fn
    
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
class baseline22_loss(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline22_loss, self).__init__()
        self.weights = torch.Tensor(data_params['weights'])
        self.weights = util.to_gpu(self.weights)
        
        self.filter_thresh = hyperparams['fil_thr']
        self.filter_thresh2 = hyperparams['fil_thr2']
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.a1 = hyperparams['a1']
        self.a2 = hyperparams['a2']
        self.a3 = hyperparams['a3']

    def forward(self, outputs, labs, extra_args=None):
        loss = 0
        
        inp = outputs['inp']
        reconst = outputs['reconst']
        gt_in = extra_args['gt_in']
        gt_labs = extra_args['gt_labs']

        loss += torch.mean((inp - reconst) ** 2) * self.a1
        
        preds_orig = outputs['preds']
        fp, fn = outputs['fp'], outputs['fn']
        keep = np.where(gt_in > 0)[0]#known ground truth
        if keep.shape[0] > 0:
            fp_diff = (preds_orig[keep, 1][gt_labs[keep]==0] - fp[keep][gt_labs[keep]==0]) ** 2
            fn_diff = (preds_orig[keep, 0][gt_labs[keep]==1] - fn[keep][gt_labs[keep]==1]) ** 2 
            loss += torch.mean(torch.cat((fp_diff, fn_diff))) * self.a2
        
        preds_r = outputs['obs_preds']
        loss += self.ce_loss(preds_r, labs)
        
        output_orig = outputs['emb']
        diffs = torch.abs(labs - preds_orig[:, 1]).detach().cpu().numpy()
        if keep.shape[0] > 0:
            loss += self.ce_loss(output_orig[keep, :], gt_labs[keep]) * self.a3

        return loss


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams, gt_val):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    
    model = baseline22_net(hyperparams, data_params)
    loss_fx = baseline22_loss(hyperparams, data_params)
        
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, train_data, val_data, data_params, approach, gt_val, dataset_name)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.5))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.5))
            self.attention_b.append(nn.Dropout(0.5))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384], "dino_version": [1000, 512, 256]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.5))
        if gate:
            # for "small" configuration: L = 512, D = 256
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        # ATTENTION NETWORK (case "small" configuration):
        #   LINEAR(1024, 512)  features [K, 1024] => [K, 512]
        #   RELU
        #   DROPOUT
        #   ATTENTION NET (512, 256) => N_CLASSES = 1 for single branch (even in case of multi-class classification)
        #   SINGLE BRANCH [K, 512] => [K, 256], [K, 256]
        #   [K, 256] * [K, 256] => [K, 256]
        #   FINAL LAYER: [K,256] => [K,1] attention matrix for the SINGLE branch
        self.attention_net = nn.Sequential(*fc)
        # CLASSIFIER
        #   LINEAR(512, N_CLASSES)
        classifiers = [nn.Linear(size[1], 1)]
        self.classifiers = nn.Sequential(*classifiers)
        #self.classifiers = nn.Linear(size[1], n_classes)
        # instance classifiers -> N_CLASSES binary classifiers for the instance clustering
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        # MOVE THE MODEL TO DEVICE
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # A -> score matrix [1, K]
        # k_sample -> B (set to 8 as default)
        # h -> patch output features [K, 512]
        # top_p_ids -> indexes of the 8 highest scores of A
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        # top_p contains the features of the 8 patches having the highest scores
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        # top_n_ids -> indexes of the 8 lowest scores of A
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        # top_n contains the features of the 8 patches having the lowest scores
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        # p_targets -> tensor of shape [1, k_sample] containing 1s (e.g.: [1,1,1,...,1]
        p_targets = self.create_positive_targets(self.k_sample, device)
        # n_targets -> tensor of shape [1, k_sample] containing 0s (e.g.: [0,0,0,...,0]
        n_targets = self.create_negative_targets(self.k_sample, device)
        # all_targets -> concatenation of p_targets and n_targets (e.g.: [1,1,1,...,1,0,0,0,...,0]
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        idx = torch.randperm(all_targets.shape[0])
        #print('all_targets: ', all_targets[idx].view(all_targets.size()))

        # all_instances -> concatenation of 16 patch features: 8 positive influences (having the highest
        # attention scores) and 8 negative influences (having the lowest attention scores)
        all_instances = torch.cat([top_p, top_n], dim=0)
        all_instances = all_instances[idx].view(all_instances.size())
        # IDEA: THE FIRST K_SAMPLE (8) INSTANCES SHOULD FALL INTO THE FIRST CLUSTER (POSTIVE INFLUENCE),
        # THE LAST 8 IN THE SECOND CLUSTER (NEGATIVE INFLUENCE); the output is [16, 2] if k_sample = 8
        logits = classifier(all_instances)
        # all_preds is a vector indicating, per each of the k_sample*2 patches, the corresponding clustering label
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        # instance loss between logits and targets
        instance_loss = self.instance_loss_fn(logits, all_targets[idx].view(all_targets.size()))
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        # inst_eval_out: in this case, we are working on a class different from the ground-truth one; we want the
        # k_sample patches with the highest attention scores to be considered as NEGATIVE INFLUENCE of such classes,
        # therefore their target is a vector of 0s. (e.g. [0,0,0,0,0,0,0,0] if k_sample = 8
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=True, attention_only=False):
        # h -> patch features
        device = h.device
        # The attention net takes as input the patch features h [K, 1024] - where K is the number of patches - and outputs
        # a [K,1] tensor A containing the attention score per each patch and the features [K, 512] reduced by the fc
        # layers
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        # Softmax over the attention scores
        A = F.softmax(A, dim=1)  # softmax over N

        # Additive MIL part
        m = (A.T * h)
        patch_wise_scores = self.classifiers(m) 
        logits = sum(patch_wise_scores)

        # softmax over logits
        Y_prob = F.sigmoid(logits)
        
        # Y_hat: index of the largest logit
        #Y_hat = torch.topk(logits, 1, dim = 0)[1]
        Y_hat = (Y_prob >= 0.5).float()

        if return_features:
            results_dict.update({'slide_level_embeddings': m, 'attention_scores': A, 'patch_scores':patch_wise_scores})
        return logits, Y_prob, Y_hat, A_raw, results_dict

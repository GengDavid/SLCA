'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers.weight_init import trunc_normal_
from timm.models.layers import Mlp
from copy import deepcopy

class MyContinualClassifier(nn.Module):
    def __init__(self, embed_dim, nb_old_classes, nb_new_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_old_classes = nb_old_classes
        heads = []
        if nb_old_classes>0:
            heads.append(nn.Linear(embed_dim, nb_old_classes, bias=True))
            self.old_head = nn.Linear(embed_dim, nb_old_classes, bias=True)
        heads.append(nn.Linear(embed_dim, nb_new_classes, bias=True))
        self.heads = nn.ModuleList(heads)
        self.aux_head = nn.Linear(embed_dim, 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, with_aux=False):
        assert len(x.size())==2
        out = []
        for ti in range(len(self.heads)):
            out.append(self.heads[ti](x))
        out = {'logits': torch.cat(out, dim=1)}
        if len(self.heads)>1:
            out['old_logits'] = self.old_head(x)
        if with_aux:
            out['aux_logits'] = self.aux_head(x)
        return out

class MlpHead(nn.Module):
    def __init__(self, dim, nb_classes, mlp_ratio=3., drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self._fc = nn.Linear(dim, nb_classes, bias=True)

    def forward(self, x):
        x = x + self.mlp(self.norm(x))
        x = self._fc(x)
        return x

class TaskEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.task_token = nn.Parameter(torch.zeros(1, embed_dim))
        trunc_normal_(self.task_token, std=.02) 
        self.merge_fc = nn.Linear(2*embed_dim, embed_dim)
        trunc_normal_(self.merge_fc.weight, std=.02) 

    def forward(self, x):
        x = F.gelu(self.merge_fc(torch.cat([x, self.task_token.repeat(x.size(0), 1)], 1)))+x
        return x


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False, with_mlp=False, with_task_embed=False, with_preproj=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        self.with_mlp = with_mlp
        self.with_task_embed = with_task_embed
        self.with_preproj = with_preproj
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))
        if with_task_embed:
            single_head.append(TaskEmbed(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        if with_mlp:
            head = MlpHead(embed_dim, nb_classes)
        heads.append(head)
        self.heads = nn.ModuleList(heads)
        if self.with_preproj:
            self.preproj = nn.Sequential(*[nn.Linear(embed_dim, embed_dim, bias=True), nn.GELU()])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
        if self.with_preproj:
            for p in self.preproj.parameters():
                p.requires_grad=False


    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)


    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
        if self.with_task_embed:
            single_head.append(TaskEmbed(self.embed_dim))

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True)
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0) 
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if self.with_mlp:
            head = MlpHead(self.embed_dim, nb_classes)
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02) 
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)          

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False

        if self.with_preproj:
            for p in self.preproj.parameters():
                p.requires_grad=False

        self.heads.append(new_head)

    def forward(self, x):
        #assert len(x.size())==2
        if self.with_preproj:
            x = self.preproj(x)
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True, init_method='kaiming'):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init_method=init_method)

    def reset_parameters(self, init_method='kaiming'):
        if init_method=='kaiming':
            nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        else:
            trunc_normal_(self.weight, std=.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


'''
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': out1['logits'],
            'new_scores': out2['logits'],
            'logits': out
        }
'''

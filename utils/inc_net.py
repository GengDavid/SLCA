import copy
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, MyContinualClassifier, SimpleContinualLinear
from convs.vits import vit_base_patch16_224_in21k, vit_base_patch16_224_mocov3
import torch.nn.functional as F

def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet18_cifar':
        return resnet18(pretrained=pretrained, cifar=True)
    elif name == 'resnet18_cifar_cos':
        return resnet18(pretrained=pretrained, cifar=True, no_last_relu=True)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'vit-b-p16':
        return vit_base_patch16_224_in21k(pretrained=pretrained)
    elif name == 'vit-b-p16-mocov3':
        return vit_base_patch16_224_mocov3(pretrained=True)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False, use_aux=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.use_aux = use_aux
        self.aux_fc = None

    def update_fc(self, nb_classes, new_task_size=-1):
        if self.use_aux:
            assert new_task_size!=-1
            old_tasks = (nb_classes-new_task_size)//new_task_size # WARNING: does not consider the case of difference task sizs
            #aux_fc=self.generate_fc(self.feature_dim, new_task_size+1, bias=True)
            aux_fc=self.generate_fc(self.feature_dim, new_task_size+old_tasks, bias=True)
            if self.aux_fc is not None:
                old_aux_w = copy.deepcopy(self.aux_fc.weight.data)
                aux_fc.weight.data[:old_tasks-1] = old_aux_w[:old_tasks-1]
                compress_old_w = old_aux_w[old_tasks-1:].mean(0)
                aux_fc.weight.data[old_tasks-1] = compress_old_w
                old_bias = copy.deepcopy(self.aux_fc.bias.data)
                aux_fc.bias.data[:old_tasks-1] = old_bias[:old_tasks-1]
                compress_old_b = old_bias[old_tasks-1:].mean(0)
                aux_fc.bias.data[old_tasks-1] = compress_old_b
                del self.aux_fc
            self.aux_fc = aux_fc

        nb_old_classes = nb_classes-new_task_size 
        fc = MyContinualClassifier(self.feature_dim, nb_old_classes, new_task_size) 
        if self.fc is not None:
            weights = [copy.deepcopy(self.fc.heads[hi].weight.data) for hi in range(len(self.fc.heads))]
            weights = torch.cat(weights, dim=0)
            fc.heads[0].weight.data = weights
            fc.old_head.weight.data = weights
            bias = [copy.deepcopy(self.fc.heads[hi].bias.data) for hi in range(len(self.fc.heads))]
            bias = torch.cat(bias, dim=0)
            fc.heads[0].bias.data = bias
            fc.old_head.bias.data = bias
            fc.old_head.weight.requires_grad=False
            fc.old_head.bias.requires_grad=False
        #fc = self.generate_fc(self.feature_dim, nb_classes)
        # if self.fc is not None:
        #     nb_output = self.fc.out_features
        #     weight = copy.deepcopy(self.fc.weight.data)
        #     fc.weight.data[:nb_output] = weight
        #     if self.fc.bias is not None:
        #         bias = copy.deepcopy(self.fc.bias.data)
        #         fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def weight_align(self, increment, align_avg=False):
        #weights=self.fc.weight.data.detach()
        #newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        #oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        newnorm=(torch.norm(self.fc.heads[1].weight,p=2,dim=1))
        oldnorm=(torch.norm(self.fc.heads[0].weight,p=2,dim=1))
        #oldnorm=(torch.norm(self.fc.old_head.weight,p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        if align_avg:
            avgnorm = (meannew+meanold)/2
            #gamma1 = avgnorm/meanold
            gamma2 = avgnorm/meannew
            #self.fc.weight.data[:-increment,:]*=gamma1
            self.fc.weight.data[-increment:,:]*=gamma2
            #return [gamma1, gamma2]   
            return gamma2
        gamma=meanold/meannew
        #gamma = 0.9
        self.fc.heads[1].weight.data*=gamma
        #self.fc.heads[0].weight.data = self.fc.old_head.weight.data
        #self.fc.heads[0].bias.data = self.fc.old_head.bias.data 
        return gamma

    def generate_fc(self, in_dim, out_dim, bias=True):
        fc = SimpleLinear(in_dim, out_dim, bias=bias)

        return fc

    def forward_head(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x, backbone_only=False, with_aux=False, fc_only=False):
        if fc_only:
            return self.forward_head(x)
        x = self.convnet(x)
        if backbone_only:
            return x['fmaps'][-1]
        out = self.fc(x['features'], with_aux=with_aux)
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        #if self.use_aux:
        #    out_aux = self.aux_fc(x['features'])["logits"]
        #    out.update({"aux_logits": out_aux})
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

class DerppIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes):
        #fc = self.generate_fc(self.feature_dim, nb_classes)
        #if self.fc is not None:
        #    nb_output = self.fc.out_features
        #    weight = copy.deepcopy(self.fc.weight.data)
        #    bias = copy.deepcopy(self.fc.bias.data)
        #    fc.weight.data[:nb_output] = weight
        #    fc.bias.data[:nb_output] = bias

        #del self.fc
        #self.fc = fc
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)

        return out


class CosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                #fc.fc1.weight.data = self.fc.weight.data
                fc.weight.data[:self.fc.weight.data.size(0)] = self.fc.weight.data
                fc.sigma.data[:self.fc.weight.data.size(0)] = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        #if self.fc is None:
        #    fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        #else:
        #    prev_out_features = self.fc.out_features // self.nb_proxy
        #    # prev_out_features = self.fc.out_features
        #    fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)
        fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        return fc

class GdumbIncrementalNet(BaseNet):
    def __init__(self, convnet_type, pretrained, new_norm=False):
        super().__init__(convnet_type, pretrained)
        self.init_convnet = copy.deepcopy(self.convnet)
        if new_norm:
            self.fc_norm = nn.LayerNorm(768)
        else:
            self.fc_norm = nn.Identity()

    def forward(self, x, bcb_no_grad=False):
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x)
        else:
            x = self.convnet(x)
        out = self.fc(self.fc_norm(x['features']))
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            del self.fc
        self.fc = fc
        self.convnet = copy.deepcopy(self.init_convnet)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim, init_method='normal')

        return fc



class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False, feat_expand=False, fc_with_ln=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.feat_expand = feat_expand 
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []
        if feat_expand:
            self.expand_base_block = copy.deepcopy(self.convnet.blocks[-1])
        self.fc_with_ln = fc_with_ln

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.bias_correction:
                logits = fc_out['logits']
                for i, layer in enumerate(self.bias_layers):
                    logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
                fc_out['logits'] = logits
            return fc_out  
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x)
        else:
            x = self.convnet(x)
        if self.feat_expand and not isinstance(x['features'], list):
            x['features']  = [x['features']]
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            if bcb_no_grad:
                logits = logits.detach()
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
            out['logits'] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes, freeze_old=False):
        #fc = self.generate_fc(self.feature_dim, nb_classes)
        #if self.fc is not None:
        #    nb_output = self.fc.out_features
        #    weight = copy.deepcopy(self.fc.weight.data)
        #    bias = copy.deepcopy(self.fc.bias.data)
        #    fc.weight.data[:nb_output] = weight
        #    fc.bias.data[:nb_output] = bias

        #del self.fc
        #self.fc = fc

        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)
            if self.feat_expand:
                for p in self.convnet.parameters():
                    p.requires_grad=False
                self.convnet.extra_blocks.append(nn.Sequential(copy.deepcopy(self.expand_base_block), nn.LayerNorm(self.feature_dim))) 
             
        #new_task_size = nb_classes - sum(self.task_sizes)
        new_task_size = nb_classes
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        #fc = SimpleLinear(in_dim, out_dim, init_method='normal')
        fc = SimpleContinualLinear(in_dim, out_dim, feat_expand=self.feat_expand, with_norm=self.fc_with_ln)

        return fc

    def extract_vector(self, x):
        features = self.convnet(x)['features']
        if isinstance(features, list):
            features = torch.stack(features, 0).mean(0)
            
        return features

    def extract_layerwise_vector(self, x):
        with torch.no_grad():
            features = self.convnet(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            features[f_i] = features[f_i].mean(1).cpu().numpy() 
        return features

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(DERNet,self).__init__()
        self.convnet_type=convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features
    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out=self.fc(features) #{logics: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out.update({"aux_logits":aux_logits,"features":features})
        return out        
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self, nb_classes):
        if len(self.convnets)==0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()
    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

class SimpleCosineIncrementalNet(BaseNet):
    
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data=self.fc.sigma.data
            if nextperiod_initialization is not None:
                
                weight=torch.cat([weight,nextperiod_initialization])
            fc.weight=nn.Parameter(weight)
        del self.fc
        self.fc = fc
        

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

class FinetuneIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, fc_with_ln=False, fc_with_mlp=False, with_task_embed=False, fc_with_preproj=False):
        super().__init__(convnet_type, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln
        self.fc_with_mlp = fc_with_mlp
        self.with_task_embed = with_task_embed
        self.fc_with_preproj = fc_with_preproj


    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.convnet(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy() 
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy() 
        return features


    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim, with_norm=self.fc_with_ln, with_mlp=self.fc_with_mlp, with_task_embed=self.with_task_embed, with_preproj=self.fc_with_preproj)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x)
        else:
            x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)

        return out



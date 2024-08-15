import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.slca import SLCA
from utils.inc_net import FinetuneIncrementalNet
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from utils.toolkit import tensor2numpy, accuracy
import copy
import os

epochs = 20
lrate = 0.01 
milestones = [60,100,140]
lrate_decay = 0.1
batch_size = 128
split_ratio = 0.1
T = 2
weight_decay = 5e-4
num_workers = 8
ca_epochs = 5


class SLCApp(SLCA):
    def __init__(self, args):
        super().__init__(args)
        self.model_name_ = args['model_name']
        self.sce_a, self.sce_b = args['sce_a'], args['sce_b']
        assert 'fc_scale_mu' in args.keys() and args['fc_scale_mu']>0
        self.fc_scale_mu = args['fc_scale_mu']

        # hybrid-SL
        for n, p in self._network.convnet.named_parameters():
            if 'norm' in n or 'bias' in n or 'cls_token' in n or 'lora' in n:
                p.requires_grad=True
                print(n)
            else:
                p.requires_grad=False

        # reset global values according to args
        if 'weight_decay' in args.keys():
            global weight_decay
            weight_decay = args['weight_decay']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs']
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs']

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb, learnable_only=True)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(task_size)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)

        if self._cur_task==0 and self._network.convnet.lora_rank>0:
            for b_idx in range(self._network.convnet.lora_lp):
                self._network.convnet.blocks[b_idx].mlp.init_lora()
                self._network.convnet.blocks[b_idx].attn.init_lora()

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[], with_raw=False)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        if self.save_before_ca:
            self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>0 and ca_epochs>0:
            self._stage2_compact_classifier(task_size)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.fc.update_scale()
        

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                cur_logits = logits[:, self._known_classes:]

                ce_loss = F.cross_entropy(cur_logits, cur_targets)
                pred = F.softmax(cur_logits, dim=1)
                pred = torch.clamp(pred, min=1e-7, max=1.0)
                label_one_hot = torch.nn.functional.one_hot(cur_targets, pred.size(1)).float().to(pred.device)
                label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
                rce_loss = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
                loss = self.sce_a*ce_loss + self.sce_b*rce_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        if self._network.convnet.lora_rank>0:
            base_loraA_params = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora_A' in n]
            base_loraB_params = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora_B' in n]
            base_others_params = [p for n, p in self._network.convnet.named_parameters() if p.requires_grad==True and 'lora' not in n]
        else:
            base_params = self._network.convnet.parameters()

        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. 
        if not self.fix_bcb:
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            if self._network.convnet.lora_rank>0:
                lora_scale = 0.5 if self._cur_task==0 else self.bcb_lrscale
                base_loraA_params = {'params': base_loraA_params, 'lr': lrate*lora_scale, 'weight_decay': weight_decay}
                base_loraB_params = {'params': base_loraB_params, 'lr': lrate*lora_scale, 'weight_decay': weight_decay}
                base_others_params = {'params': base_others_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
                network_params = [base_loraA_params, base_loraB_params, base_others_params, base_fc_params]
            else:
                base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
                network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lrate*self.bcb_lrscale*lrate_decay**len(milestones))

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)


    def _stage2_compact_classifier(self, task_size):
        for n, p in self._network.fc.named_parameters():
            p.requires_grad=True
            if 'scales' in n:
                p.requires_grad=False
                print('fixed ', n)

        run_epochs = ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': lrate,
                           'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device)
                cls_cov = self._class_covs[c_id].to(self._device)
                
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

                # a more robust implementation of feature scaling
                # fixed scaling during inference, dynamic during training
                rand_scaling = self.fc_scale_mu + 0.02*(torch.rand(sampled_data_single.size(0), device=self._device)-0.5)
                rand_scaling = 1/(1+rand_scaling*(self._cur_task-t_id))
                sampled_data_single = rand_scaling.unsqueeze(1)*sampled_data_single

                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)



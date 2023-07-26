import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    res_finals, res_avgs = [], []
    for run_id, seed in enumerate(seed_list):
        args['seed'] = seed
        args['run_id'] = run_id
        args['device'] = device
        res_final, res_avg = _train(args)
        res_finals.append(res_final)
        res_avgs.append(res_avg)
    logging.info('final accs: {}'.format(res_finals))
    logging.info('avg accs: {}'.format(res_avgs))
        


def _train(args):
    try:
        os.mkdir("logs/{}_{}".format(args['model_name'], args['model_postfix']))
    except:
        pass
    logfilename = 'logs/{}_{}/{}_{}_{}_{}_{}_{}_{}'.format(args['model_name'], args['model_postfix'], args['prefix'], args['seed'], args['model_name'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top1 avg: {}'.format(np.array(cnn_curve['top1']).mean()))
            if 'task_acc' in cnn_accy.keys():
                logging.info('Task: {}'.format(cnn_accy['task_acc']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top1 avg: {}'.format(np.array(cnn_curve['top1']).mean()))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))

    return (cnn_curve['top1'][-1], np.array(cnn_curve['top1']).mean())

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

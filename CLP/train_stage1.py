import os
import argparse
import datetime
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from train import pretrain
from models import encoder, projectionhead_for32, projectionhead_for64, decoder, SupConLoss
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR

import sys
sys.path.append("/home/lishuoshi/fangsongyu/ARPL/")
from utils import save_model



parser = argparse.ArgumentParser("PreTrain")

# Dataset
parser.add_argument('--dataset', type=str, default='cifar100', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='/home/lishuoshi/Desktop/OSR/data')
parser.add_argument('--outf', type=str, default='../log_cac')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-3, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--model', type=str, default='SupCL+AE')
parser.add_argument('--temperature', type=int, default=0.07)
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--alpha_res', default=1, type=int)
parser.add_argument('--alpha_supcl', default=1, type=int)
# misc
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu',
                    action='store_true')  # 不指定default的时候: action='store_true' 的默认值为 False; action='store_false' 的默认值为 True; 如果指定default: 那就按照default的值作为初始值,例如: action='store_true', default = False的默认值就为 False; 命令行指定了参数: 如果命令行中指定了该参数, 那么就会变为action里面的值, 例如：action='store_true'的参数, 那么命令行指定了该参数，那么它就为 true,和default无关
parser.add_argument('--save-dir', type=str, default='../log')


def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                         img_size=options['img_size'])
        trainloader, train_constrative_loader ,testloader, outloader = Data.train_loader, Data.train_contrastive_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        trainloader, train_constrative_loader,testloader, outloader = Data.train_loader, Data.train_contrastive_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                        img_size=options['img_size'])
        trainloader, train_constrative_loader, testloader, outloader = Data.train_loader, Data.train_contrastive_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        trainloader, train_constrative_loader,testloader = Data.train_loader, Data.train_contrastive_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'],
                                batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader

    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])
        trainloader, train_constrative_loader, testloader, outloader = Data.train_loader, Data.train_contrastive_loader,Data.test_loader, Data.out_loader


    options['num_classes'] = Data.num_classes

    # Model
    Encoder = encoder()
    Decoder = decoder()
    if options['dataset'] == 'tiny_imagenet':
        MLP = projectionhead_for64(options['feature_dim'])
    else:
        MLP = projectionhead_for32(options['feature_dim'])

    # Loss
    options.update(
        {
            'use_gpu': use_gpu
        }
    )

    criterion = SupConLoss ()

    if use_gpu:
        Encoder = Encoder.cuda()
        Decoder = Decoder.cuda()
        MLP = MLP.cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'Pretrain_models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if options['dataset'] == 'cifar100':
        model_path += '_{}'.format(options['out_num'])
        file_name = '{}_{}'.format(options['model'], options['item'])
    else:
        file_name = '{}_{}'.format(options['model'], options['item'])

    params_list = [{'params': Encoder.parameters()},
                   {'params': Decoder.parameters()},
                   {'params': MLP.parameters()},
                   {'params': criterion.parameters()}]

    optimizer = torch.optim.Adam(params_list, lr=options['lr'])

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150])

    # training loop

    # start_time = time.time()
    results = {'train_loss': []}

    for epoch in range(options['max_epoch']):

        loss_pretrain = pretrain(Encoder, MLP, Decoder, criterion, optimizer, train_constrative_loader, epoch=epoch, **options)
        results['train_loss'].append(loss_pretrain)

        if options['stepsize'] > 0: scheduler.step()
        if (epoch+1) == options['max_epoch']:
            save_model(Encoder, model_path, file_name)

    return results


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    simclr_train = dict()

    from split_cac import splits_2020 as splits
    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset'] + '-' + str(options['out_num'])][
                len(splits[options['dataset']]) - i - 1]
        elif options['dataset'] == 'tiny_imagenet':
            img_size = 64
            options['feature_dim'] = 256
            unknown = list(set(list(range(0, 200))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))
        print(known)
        print(unknown)
        options.update(
            {
            'item': i,
            'known': known,
            'unknown': unknown,
            'img_size': img_size
            }
        )
        dir_name = '{}'.format(options['model'])
        dir_path = os.path.join(options['outf'], 'pretrain_loss', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if options['dataset'] == 'cifar100':
            file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])

        else:
            file_name = options['dataset'] + '.csv'


        res = main_worker(options)

        res['unknown'] = unknown
        res['known'] = known
        simclr_train[str(i)] = res

        df = pd.DataFrame(simclr_train)  # 创建DataFrame的单元格存放结果
        df.to_csv(os.path.join(dir_path, file_name))  # 保存在相应位置
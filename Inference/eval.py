import os
import os.path as osp
import numpy as np
import torch.nn as nn

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.append("/media/disk4/fsy2/Contrastive_learning_based_OSR/")
from core import evaluation

def eval(net, net_teacher, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    net_teacher.eval()
    correct, total = 0, 0
    criterion_cos = torch.nn.CosineSimilarity(dim=1).cuda()

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, out, feature_student1, feature_student2, feature_student3 = net(data, True)
                feature_teacher1, feature_teacher2, feature_teacher3 = net_teacher(data)
                logits, _ = criterion(x, out)

                # different terms of KD_loss

                loss_1 = 1 - criterion_cos(feature_student1.view(feature_student1.shape[0], -1),
                                                      feature_teacher1.view(feature_teacher1.shape[0], -1))
                abs_loss_1 = torch.mean((feature_student1 - feature_teacher1) ** 2, dim=(1, 2, 3))
                loss_2 = 1 - criterion_cos(feature_student2.view(feature_student2.shape[0], -1),
                                                      feature_teacher2.view(feature_teacher2.shape[0], -1))
                abs_loss_2 = torch.mean((feature_student2 - feature_teacher2) ** 2, dim=(1, 2, 3))
                loss_3 = 1 - criterion_cos(feature_student3.view(feature_student3.shape[0], -1),
                                                      feature_teacher3.view(feature_teacher3.shape[0], -1))
                abs_loss_3 = torch.mean((feature_student3 - feature_teacher3) ** 2, dim=(1, 2, 3))

                loss_dil = loss_1 + loss_2 + loss_3 + 0.2 * (abs_loss_1 + abs_loss_2 + abs_loss_3)

                logits = logits / loss_dil.view(-1, 1)

                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, out, feature_student1, feature_student2, feature_student3 = net(data, True)
                feature_teacher1, feature_teacher2, feature_teacher3 = net_teacher(data)
                logits, _ = criterion(x, out)

                # different terms of KD_loss
                loss_1 = 1 - criterion_cos(feature_student1.view(feature_student1.shape[0], -1),
                                                      feature_teacher1.view(feature_teacher1.shape[0], -1))
                abs_loss_1 = torch.mean((feature_student1 - feature_teacher1) ** 2, dim=(1, 2, 3))

                loss_2 = 1 - criterion_cos(feature_student2.view(feature_student2.shape[0], -1),
                                                      feature_teacher2.view(feature_teacher2.shape[0], -1))
                abs_loss_2 = torch.mean((feature_student2 - feature_teacher2) ** 2, dim=(1, 2, 3))
                loss_3 = 1 - criterion_cos(feature_student3.view(feature_student3.shape[0], -1),
                                                      feature_teacher3.view(feature_teacher3.shape[0], -1))
                abs_loss_3 = torch.mean((feature_student3 - feature_teacher3) ** 2, dim=(1, 2, 3))

                loss_dil = loss_1 + loss_2 + loss_3 + 0.2 * (abs_loss_1 + abs_loss_2 + abs_loss_3)

                logits = logits / loss_dil.view(-1, 1)

                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results
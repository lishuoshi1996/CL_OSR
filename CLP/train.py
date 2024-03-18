import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from thop import profile, clever_format
import argparse
import os
import pandas as pd
import numpy
import torch.nn as nn

def pretrain(encoder, MLP, decoder, criterion, optimizer, trainloader, epoch=None, **options):
    encoder.train()
    MLP.train()
    decoder.train()
    mse = nn.MSELoss()

    torch.cuda.empty_cache()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(trainloader)

    for i, (images, label) in enumerate(train_bar):
        images_contrastive = torch.cat([images[0], images[1]], dim=0)
        image_reconstruct = images[2]
        if options['use_gpu']:
            images_contrastive, image_reconstruct, label = images_contrastive.cuda(non_blocking=True), image_reconstruct.cuda(non_blocking=True), label.cuda(non_blocking=True)

        features_contrastive = encoder(images_contrastive)
        outs_contrastive = MLP(features_contrastive)

        batch_size = options['batch_size']
        temperature = options['temperature']
        out1, out2 = torch.split(outs_contrastive, [batch_size, batch_size], dim=0)
        outs = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1)], dim=1)
        loss_contrast = criterion(outs, label)

        feature_reconstruct = encoder(image_reconstruct)
        reconstruct_image = decoder(feature_reconstruct)
        loss_re = mse(image_reconstruct, reconstruct_image)

        loss = options['alpha_supcl']*loss_contrast + options['alpha_res']*loss_re
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        loss = total_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, options['max_epoch'], loss))

    return loss



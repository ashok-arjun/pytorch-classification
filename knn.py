import argparse
import os
import shutil
import time
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import ngtpy

def fit_knn_model_embeddings(trainloader, model, criterion, use_cuda):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    train_embeddings = None
    train_targets = None

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            features, outputs = model.forward(inputs, features=True)
            loss = criterion(outputs, targets)
            
            # append features/targets
            if type(train_embeddings) != type(None):
                train_embeddings = torch.cat([train_embeddings, features])
            else:
                train_embeddings = features
            
            if type(train_targets) != type(None):
                train_targets = torch.cat([train_targets, targets])
            else:
                train_targets = targets

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(trainloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish() 

    ### Create KNN index
    train_embeddings = train_embeddings.cpu().numpy()
    np.save("features/cifar100test_alexnet.npy", train_embeddings)
    
    dim = train_embeddings.shape[1]
    index_path = "indexes/cifar100_index.anng"
    ngtpy.create(index_path, dim, distance_type="Cosine")
    index = ngtpy.Index(index_path)
    index.batch_insert(train_embeddings)
    index.save()

    print("\n\n\nCREATED KNN INDEX!!!\n\n\n")

    return index, train_embeddings, train_targets


def get_knn_predictions(index, train_targets, testloader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_knn = AverageMeter()
    top1_knn = AverageMeter()
    top5_knn = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            feats, outputs = model(inputs, features=True)
            knn_preds = []
            for i in range(feats.shape[0]):
                knn_result = index.search(feats[i].cpu().numpy(), epsilon=0.1)
                prediction = train_targets[knn_result[0][0]]
                prediction_prob = torch.zeros(outputs.shape[1])
                prediction_prob[prediction] = 1
                knn_preds.append(prediction_prob)
            predictions = torch.stack(knn_preds, axis=0).squeeze()

            if use_cuda:
                predictions = predictions.cuda()

            loss = criterion(outputs, targets)
            loss_knn = criterion(predictions, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure accuracy and record loss
            prec1_knn, prec5_knn = accuracy(predictions.data, targets.data, topk=(1, 5))
            losses_knn.update(loss_knn.item(), inputs.size(0))
            top1_knn.update(prec1_knn.item(), inputs.size(0))
            top5_knn.update(prec5_knn.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = \
                '({batch}/{size}) Data: {data:.3f}s |' + \
                'Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |' \
                'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}' \
                'Loss KNN: {loss_knn:.4f} | top1 KNN: {top1_knn: .4f} | top5 KNN: {top5_knn: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        loss_knn=losses_knn.avg,
                        top1_knn=top1_knn.avg,
                        top5_knn=top5_knn.avg
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)
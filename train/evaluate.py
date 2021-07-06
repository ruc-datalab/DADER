"""Adaptation to train target encoder."""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("..")
import param
from utils import make_cuda,save_model,init_model
import csv
import os
import datetime

def evaluate(encoder, classifier, data_loader,args=None,flag=None,discriminator=None,exp_idx=None):
    """Evaluation for encoder and classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0
    need_preds = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    count = 0
    # evaluate network
    for (reviews, mask,segment, labels,exm_id) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        truelen = torch.sum(mask, dim=1)

        with torch.no_grad():
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        if args != None and (args.need_pred_res or args.al1 or args.al2):
            for i in range(len(labels)):
                confidence = preds.data[i].cpu().numpy().tolist()
                entro = abs(confidence[1]-confidence[0])
                need_preds.append([exm_id[i].item(),labels[i].item(),pred_cls[i].item(),confidence,entro])

        acc += pred_cls.eq(labels.data).cpu().sum().item()
        for i in range(len(labels)):
            if labels[i] == 1:
                p += 1
                if pred_cls[i] == 1:
                    tp += 1
            else:
                if pred_cls[i] == 1:
                    fp += 1

    div_safe = 0.000001
    print("===== RES ====")
    print("p",p)
    print("tp",tp)
    print("fp",fp)
    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)
    print("recall",recall)
    print("precision",precision)
    print("f1",f1)

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    return f1

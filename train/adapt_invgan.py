import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import param
from utils import make_cuda,save_model,init_model
from train.evaluate import evaluate
import csv
import os
import math
import datetime

def adapt_adda_best(args, src_encoder, tgt_encoder, discriminator,
          src_classifier, src_data_loader, tgt_data_train_loader, tgt_data_valid_loader):
    """INvGAN with valid data"""

    # set train state for Dropout and BN layers
    src_encoder.eval()
    src_classifier.eval()
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=args.d_learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    best_f1 = 0
    res = -1.0
    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask,src_segment, _,_), (reviews_tgt, tgt_mask,tgt_segment, _,_)) in data_zip:
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)
            src_segment = make_cuda(src_segment)

            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)
            tgt_segment = make_cuda(tgt_segment)

            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = src_encoder(reviews_src, src_mask,src_segment)
            feat_tgt = tgt_encoder(reviews_tgt, tgt_mask,tgt_segment)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src.size(0))).unsqueeze(1)
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for discriminator
            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            # optimize discriminator
            optimizer_D.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = args.temperature

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # compute loss for target encoder
            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = args.alpha * gen_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), args.max_grad_norm)
            # optimize target encoder
            optimizer_G.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         gen_loss.item(),
                         dis_loss.item()
                         ))
        
        f1 = evaluate(tgt_encoder, src_classifier, tgt_data_valid_loader)
        if f1 > best_f1:
            print("best epoch: ",epoch)
            print("F1: ",f1)
            best_f1 = f1
            # save_model(args, tgt_encoder, param.src_encoder_path+'adkd_best')
            print("======== tgt result =======")
            res = evaluate(tgt_encoder, src_classifier, tgt_data_train_loader)
    return tgt_encoder,discriminator,res, best_f1

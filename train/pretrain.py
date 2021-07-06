"""Pretrain F and M with labeled Source data."""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import param
from utils import make_cuda,save_model,init_model
import csv
import os
import datetime
from train.evaluate import evaluate
def pretrain(args, encoder, classifier, data_loader, tgt_data_loader):
    """Train F and M for source domain without valid dataset."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    start = datetime.datetime.now()
    for epoch in range(args.pre_epochs):
        for step, (reviews, mask,segment, labels) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            segment = make_cuda(segment)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
        if args.rec_epoch:
            """record the F1 score for each epoch"""
            end = datetime.datetime.now()
            now_time = end-start
            res = evaluate(encoder, classifier, tgt_data_loader)
            f = open(args.epoch_path + args.src + args.srcfix + args.tgt + args.tgtfix+'.csv','a+',encoding='utf-8',newline="")
            csv_writer = csv.writer(f)
            row = [epoch+1,res,now_time]
            csv_writer.writerow(row)
            f.close()
    # save final model
    save_model(args, encoder, param.src_encoder_path)
    save_model(args, classifier, param.src_classifier_path)

    return encoder, classifier

def pretrain_best(args, encoder, classifier, data_loader, tgt_data_valid_loader,tgt_data_train_loader):
    """Train F and M for source domain with valid dataset."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    start = datetime.datetime.now()
    best_f1 = 0
    tgt_res = -1
    best_epoch = -1
    for epoch in range(args.pre_epochs):
        for step, (reviews, mask,segment, labels,_) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            segment = make_cuda(segment)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
        
        if args.rec_epoch:
            """record the F1 score for each epoch"""
            end = datetime.datetime.now()
            now_time = end-start
            res = evaluate(encoder, classifier, tgt_data_train_loader)
            f = open(args.epoch_path + args.src+args.srcfix+args.tgt+args.tgtfix+str(args.train_seed)+args.rec_lr+'.csv','a+',encoding='utf-8',newline="")
            csv_writer = csv.writer(f)
            row = [epoch,res,now_time]
            csv_writer.writerow(row)
            f.close()
        if 1:
            f1 = evaluate(encoder, classifier, tgt_data_valid_loader)
            if f1 > best_f1:
                print("best epoch number: ",epoch)
                print("valid F1: ",f1)
                best_f1 = f1
                if args.rec_epoch:
                    tgt_res = res
                else:
                    print("tgt_res:")
                    tgt_res = evaluate(encoder, classifier, tgt_data_train_loader)
                if epoch < 5:
                    """
                    Save the best Baseline of the first five epochs for later adaptation;
                    To prevent over-fitting of the model on the Source, we only pretrain 5 epochs.
                    """
                    best_epoch = epoch
                    save_model(args, encoder, param.src_encoder_path+args.tgt+'best')
                    save_model(args, classifier, param.src_classifier_path+args.tgt+'best')
        
    encoder = init_model(args, encoder, restore=param.src_encoder_path+args.tgt+'best')
    classifier = init_model(args, classifier, restore=param.src_classifier_path+args.tgt+'best')
    return encoder, classifier,tgt_res

def pretrain_best_semi(args, encoder, classifier, data_loader, tgt_data_valid_loader,tgt_data_test_loader):
    """Train F and M for source domain with valid dataset and record data for semi."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    start = datetime.datetime.now()
    best_f1 = 0
    tgt_res = -1
    best_epoch = -1
    for epoch in range(args.pre_epochs):
        for step, (reviews, mask,segment, labels,_) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            segment = make_cuda(segment)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
        
        if 1:
            f1 = evaluate(encoder, classifier, tgt_data_valid_loader)
            if f1 > best_f1:
                print("best epoch number: ",epoch)
                print("valid F1: ",f1)
                best_f1 = f1
                print("tgt_res:")
                tgt_res = evaluate(encoder, classifier, tgt_data_test_loader)
                """Save the best Baseline for semi-supervised learning """
                save_model(args, encoder, param.src_encoder_path+args.tgt+'best_semi')
                save_model(args, classifier, param.src_classifier_path+args.tgt+'best_semi')

                if epoch < 5:
                    best_epoch = epoch
                    """
                    Save the best Baseline of the first five epochs for later adaptation;
                    To prevent over-fitting of the model on the Source, we only trained 5 epochs.
                    """
                    save_model(args, encoder, param.src_encoder_path+args.tgt+'best_semi_base')
                    save_model(args, classifier, param.src_classifier_path+args.tgt+'best_semi_base')

    encoder = init_model(args, encoder, restore=param.src_encoder_path+args.tgt+'best_semi_base')
    classifier = init_model(args, classifier, restore=param.src_classifier_path+args.tgt+'best_semi_base')
    return encoder, classifier,tgt_res

def pretrain_best_label(args, encoder, classifier, data_loader, tgt_data_valid_loader,ite=None):
    """Train F and M for few labeled target data (Active Learning)."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=args.semi_lr)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    start = datetime.datetime.now()
    best_f1 = 0
    tgt_res = -1
    best_epoch = -1
    for epoch in range(args.al_epochs):
        for step, (reviews, mask,segment, labels,_) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            segment = make_cuda(segment)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
        
        if 1:
            f1 = evaluate(encoder, classifier, tgt_data_valid_loader)
            if f1 > best_f1:
                print("best epoch number: ",epoch)
                print("valid F1: ",f1)
                best_f1 = f1
                best_epoch = epoch
                save_model(args, encoder, param.src_encoder_path+args.tgt+'best_semi'+str(ite))
                save_model(args, classifier, param.src_classifier_path+args.tgt+'best_semi'+str(ite))

    encoder = init_model(args, encoder, restore=param.src_encoder_path+args.tgt+'best_semi'+str(ite))
    classifier = init_model(args, classifier, restore=param.src_classifier_path+args.tgt+'best_semi'+str(ite))
    return encoder, classifier


def pretrain_best_rec_epoch(args, encoder, classifier, data_loader, tgt_data_valid_loader,tgt_data_train_loader):
    """Train classifier for source domain."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=param.c_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    start = datetime.datetime.now()
    best_f1 = 0
    best_epoch = 0
    for epoch in range(args.pre_epochs):
        for step, (reviews, mask,segment, labels) in enumerate(data_loader):
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            segment = make_cuda(segment)
            labels = make_cuda(labels)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source classifier
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % args.pre_log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: cls_loss=%.4f"
                      % (epoch + 1,
                         args.pre_epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
        # evaluate(encoder, classifier, data_loader)
        if epoch < 5:
            f1 = evaluate(encoder, classifier, tgt_data_valid_loader)
            if f1 > best_f1:
                print("best epoch number: ",epoch)
                print("valid F1: ",f1)
                best_f1 = f1
                best_epoch = epoch
                save_model(args, encoder, param.src_encoder_path+'best')
                save_model(args, classifier, param.src_classifier_path+'best')
        
        if args.rec_epoch:
            end = datetime.datetime.now()
            now_time = end-start
            res = evaluate(encoder, classifier, tgt_data_train_loader)
            f = open(args.epoch_path + args.src+args.srcfix+args.tgtfix+'.csv','a+',encoding='utf-8',newline="")
            csv_writer = csv.writer(f)
            row = [epoch+1,res,now_time]
            csv_writer.writerow(row)
            f.close()
    
    f = open(args.epoch_path +'best_epoch_num.csv','a+',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    row = [args.src,args.tgt,args.train_seed,best_epoch]
    csv_writer.writerow(row)
    f.close()

    encoder = init_model(args, encoder, restore=param.src_encoder_path+'best')
    classifier = init_model(args, classifier, restore=param.src_classifier_path+'best')
    return encoder, classifier

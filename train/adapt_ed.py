import torch
from utils import make_cuda
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from transformers import BartTokenizer
import param
from utils import save_model,init_model
import csv
import os


def train(args, encoder, classifier, decoder, src_data_loader, tgt_data_train_loader, tgt_data_valid_loader):
    """Train encoder for target domain."""
    bestf1 = 0.0
    besttrainf1 = 0.0
    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    decoder.train()
    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    CELoss = CrossEntropyLoss()
    optimizer1 = optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr=param.c_learning_rate)
    optimizer2 = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=param.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        for step, ((reviews_src, src_mask,labels), (reviews_tgt, tgt_mask,_)) in data_zip:
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)
            labels = make_cuda(labels)
            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)

            # zero gradients for optimizer
            optimizer1.zero_grad()
            
            # train encoder and classifier
            encoder_output,logit_,pooled_output = encoder(reviews_src, src_mask)
            vocab_size=decoder.config.vocab_size
            preds = classifier(pooled_output)
  
            cls_loss = CELoss(preds, labels)
            loss = args.alpha * cls_loss
            loss.backward()
            optimizer1.step()
            
            if 1:
                optimizer2.zero_grad()
                encoder_outputs,_,_1=encoder(reviews_tgt,tgt_mask)#(input_ids=input_id,attention_mask=inputs_attention_mask)
                decoder_outputs,logits1=decoder(reviews_tgt, encoder_hidden_states=encoder_outputs[0],attention_mask=tgt_mask)                
                hidden_size=decoder.config.hidden_size
                loss1=CELoss(logits1.view(-1,vocab_size),reviews_tgt.view(-1))
    
                loss1.backward()
                optimizer2.step()
            if 1:
                optimizer2.zero_grad()
                encoder_outputs,_,_1=encoder(reviews_src,src_mask)#(input_ids=input_id,attention_mask=inputs_attention_mask)
                decoder_outputs,logits1=decoder(reviews_src, encoder_hidden_states=encoder_outputs[0],attention_mask=src_mask)
                loss1=CELoss(logits1.view(-1,vocab_size),reviews_src.view(-1))
                loss1.backward()
                optimizer2.step()
                
              
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "cls_loss=%.4f,loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         loss.item(),
                         loss1.item()))        
        print("Tgt valid: ")  
        f1_valid = evaluate(encoder, classifier, tgt_data_valid_loader)        
        if f1_valid>bestf1:
            bestf1 = f1_valid
            print("Now best epoch: ",epoch)
            print("best valid F1: ",bestf1)
            print("Tgt train: ")  
            besttrainf1 = evaluate(encoder, classifier, tgt_data_train_loader)

            save_model(args, encoder, param.src_encoder_path+args.tgt+'best')
            save_model(args, classifier, param.src_classifier_path+args.tgt+'best')
            
    f = open(args.out_file+'.csv','a+',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow([args.src,args.tgt,args.train_seed,besttrainf1]) 
    return encoder,classifier,decoder

        
def evaluate(encoder, classifier, data_loader,args=None,flag=None,discriminator=None):
    """Evaluation for ED."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    count = 0
    # evaluate network
    for (reviews, mask, labels) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        labels = make_cuda(labels)

        with torch.no_grad():
            feat,_,pooler_output = encoder(reviews, mask)#,segment)
            preds = classifier(pooler_output)
            if discriminator:
                dom = discriminator(feat)
                print(dom.cpu().numpy().tolist())
                # exit()

        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]

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


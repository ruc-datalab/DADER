#!/usr/bin/env python
# encoding: utf-8

import torch

def cal_coral_loss(source, target):
    batch_size = int(source.size()[0])
    dim = int(source.size()[1])
    source_T = torch.transpose(source,0,1)
    target_T = torch.transpose(target,0,1)
    cov_s = (1/(batch_size-1))*torch.mm(source_T, source)
    cov_t = (1/(batch_size-1))*torch.mm(target_T, target)
    mean_s = torch.mm(torch.ones(1,batch_size).cuda(),source)
    mean_t = torch.mm(torch.ones(1,batch_size).cuda(),target)
    square_mean_s = (1/(batch_size*(batch_size-1)))*torch.mm(torch.transpose(mean_s,0,1),mean_s)
    square_mean_t = (1/(batch_size*(batch_size-1)))*torch.mm(torch.transpose(mean_t,0,1),mean_t)
    cov_s = cov_s - square_mean_s
    cov_t = cov_t - square_mean_t
    #print(cov_s.size())
    coral_loss = 1/(4*dim*dim)*(torch.sum((cov_s-cov_t)**2))
    #print(coral_loss.size())
    return coral_loss
    
    
    

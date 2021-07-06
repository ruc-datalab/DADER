import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import param


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids=None, input_mask=None, segment_ids=None,label_id=None,exm_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id
class InputFeaturesED(object):
    """A single set of features of data for ED."""
    def __init__(self, input_ids, attention_mask,label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

def CSV2Array(path):
    """Read data from csv"""
    data = pd.read_csv(path, encoding='latin')
    pairs, labels = data.pairs.values.tolist(), data.labels.values.tolist()
    return pairs, labels

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    """ restore model weights """
    if restore is not None:
        path = os.path.join(param.model_root, args.src, args.model, str(args.train_seed), restore)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Restore model from: {}".format(os.path.abspath(path)))

    """ check if cuda is available """
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net

def save_model(args, net, name):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.train_seed))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))


def convert_examples_to_features(pairs, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0,csv_writer=None,exp_idx=-1):
    features = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        # add ER situation
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : # remove excessively long string
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("too long!")
                    continue
            tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
        if ex_index == exp_idx:
            """This is for recording attention"""
            with open('tokens.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                writer.writerow(tokens)
            with open('token_type_ids.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                writer.writerow(segment_ids)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids = segment_ids,
                          label_id=label,
                          exm_id=ex_index))
        if csv_writer != None:
            """Record training data for semi"""
            csv_writer.writerow([ex_index, pair, label])
    return features

def get_data_loader(features, batch_size,flag):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_label_ids,all_exm_ids)
    sampler = RandomSampler(dataset)
    if flag == "dev":
        """Read all data"""
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        """Delet the last incomplete epoch"""
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader

def get_data_loaderED(features, batch_size,flag):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,all_label_ids)
    sampler = RandomSampler(dataset)
    if flag == "dev":
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader


def bart_convert_examples_to_features(pairs, labels, max_seq_length, tokenizer, pad_token=0, cls_token='<s>',sep_token='</s>'):
    features = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : #从rtokens中删除多余的部分
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("bad example!")
                    continue
            tokens =  [cls_token] +ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        features.append(InputFeaturesED(input_ids=input_ids,
                        attention_mask=input_mask,
                        label_id=label
                        ))
    return features
  


def MMD(source, target):
    """Compute MMD"""
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss

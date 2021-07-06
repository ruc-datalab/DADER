"""Main script for ED."""
import sys
sys.path.append("..")
import param
from train.pretrain import pretrain,pretrain_best
from train.adapt_ed import train, evaluate
from modules.extractor import BartEncoder
from modules.matcher import BertClassifier
from modules.alignment import BartDecoder
from utils import CSV2Array, bart_convert_examples_to_features, get_data_loaderED, init_model, save_model
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer
import torch
import os
import random
import argparse
import datetime
import csv

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="b2",help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="fz",help="Specify tgt dataset")

    parser.add_argument('--srcfix', type=str, default="",help="Specify src dataset")

    parser.add_argument('--tgtfix', type=str, default="",help="Specify tgt dataset")

    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--adapt', default=False, action='store_true',
                        help='Force to adapt target encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=1000,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bart",
                        choices=["bert"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Specify adversarial weight")

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify KD loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default=20,
                        help="Specify the number of epochs for pretrain")
    
    parser.add_argument('--epoch', type=int, default=0,
                        help="Specify the number of epochs for pretrain")
    
    parser.add_argument('--pre_log_step', type=int, default=10,
                        help="Specify log step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=20,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=10,
                        help="Specify log step size for adaptation")
   
    parser.add_argument('--model_index', type=int, default=3,
                        help="Specify log step size for adaptation")
    parser.add_argument('--out_file', type=str, default="res_best_model",
                        help="Specify log step size for adaptation")
    parser.add_argument('--d_learning_rate', type=float, default=1e-5,
                        help="Specify log step size for adaptation")
    parser.add_argument('--rec_epoch', type=int, default=0,
                        help="Specify log step size for adaptation")
    parser.add_argument('--rec_lr', type=str, default="",
                        help="Specify log step size for adaptation")
    parser.add_argument('--epoch_path', type=str, default='new-all/',
                        help="Specify log step size for adaptation")
    parser.add_argument('--adda', type=int, default=0,
                        help="Specify log step size for adaptation")
    parser.add_argument('--seed_list', type=str, default="",
                        help="Specify log step size for adaptation")
    parser.add_argument('--need_kd_model', type=int, default=0,
                        help="Specify log step size for adaptation")
    parser.add_argument('--need_pred_res', type=int, default=0,
                        help="Specify log step size for adaptation")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_arguments()
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("pre_epochs: " + str(args.pre_epochs))
    print("num_epochs: " + str(args.num_epochs))
    print("AD weight: " + str(args.alpha))
    print("KD weight: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    set_seed(args.train_seed)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # preprocess data
    print("=== Processing datasets ===")
    src_x, src_y = CSV2Array(os.path.join('../data', args.src, args.src + args.srcfix+'.csv'))

    tgt_x, tgt_y = CSV2Array(os.path.join('../data', args.tgt, args.tgt + args.tgtfix+'.csv'))
    tgt_train_x, tgt_valid_x, tgt_train_y, tgt_valid_y = train_test_split(tgt_x, tgt_y,
                                                                        test_size=0.1,
                                                                        stratify=tgt_y,
                                                                        random_state=args.seed)

    src_features = bart_convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
    tgt_train_features = bart_convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
    tgt_valid_features = bart_convert_examples_to_features(tgt_valid_x, tgt_valid_y, args.max_seq_length, tokenizer)

    # load dataset        
    src_data_loader = get_data_loaderED(src_features, args.batch_size,"train")
    tgt_data_train_loader = get_data_loaderED(tgt_train_features, args.batch_size,"train")
    tgt_data_valid_loader = get_data_loaderED(tgt_valid_features, args.batch_size,"dev")
    # load models
    if args.model == 'bart':
        encoder = BartEncoder()
        classifier = BertClassifier()
        decoder = BartDecoder()
        
        encoder = init_model(args, encoder)
        classifier = init_model(args, classifier)
        decoder = init_model(args, decoder)

    encoder, classifier, decoder = train(args, encoder, classifier, decoder,src_data_loader, tgt_data_train_loader, tgt_data_valid_loader) 

    print("=== Result of ED: ===")
    evaluate(encoder, classifier, tgt_data_train_loader)

if __name__ == '__main__':
    main()

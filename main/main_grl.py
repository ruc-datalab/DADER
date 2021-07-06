import sys
sys.path.append("../")
import param
from train.adapt_grl import train, evaluate
from modules.extractor import BertEncoder
from modules.matcher import BertClassifier
from modules.alignment import DomainClassifier
from utils import CSV2Array, convert_examples_to_features, get_data_loader, init_model, save_model
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import os
import random
import argparse
import csv

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="b2",help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="fz",help="Specify tgt dataset")

    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--adapt', default=False, action='store_true',
                        help='Force to adapt target encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=3000,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bert",
                        choices=["bert"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--alpha', type=float, default=0.01,
                        help="gradient reverse weight")

    parser.add_argument('--beta', type=float, default=0.1,
                        help="domain loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--num_epochs', type=int, default=40,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=50,
                        help="Specify log step size for adaptation")

    parser.add_argument('--source_only', type=int, default=0,
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
    print("num_epochs: " + str(args.num_epochs))
    print("reverse weight: " + str(args.alpha))
    print("domain loss weight: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    set_seed(args.train_seed)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # preprocess data
    print("=== Processing datasets ===")

    src_x, src_y = CSV2Array(os.path.join('../data', args.src, args.src + '.csv'))

    tgt_x, tgt_y = CSV2Array(os.path.join('../data', args.tgt, args.tgt + '.csv'))

    tgt_x, tgt_valid_x, tgt_y, tgt_valid_y = train_test_split(tgt_x, tgt_y,
                                                                        test_size=0.1,
                                                                        stratify=tgt_y,
                                                                        random_state=args.seed)

    src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
    tgt_features = convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
    tgt_valid_features = convert_examples_to_features(tgt_valid_x, tgt_valid_y, args.max_seq_length, tokenizer)

    # load dataset

    src_data_loader = get_data_loader(src_features, args.batch_size,"train")
    tgt_data_train_loader = get_data_loader(tgt_features, args.batch_size,"train")
    tgt_data_valid_loader = get_data_loader(tgt_valid_features, args.batch_size,"dev")

    # load models
    if args.model == 'bert':
        src_encoder = BertEncoder()
        src_classifier = BertClassifier()
    dom_classifier = DomainClassifier()
    if args.load:
        src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path)
        src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
        dom_classifier = init_model(args, dom_classifier, restore=param.d_model_path)
    else:
        src_encoder = init_model(args, src_encoder)
        src_classifier = init_model(args, src_classifier)
        dom_classifier = init_model(args, dom_classifier)

    # train source model
    print("=== Training classifier for source domain ===")
    src_encoder, src_classifier,bestf1 = train(args, src_encoder, src_classifier, dom_classifier,src_data_loader, tgt_data_train_loader, tgt_data_valid_loader) 
    print("=== Result of GRL: ===")
    print(bestf1)

if __name__ == '__main__':
    main()

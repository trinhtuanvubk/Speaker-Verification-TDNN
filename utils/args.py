import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=1234)
    # parser.add_argument('--scenario', type=str, default='test_output_model')
    parser.add_argument('--scenario', type=str, default='train')
    parser.add_argument('--load_pretrained', action="store_true")

    # parser.add_argument('--model', type=str, default='SVTR')
    parser.add_argument('--pretrained_path', type=str, default='ckpts/param.model')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts')


    parser.add_argument('--mode', type=str, default='train', help="")
    parser.add_argument('--people_num', type=int, default=5)
    parser.add_argument('--data_per_people', type=int, default=100)
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--logs_path', type=str, default="logs")

    parser.add_argument('--noise', action='store_false')
    parser.add_argument('--mel', action='store_false')
    parser.add_argument('--k', type=float, default=4.5)
    parser.add_argument('--persistent', type=str)
    parser.add_argument('--prefetch_factor', type=str)
    parser.add_argument('--margin', type=float, default=0.6)
    parser.add_argument('--scale', type=float, default=20)
    parser.add_argument('--easy_margin', action='store_true')



    # parser.add_argument('--train_path', type=str, default='./data/own_lmdb/train/')
    # parser.add_argument('--eval_path', type=str, default='./data/own_lmdb/eval/')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_epoch', type=str, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--shuffle', action='store_false')

    args = parser.parse_args()

    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    return args
import torch

import scenario.module as d2l
from utils.tools import get_embedding, eval_net
# from d2l import torch as d2l
from nnet.tdnn.tdnn_pretrain import Pretrain_TDNN


def test_folder(args):
    model = Pretrain_TDNN(args.people_num,
                            1024, 
                            output_embedding=False, 
                            not_grad=False)
    model.load_parameters(f"{args.ckpt_dir}/{args.ckpt_name}", args.device)
    # model = torch.load('net.pth')

    EER, minDCF = eval_net(model, args.device, 10, 10)
    print(f'EER:{EER:.4f} minDCF:{minDCF:.4f}')


def test_two_files(args):
    model = Pretrain_TDNN(args.people_num,
                            1024, 
                            output_embedding=False, 
                            not_grad=False)

    model.load_parameters(f"{args.ckpt_dir}/{args.ckpt_name}", args.device)

    embedding_1 = get_embedding(model, args.filetest_1, args.device)

    embedding_2 = get_embedding(model, args.filetest_2, args.device)

    print(embedding_1.shape)
    cosine_metric = torch.nn.CosineSimilarity(dim=1, eps=1e-10)
    score = cosine_metric(embedding_1, embedding_2)

    score = float(score.detach().cpu())
    print(score)
    return score


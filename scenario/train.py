import os
import torch
import dataloader.dataset as dataset
import dataloader.loader as loader
import scenario.module as module
# from module import torch as module
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from nnet.tdnn.tdnn import ECAPA_TDNN
from nnet.tdnn.tdnn_pretrain import Pretrain_TDNN
from nnet.loss.loss import AAMSoftmax, evaluate_accuracy_gpu

class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.val_loader = loader.get_loader(args)
        self.writer = SummaryWriter(args.logs_path)
        if args.load_pretrained:
            print("finetune+==================")
            model = Pretrain_TDNN(args.people_num,
                                  1024, 
                                  output_embedding=False, 
                                  not_grad=False)
            model.load_parameters(args.pretrained_path, args.device)
        else:
            model = ECAPA_TDNN(in_channels=80, 
                                channels=512, 
                                embd_dim=192,
                                output_num=args.people_num, 
                                context=True, 
                                embedding=False)
        
        self.model = model.to(args.device)
        self.loss = AAMSoftmax(192,
                                args.people_num, 
                                args.margin, 
                                args.scale, 
                                args.easy_margin)

        self.optimizer = torch.optim.Adam(params=(param for param in self.model.parameters()
                                        if param.requires_grad), 
                                        lr=args.learning_rate,
                                        weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                        base_lr=1e-4,
                                        max_lr=0.1,
                                        step_size_up=6250,
                                        mode="triangular2",
                                        cycle_momentum=False)


    def init_logs(self):
        for root, dirs, files in os.walk(self.args.logs_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


    def fit(self):
        timer = module.Timer()
        sum, img = 0, None
        for epoch in range(self.args.num_epoch):
            print(f'\nepoch {epoch + 1}:')
            train_acc = train_l = 0
            metric = module.Accumulator(3)
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                # if i == 0 and epoch == num_epoch - 1:
                #     img = x.to(device)
                timer.start()
                x, y = x.to(self.args.device), y.to(self.args.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                l, prec = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(l * x.shape[0], prec * x.shape[0], x.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                self.scheduler.step()
            sum += metric[2]
            # test_acc = 0
            test_acc = evaluate_accuracy_gpu(self.model, self.val_loader, self.args.device)
            print(f'\tloss {train_l:.3f}, train acc {train_acc:.3f}, '
                f'test acc {test_acc:.3f}')
            self.writer.add_scalar('loss', train_l, epoch)
            self.writer.add_scalars('acc', {'test_acc': test_acc, 'train_acc': train_acc}, epoch)
        print(f'\n{sum / timer.sum():.1f} examples/sec '
            f'on {str(self.args.device)}')
        

        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.args.ckpt_dir}/{self.args.ckpt_name}")
        # write.add_graph(net, img)

def train(args):
    trainer = Trainer(args)
    trainer.fit()
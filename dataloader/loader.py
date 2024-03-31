
from .dataset import MyDataset, load_files

from torch.utils.data import DataLoader

def get_loader(args):

    train_dict, eval_dict, people_num = load_files(mode=args.mode, folder_num=args.people_num,
                                                file_num=args.data_per_people, k=args.k, data_path=args.data_path)
    train_dataset = MyDataset(data_dict=train_dict, people_num=people_num, train=True, mel=args.mel,
                            noise=args.noise)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_worker, pin_memory=True,
                        persistent_workers=args.persistent, prefetch_factor=args.prefetch_factor)
    eval_dataset = MyDataset(data_dict=eval_dict, people_num=people_num, train=False, mel=args.mel,
                                    noise=args.noise)
    # print(iter(eval_dataset))
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_worker, pin_memory=True,
                        persistent_workers=args.persistent, prefetch_factor=args.prefetch_factor)
    return train_loader, eval_loader
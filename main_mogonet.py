""" Example for MOGONET classification
"""
from train_test import train_test
import torch
import random
# random.seed(2)

seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":    
    data_folder = 'huwenjuan'
    view_list = [1, 2, 3, 4]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == 'huwenjuan':
        num_class = 2


    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch)
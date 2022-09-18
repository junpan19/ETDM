from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
import os
import sys
import random
from utils import Logger
import torch.backends.cudnn as cudnn
import argparse
from math import log10
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from data import get_training_set
import pdb
from torch.optim import lr_scheduler
import socket
import time
import cv2
import math
from utils import Logger
import numpy as np
from arch import RRN
import datetime
import torchvision.utils as vutils
import random
from loss import CharbonnierLoss
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


def dist_process(gpu, opt, gpu_num): 

    rank = gpu
    dist.init_process_group(backend = 'nccl')

    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    sys.stdout = Logger(os.path.join(opt.save_train_log, 'train_'+opt.log_name+'.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
       use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed_all(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
        
    pin_memory = True if use_gpu else False

    print(opt)
    print('===> Loading Datasets')
    train_set = get_training_set(opt.data_dir, opt.scale, opt.data_augmentation, opt.file_list)
    datasampler = DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=opt.local_rank)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=False, pin_memory=pin_memory, drop_last=True, sampler=datasampler)
    print('===> DataLoading Finished')
    # Selecting network layer
    n_c = 128
    n_b = opt.layer
    rrn = RRN(opt.scale, n_c, n_b) # initial filter generate network 
    p = sum(p.numel() for p in rrn.parameters())*4/1048576.0
    print('Model Size: {:.2f}M'.format(p))
    print(rrn)
    print('===> {}L model has been initialized'.format(n_b))
    #rrn = torch.nn.DataParallel(rrn, device_ids = [0,1])
    criterion = CharbonnierLoss(1e-3)
    if use_gpu:
        device = torch.device(opt.local_rank)
        #rrn = rrn.cuda(rank)
        rrn.to(device)
        #rrn = rrn.to(f'cuda:{rrn.device_ids[0]}')
        #rrn = DDP(rrn, device_ids=[rank], find_unused_parameters=True)
        rrn = DDP(rrn, device_ids=[opt.local_rank], find_unused_parameters=True)
        criterion.to(device)
        #criterion.cuda()
        #criterion = criterion.to(f'cuda:{rrn.device_ids[0]}')
    optimizer = optim.Adam(rrn.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    if opt.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = opt.stepsize, gamma=opt.gamma)

    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        train(train_loader, rrn, opt.scale, criterion, optimizer, epoch, use_gpu, n_c, device) #fed data into network
        scheduler.step()
        if (epoch) % (opt.snapshots) == 0:
            checkpoint(rrn, epoch)

def train(train_loader, rrn, scale, criterion, optimizer, epoch, use_gpu, n_c, device):
    train_mode = True
    epoch_loss = 0
    rrn.train()
    for iteration, data in enumerate(train_loader):
        x_input, target, ref = data[0], data[1], data[2] # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
        if use_gpu:
            x_input = Variable(x_input).to(device)
            target = Variable(target).to(device)
            ref = Variable(ref).to(device)
        optimizer.zero_grad()
        B, _, T, H, W = x_input.shape
        print(x_input.shape)
        #init_temp = torch.zeros_like(x_input[:,0:1,0,:,:])
        #init_o = init_temp.repeat(1, scale*scale*3,1,1)
        #init_h = init_temp.repeat(1, n_c, 1,1)
        torch.cuda.synchronize()
        t0 = time.time()
        prediction = rrn(x_input, ref)
        loss = criterion(prediction, target)/(B*T)
        #target_vis = target.view(B, -1, H*scale, W*scale)
        #prediction_vis = prediction.view(B, -1, H, W)
        #target_vis = vutils.make_grid(target_vis, normalize=True, scale_each=True)
        #prediction_vis = vutils.make_grid(prediction_vis, normalize=True, scale_each=True)
        #writer.add_image('{}/prediction'.format(opt.log_name), prediction, iteration+1)
        #writer.add_image('{}/target'.format(opt.log_name), target, iteration+1)
        #writer.add_scalar('{}/Loss'.format(opt.log_name), loss.item(), iteration+1)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        epoch_loss += loss.item()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(train_loader), loss.item(), (t1 - t0)))

def checkpoint(rrn, epoch):
    save_model_path = os.path.join(opt.save_model_path, systime)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name  = 'X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    torch.save(rrn.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

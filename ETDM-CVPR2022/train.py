from __future__ import print_function
import argparse
from math import log10
import os
import sys
import cv2
import time
import math
import torch
import random
import argparse
import datetime
import numpy as np
from arch import ETDM
from utils import Logger, CosineAnnealingRestartLR
import torch.optim as optim
from loss import CharbonnierLoss
from data import get_training_set, get_test_set
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import checkpoint, crop_border_Y, crop_border_RGB, calculate_psnr, ssim, calculate_ssim, save_img, bgr2ycbcr

parser = argparse.ArgumentParser(description='PyTorch ETDM')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=90, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots. This is a savepoint, using to save training model.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt', help='where record all of image name in dataset.')
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--test_dir',type=str,default='/home/dataset/validation')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--channel', type=int, default=96, help='network channels')
parser.add_argument('--layer', type=int, default=18, help='network layer')
parser.add_argument('--save_model_path', type=str, default='./{}/{}/weight', help='Location to save checkpoint models')
parser.add_argument('--save_train_log', type=str ,default='./{}/{}/log')
parser.add_argument('--weight_decay', default=5e-04, type=float,help="weight decay (default: 5e-04)")
parser.add_argument('--log_name', type=str, default='training_log')
parser.add_argument('--gpu-devices', default='0,1,2,3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
opt = parser.parse_args()
opt.data_dir_vi = './dataset/vimeo'
opt.save_model_path = './{}/{}/weight'.format(opt.log_name, systime)
opt.save_train_log = './{}/{}/log'.format(opt.log_name, systime)
sys.stdout = Logger(os.path.join(opt.save_train_log, 'train_' + opt.log_name + '.txt'))
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

def set_random_seed():
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    random.seed(opt.seed)

def test(test_loader, model, test_name, epoch):
    train_mode = False
    model.eval()
    count = 0
    PSNR = 0
    SSIM = 0
    PSNR_ = 0
    SSIM_ = 0
    for image_num, data in enumerate(test_loader):
        LR, target, left_mask_HV, left_mask_LV, right_mask_HV, right_mask_LV, L = data[0], data[1], data[2], data[3], data[4], data[5], data[6]
        with torch.no_grad():
            LR = Variable(LR).cuda()
            target = Variable(target).cuda()
            left_mask_HV = Variable(left_mask_HV).cuda()
            left_mask_LV = Variable(left_mask_LV).cuda()
            right_mask_HV = Variable(right_mask_HV).cuda()
            right_mask_LV = Variable(right_mask_LV).cuda()
            B, _, T, _, _ = LR.shape
            t0 = time.time()
    
            prediction = model(LR, 'test', target, left_mask_HV, left_mask_LV, right_mask_HV, right_mask_LV)
            torch.cuda.synchronize()
            t1 = time.time()
            print("===> Timer: %.4f sec." % (t1 - t0))
        count += 1
        prediction = prediction.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
        prediction = prediction.cpu().numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr 

        L = L.numpy()
        L = int(L) # real frame numbers
        target = target.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
        target = target.cpu().numpy()[1:,:,:,::-1] # tensor -> numpy, rgb -> bgr
        target = crop_border_RGB(target, 8)
        prediction = crop_border_RGB(prediction, 8)
        for i in range(L):
            # test_Y______________________
            prediction_Y = bgr2ycbcr(prediction[i])
            target_Y = bgr2ycbcr(target[i])
            prediction_Y = prediction_Y * 255
            target_Y = target_Y * 255
            # calculate PSNR and SSIM
            PSNR += calculate_psnr(prediction_Y, target_Y)
            SSIM += calculate_ssim(prediction_Y, target_Y)
        PSNR_ += PSNR / L
        SSIM_ += SSIM / L
    return PSNR_, SSIM_


def train():
    set_random_seed()
    criterion = CharbonnierLoss(1e-3).cuda()
    model = ETDM(opt.scale, opt.channel, opt.layer)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    p = sum(p.numel() for p in model.parameters())*4/1048576.0
    print('Model Size: {:.2f}M'.format(p))

    normal_params = []
    spynet_params = []
    for name, param in model.named_parameters():
        if 'spynet' in name:
            spynet_params.append(param)
        else:
            normal_params.append(param)


    optim_params_1 = [
        {  # add normal params first
           'params': normal_params,
           'lr': opt.lr
        },]

    optim_params_2 = [
     {
            'params': spynet_params,
            'lr': 2.5e-5
        }, ]

    optimizer_1 = optim.Adam(optim_params_1, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    optimizer_2 = optim.Adam(optim_params_2, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
    period = [opt.nEpochs]
    restart_weights = [1]
    scheduler_1 = CosineAnnealingRestartLR(
        optimizer_1,
        period,
        restart_weights=restart_weights,
        eta_min=1e-7,
    )

    scheduler_2 = CosineAnnealingRestartLR(
        optimizer_2,
        period,
        restart_weights=restart_weights,
        eta_min=1e-7,
    )

    best_PSNR = 0
    for epoch in range(opt.start_epoch, opt.nEpochs+1):

        if epoch == 1:
            for k, v in model.named_parameters():
                if 'spynet' in k:
                    v.requires_grad = False
        else:
            for k, v in model.named_parameters():
                if 'spynet' in k:
                    v.requires_grad = True

        train_set = get_training_set(opt.data_dir_vi, opt.scale, opt.data_augmentation, opt.file_list, 'vimeo')
        train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchsize, shuffle=True, drop_last=True)
       
        for iteration, data in enumerate(train_loader):
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            x_input, target, left_mask_HV, left_mask_LV, right_mask_HV, right_mask_LV = data[0], data[1], data[2], data[3], data[4], data[5] # input and target are both tensor, input:[N,C,T,H,W] 
            x_input = Variable(x_input).cuda()
            left_mask_HV = Variable(left_mask_HV).cuda()
            left_mask_LV = Variable(left_mask_LV).cuda()
            right_mask_HV = Variable(right_mask_HV).cuda()
            right_mask_LV = Variable(right_mask_LV).cuda()
            target = Variable(target).cuda()
            B, _, T, _, _ = target.shape
            t0 = time.time()
            s_t_0, s_t_1, s_t_2, pre_left_0, pre_left_1, pre_right, hr_left_0, hr_left_1, hr_right, G_0, G_1 = model(x_input, 'train', target, left_mask_HV, left_mask_LV, right_mask_HV, right_mask_LV)
            torch.cuda.synchronize()
            t1 = time.time()
            loss_s_t_0 = criterion(s_t_0, G_0)/(B*(T-2))
            loss_s_t_1 = criterion(s_t_1, G_1)/(B*(T-2))
            loss_s_t_2 = criterion(s_t_2, target[:,:,1:-1,:,:])/(B*(T-2))
            loss_left_0 = criterion(pre_left_0, hr_left_0)/(B*(T-2))
            loss_left_1 = criterion(pre_left_1, hr_left_1)/(B*(T-2))
            loss_right = criterion(pre_right, hr_right)/(B*(T-2))
            loss = 0.2*loss_s_t_0 + 0.5*loss_s_t_1 + loss_s_t_2 + 0.5*loss_left_0 + loss_left_1 + loss_right
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()
    
            print("===> Epoch[{}]({}/{}): ".format(epoch, iteration, len(train_loader)))

            print("===> loss_s_t_0: {:.4f} || Timer: {:.4f} sec.".format(loss_s_t_0.item(), (t1 - t0)))

            print("===> loss_s_t_1: {:.4f} || Timer: {:.4f} sec.".format(loss_s_t_1.item(), (t1 - t0)))

            print("===> loss_s_t_2: {:.4f} || Timer: {:.4f} sec.".format(loss_s_t_2.item(), (t1 - t0)))

            print("===> loss_right: {:.4f} || Timer: {:.4f} sec.".format(loss_right.item(), (t1 - t0)))

            print("===> loss_left_0: {:.4f} || Timer: {:.4f} sec.".format(loss_left_0.item(), (t1 - t0)))

            print("===> loss_left_1: {:.4f} || Timer: {:.4f} sec.".format(loss_left_1.item(), (t1 - t0)))
            break

        scheduler_1.step()
        scheduler_2.step()

        f = open(os.path.join(opt.save_train_log,'PSNR.txt'), 'a')
        if (epoch) % (opt.snapshots) == 0:
            checkpoint(model, epoch, systime, opt)
            print('===> Loading test Datasets')
            PSNR_avg = 0
            SSIM_avg = 0
            test_list = ['validation.txt']
            for test_name in test_list:
                opt.file_test_list = test_name
                test_set = get_test_set(opt.test_dir, opt.file_test_list, opt.scale, test_name.split('.')[0])
                test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, drop_last=False)
                print('===> DataLoading Finished')
                PSNR, SSIM = test(test_loader, model, test_name.split('.')[0], epoch)
                PSNR_avg += PSNR
                SSIM_avg += SSIM
            PSNR_avg = PSNR_avg/len(test_list)
            SSIM_avg = SSIM_avg/len(test_list)
            print('==> Average PSNR = {:.6f}'.format(PSNR_avg))
            print('==> Average SSIM = {:.6f}'.format(SSIM_avg))
            f.write(str(epoch)+ ' ' + str(PSNR_avg) + ' ' + str(SSIM_avg) +'\n')
            if PSNR_avg > best_PSNR:
                best_PSNR = PSNR_avg
                checkpoint(model, epoch, systime, opt, True)
if __name__ == '__main__':
    train()

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
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=16, help='number of threads for dat8 loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--image_out', type=str, default='./out/')
parser.add_argument('--test_dir',type=str,default='./dataset/Vid4')
# parser.add_argument('--test_dir',type=str,default='./dataset/UDM10')
#parser.add_argument('--test_dir',type=str,default='./dataset/SPMCS')
parser.add_argument('--channel', type=int, default=96, help='network channels')
parser.add_argument('--layer', type=int, default=18, help='network layer')
parser.add_argument('--save_test_log', type=str ,default='./log')
parser.add_argument('--log_name', type=str, default='testing_log')
parser.add_argument('--save_result', type=bool, default=True)
parser.add_argument('--pretrain', type=str, default='./model/X4_18L_64_best.pth')
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES') 

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
opt = parser.parse_args()
opt.image_out = './{}/{}/{}'.format(opt.save_test_log, 'result', systime)
sys.stdout = Logger(os.path.join(opt.save_test_log, opt.log_name+'.txt'))
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

def set_random_seed():
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    random.seed(opt.seed)

def test():
    model = ETDM(opt.scale, opt.channel, opt.layer) # initialize ETDM
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    p = sum(p.numel() for p in model.parameters())*4/1048576.0
    print('Model Size: {:.2f}M'.format(p))

    if os.path.isfile(opt.pretrain):
        model.load_state_dict(torch.load(opt.pretrain, map_location=lambda storage, loc: storage))
        print('===> Loading pretrain model')
    else:
        raise Exception('pretrain model is not exists')
        
    model.eval()
    # Vid4
    test_list = ['foliage_r.txt','walk_r.txt','city_r.txt','calendar_r.txt']  
    # UDM10
#     test_list = ['archpeople_r.txt','archwall_r.txt','auditorium_r.txt','band_r.txt','caffe_r.txt','camera_r.txt','lake_r.txt','clap_r.txt','photography_r.txt','polyflow_r.txt']
    # SPMCS
    #test_list = ['car05_001.txt','hdclub_003_001.txt','hitachi_isee5_001.txt','hk004_001.txt','HKVTG_004.txt','jvc_009_001.txt','NYVTG_006.txt','PRVTG_012.txt','RMVTG_011.txt','veni3_011.txt','veni5_015.txt']

    t_PSNR_video = 0
    t_SSIM_video = 0 
    for test_name in test_list:
        test_set = get_test_set(opt.test_dir, test_name, opt.scale, test_name.split('.')[0])
        test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, drop_last=False)   
        PSNR = 0
        SSIM = 0
        t_PSNR = 0
        t_SSIM = 0
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
            prediction = prediction.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
            prediction = prediction.cpu().numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr 

            L = L.numpy()
            L = int(L) # real frame numbers
            target = target.squeeze(0).permute(1,2,3,0) # [T,H,W,C]
            target = target.cpu().numpy()[1:,:,:,::-1] # tensor -> numpy, rgb -> bgr
            target = crop_border_RGB(target, 8)
            prediction = crop_border_RGB(prediction, 8)
            for i in range(L):
                if opt.save_result:
                    save_img(opt, prediction[i], test_name.split('.')[0] + '_pre_' + str(i))
                # test_Y______________________
                prediction_Y = bgr2ycbcr(prediction[i])
                target_Y = bgr2ycbcr(target[i])
                prediction_Y = prediction_Y * 255
                target_Y = target_Y * 255
                # calculate PSNR and SSIM
                PSNR = calculate_psnr(prediction_Y, target_Y)
                SSIM = calculate_ssim(prediction_Y, target_Y)
                print('PSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(PSNR, SSIM))
                t_PSNR += PSNR
                t_SSIM += SSIM
            print('===>{} PSNR = {}'.format(test_name, t_PSNR/L))
            print('===>{} SSIM = {}'.format(test_name, t_SSIM/L))
            t_PSNR_video += t_PSNR/L
            t_SSIM_video += t_SSIM/L
    print('==> Average PSNR = {:.6f}'.format(t_PSNR_video/len(test_list)))
    print('==> Average SSIM = {:.6f}'.format(t_SSIM_video/len(test_list)))
    return 0


if __name__ == '__main__':
    test()


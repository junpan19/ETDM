import cv2
import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from Gaussian_downsample import gaussian_downsample as DUF_downsample
from bicubic import imresize
def load_img(image_path, scale, L, image_pad):
    HR = []
    char_len = len(image_path)
    #for img_num in range(L):
    for img_num in range(L):
        index = int(image_path[char_len-7:char_len-4]) + img_num
        image = image_path[0:char_len-7]+'{0:03d}'.format(index)+'.png'
        GT_temp = modcrop(Image.open(image).convert('RGB'), scale)
        HR.append(GT_temp)
    return HR, len(HR)

def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img

class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, file_list, scale, test_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] 
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] # get image path list
        L = os.listdir(os.path.join(image_dir, test_name.split('_')[0]))
        self.L = len(L)
        L.sort()
        self.scale = scale
        self.transform = transform # To_tensor
    def __getitem__(self, index):
        target, L = load_img(self.image_filenames[index], self.scale, self.L, image_pad=True) 
        target = [np.asarray(HR) for HR in target] 
        target = np.asarray(target)
        T,H,W,C = target.shape
        if self.scale == 4:
            target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t = target.shape[0]
        h = target.shape[1]
        w = target.shape[2]
        c = target.shape[3]
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
        target = target.view(c,t,h,w)
        LR = DUF_downsample(target, self.scale) # [c,t,h,w]
        LR = torch.cat((LR[:,1:2,:,:], LR, LR[:,-2:-1,:,:]), dim=1)
        GT = torch.cat((target[:,1:2,:,:], target, target[:,-2:-1,:,:]), dim=1)

        C,T,H,W = LR.shape
        ref = LR.permute(1,2,3,0).numpy() #T,H,W,3
        LR_Y = [cv2.cvtColor(ref[i,:,:,:], cv2.COLOR_RGB2YCR_CB)[:,:,:1] for i in range(T)]
        tensor_left_HV = []
        tensor_left_LV = []
        tensor_right_HV = []
        tensor_right_LV = []
        for i in range(T-2):
            left = LR_Y[i]
            middel = LR_Y[i+1]
            right = LR_Y[i+2]
            left_diff = abs(middel - left)
            eps_left = left_diff.mean()
            mask_left_HV = (left_diff > eps_left) * 1.0

            mask_left_LV = 1 - mask_left_HV
            tensor_left_HV.append(mask_left_HV)
            tensor_left_LV.append(mask_left_LV)

            right_diff = abs(middel - right)
            eps_right = right_diff.mean()
            mask_right_HV = (right_diff > eps_right) * 1.0

            mask_right_LV = 1 - mask_right_HV
            tensor_right_HV.append(mask_right_HV)
            tensor_right_LV.append(mask_right_LV)

        tensor_left_HV = np.asarray(tensor_left_HV).astype(np.float32)
        tensor_left_LV = np.asarray(tensor_left_LV).astype(np.float32)
        tensor_right_HV = np.asarray(tensor_right_HV).astype(np.float32)
        tensor_right_LV = np.asarray(tensor_right_LV).astype(np.float32)

        T,H,W,C = tensor_right_HV.shape

        tensor_left_HV = tensor_left_HV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]
        tensor_left_LV = tensor_left_LV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]
        tensor_right_HV = tensor_right_HV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]
        tensor_right_LV = tensor_right_LV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]

        if self.transform:
            tensor_left_HV = self.transform(tensor_left_HV)
            tensor_left_LV = self.transform(tensor_left_LV)
            tensor_right_HV = self.transform(tensor_right_HV)
            tensor_right_LV = self.transform(tensor_right_LV)

        tensor_left_HV = tensor_left_HV.view(C,T,H,W) # Tensor, [C,T,H,W]
        tensor_left_LV = tensor_left_LV.view(C,T,H,W) # Tensor, [C,T,H,W]
        tensor_right_HV = tensor_right_HV.view(C,T,H,W) # Tensor, [C,T,H,W]
        tensor_right_LV = tensor_right_LV.view(C,T,H,W) # Tensor, [C,T,H,W]
        return LR, GT, tensor_left_HV, tensor_left_LV, tensor_right_HV, tensor_right_LV, L
        
    def __len__(self):
        return len(self.image_filenames) 


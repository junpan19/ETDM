import cv2
import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Gaussian_downsample import gaussian_downsample

def load_img(image_path, scale, data_type):
    HR = []
    if data_type == 'vimeo':
        for img_num in range(7):
            GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            HR.append(GT_temp)
    else:
        for img_num in range(20):
            GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            HR.append(GT_temp)
    return HR

def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img

def train_process(GT, flip_h=True, rot=True, converse=True): 
    if random.random() < 0.5 and flip_h: 
        GT = [ImageOps.flip(HR) for HR in GT]
    if rot:
        if random.random() < 0.5:
            GT = [ImageOps.mirror(HR) for HR in GT]
    return GT

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1], [H,W,C]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1], [H,W]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img = img * 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt = rlt / 255.
    return rlt.astype(in_img_type)

class DataloadFromFolder(data.Dataset): # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, data_type, transform):
        super(DataloadFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] 
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] 
        self.scale = scale
        self.transform = transform # To_tensor
        self.data_augmentation = data_augmentation # flip and rotate
        self.data_type = data_type
    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.scale, self.data_type)
        GT = train_process(GT) # input: list (contain PIL), target: PIL
        GT = [np.asarray(HR) for HR in GT]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        if random.random() < 0.5:
            GT = GT[::-1]
        GT = np.asarray(GT) # numpy, [T,H,W,C]
        T,H,W,C = GT.shape
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0),(2*self.scale,2*self.scale),(2*self.scale,2*self.scale),(0,0)), mode='reflect')
        t, h, w, c = GT.shape
        GT = GT.transpose(1,2,3,0).reshape(h, w, -1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = torch.cat((LR[:,1:2,:,:], LR, LR[:,-2:-1,:,:]), dim=1)
        GT = torch.cat((GT[:,1:2,:,:], GT, GT[:,-2:-1,:,:]), dim=1)

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
        return LR, GT, tensor_left_HV, tensor_left_LV, tensor_right_HV, tensor_right_LV

    def __len__(self):
        return len(self.image_filenames) 


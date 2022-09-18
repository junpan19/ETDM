from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import time
import functools
from spynet_arch import SpyNet
from utils_arch import flow_warp


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_last(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf, scale):
        super(ResidualBlock_noBN_last, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.convh = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_head_o = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_head_dif_left = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_head_dif_right = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_o = nn.Conv2d(nf, scale * scale * 3, 3, 1, 1, bias=True)
        self.conv_dif_left = nn.Conv2d(nf, scale * scale * 3, 3, 1, 1, bias=True)
        self.conv_dif_right = nn.Conv2d(nf, scale * scale * 3, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.convh, self.conv_head_o, self.conv_head_dif_left, self.conv_head_dif_right, self.conv_o, self.conv_dif_left, self.conv_dif_right], 0.1)

    def forward(self, x):
        identity = x
        feat = F.relu(self.conv1(x), inplace=True)
        feat = self.convh(feat)
        feat = identity + feat

        out = F.relu(self.conv_head_o(feat))
        out = self.conv_o(out)
     
        out_dif_left = F.relu(self.conv_head_dif_left(feat))
        out_dif_left = self.conv_dif_left(out_dif_left)

        out_dif_right = F.relu(self.conv_head_dif_right(feat))
        out_dif_right = self.conv_dif_right(out_dif_right)

        return out, out_dif_left, out_dif_right, feat


class neuro_lv(nn.Module):
    def __init__(self, n_c, n_b, scale):
        super(neuro_lv,self).__init__()
        pad = (1,1)
        self.conv_1 = nn.Conv2d(n_c * 2 + 3 * 3, n_c, (3,3), stride=(1,1), padding=pad)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_trunk = make_layer(basic_block, n_b)
        initialize_weights([self.conv_1], 0.1)
    def forward(self, x, left, right, h, h_):
        x = torch.cat((x, left, right, h, h_), dim=1)
        x = F.relu(self.conv_1(x))
        x = self.recon_trunk(x)
        return x

class neuro(nn.Module):
    def __init__(self, n_c, n_b, scale):
        super(neuro,self).__init__()
        pad = (1,1)
        self.neuro_lv = neuro_lv(n_c, 2, scale)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_trunk = make_layer(basic_block, n_b - 2)
        self.res_last = ResidualBlock_noBN_last(n_c, 4) # including 2 layers
    def forward(self, x, dif_left_mask_HV, dif_left_mask_LV, dif_right_mask_HV, dif_right_mask_LV, h, h_):

        lv = self.neuro_lv(x, dif_left_mask_LV, dif_right_mask_LV, h, h_)
        self.neuro_lv.conv_1.dilation=(2,2)
        self.neuro_lv.conv_1.padding=(2,2)
        for layer in self.neuro_lv.children():
            for sublayer1 in layer.children():
                for sublayer2 in sublayer1.children():
                    sublayer2.dilation=(2,2)
                    sublayer2.padding=(2,2)
        hv = self.neuro_lv(x, dif_left_mask_HV, dif_right_mask_HV, h, h_)
        self.neuro_lv.conv_1.dilation=(1,1)
        self.neuro_lv.conv_1.padding=(1,1)
        for layer in self.neuro_lv.children():
            for sublayer1 in layer.children():
                for sublayer2 in sublayer1.children():
                    sublayer2.dilation=(1,1)
                    sublayer2.padding=(1,1)
        h = lv + hv
        h = self.recon_trunk(h)
        x_o, x_o_dif_left, x_o_dif_right, x_h = self.res_last(h)
        return x_o, x_o_dif_left, x_o_dif_right, x_h

class ETDM(nn.Module):
    def __init__(self, scale, n_c, n_b):
        super(ETDM, self).__init__()
        self.neuro = neuro(n_c, n_b, scale)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.n_c = n_c
        self.past_future_num = 3
        self.conv_fusion_1_b = nn.Conv2d(scale*scale*3*self.past_future_num + scale*scale*3 + n_c, n_c, (3,3), stride=(1,1), padding=1)
        self.conv_fusion_1_b_ = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=1)
        basic_block_b = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_temporal_b = make_layer(basic_block_b, 8)
        self.conv_fusion_2_b = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=1)
        self.conv_fusion_3_b = nn.Conv2d(n_c, scale*scale*3, (3,3), stride=(1,1), padding=1)

        self.conv_fusion_1_f = nn.Conv2d(scale*scale*3*self.past_future_num + scale*scale*3 + scale*scale*3 + n_c, n_c, (3,3), stride=(1,1), padding=1)
        self.conv_fusion_1_f_ = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=1)
        basic_block_f = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_temporal_f = make_layer(basic_block_f, 8)
        self.conv_fusion_2_f_s = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=1)
        self.conv_fusion_3_f_s = nn.Conv2d(n_c, scale*scale*3, (3,3), stride=(1,1), padding=1)
        self.conv_fusion_2_f_d = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=1)
        self.conv_fusion_3_f_d = nn.Conv2d(n_c, scale*scale*3, (3,3), stride=(1,1), padding=1)

        initialize_weights([self.conv_fusion_1_b, self.conv_fusion_2_b, self.conv_fusion_1_f, self.conv_fusion_2_f_s, self.conv_fusion_2_f_d, self.conv_fusion_1_f_, self.conv_fusion_1_b_], 0.1)
        initialize_weights([self.conv_fusion_3_b, self.conv_fusion_3_f_s, self.conv_fusion_3_f_d], 0.1)

        self.spynet = SpyNet(load_path='./model/network-sintel-final.pytorch')
        

    def forward(self, x, ref, target, left_mask_HV, left_mask_LV, right_mask_HV, right_mask_LV):
        s_t_0 = []
        s_t_1 = [] 
        s_t_2 = []
        out_0 = []
        out_1 = []
        left = []
        right = []
        GT_0 = []
        GT_1 = []
        h_buf = []
    
        pre_left_total_0 = []
        pre_left_total_1 = []
        hr_left_0 = []
        hr_left_1 = []
        pre_right_total = []
        hr_right = []
        input = []
        B, C, T, H, W = x.shape # T = 22
        forward_lrs = x[:, :, 2:-1, :, :].permute(0,2,1,3,4).reshape(-1, C, H, W)    # n t c h w -> (n t) c h w
        backward_lrs = x[:, :, 1:-2, :, :].permute(0,2,1,3,4).reshape(-1, C, H, W)    # n t c h w -> (n t) c h w
        forward_flow = self.spynet(forward_lrs, backward_lrs).view(B, T-3, 2, H, W).permute(0,2,1,3,4) # n c t h w
        init = True

        # forward
        for i in range(T-2): #exclude the padded two frames.
            warp_forward = []
            warp_backward = []
            f1 = x[:,:,i,:,:]
            f2 = x[:,:,i+1,:,:]
            f3 = x[:,:,i+2,:,:]
            x_left = f2 - f1
            x_right = f2 - f3
            dif_left_mask_HV = f1 * left_mask_HV[:,:,i,:,:]
            dif_left_mask_LV = f1 - dif_left_mask_HV
            dif_right_mask_HV = f3 * right_mask_HV[:,:,i,:,:]
            dif_right_mask_LV = f3 - dif_right_mask_HV

            input.append(f2)
            if init:
                init_temp = torch.zeros_like(x[:,0:1,0,:,:])
                init_h = init_temp.repeat(1, self.n_c, 1, 1)
                pre_0, pre_left, pre_right, h = self.neuro(f2, dif_left_mask_HV, dif_left_mask_LV, dif_right_mask_HV, dif_right_mask_LV, init_h, init_h)
                out_0.append(pre_0)
                s_t_0.append( F.pixel_shuffle(pre_0, self.scale) + \
                          F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                pre_left_total_0.append( F.pixel_shuffle(pre_left, self.scale) + \
                          F.interpolate(x_left, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                right.append(pre_right)
                pre_right_total.append( F.pixel_shuffle(pre_right, self.scale) + \
                          F.interpolate(x_right, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                for i in range(self.past_future_num):
                    warp_forward.append(pre_0)

                warp_forward = torch.cat(warp_forward, dim=1)
                warp_res = torch.cat((warp_forward, pre_0, pre_left, h), dim=1)
                warp_res = F.relu(self.conv_fusion_1_f(warp_res))
                warp_res = self.conv_fusion_1_f_(warp_res)
                warp_res = warp_res + h
                warp_res = self.recon_temporal_f(warp_res)
                h_buf.append(warp_res)

                pre_1 = F.relu(self.conv_fusion_2_f_s(warp_res))
                pre_1 = self.conv_fusion_3_f_s(pre_1) + pre_0
                out_1.append(pre_1)
                s_t_1.append(F.pixel_shuffle(pre_1, self.scale) + \
                          F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False))

                pre_left_ = F.relu(self.conv_fusion_2_f_d(warp_res))
                pre_left = self.conv_fusion_3_f_d(pre_left_) + pre_left
                pre_left_total_1.append( F.pixel_shuffle(pre_left, self.scale) + \
                         F.interpolate(x_left, scale_factor=self.scale, mode='bilinear', align_corners=False) )  
                left.append(pre_left)

                init = False
            else:
                flow = forward_flow[:,:,i-1,:,:]
                h = flow_warp(h, flow.permute(0, 2, 3, 1)) # visualize the warped results.

                pre_0, pre_left, pre_right, h = self.neuro(f2, dif_left_mask_HV, dif_left_mask_LV, dif_right_mask_HV, dif_right_mask_LV, h, h_buf[-1])

                out_0.append(pre_0)
                s_t_0.append( F.pixel_shuffle(pre_0, self.scale) + \
                          F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                pre_left_total_0.append( F.pixel_shuffle(pre_left, self.scale) + \
                          F.interpolate(x_left, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                pre_right_total.append( F.pixel_shuffle(pre_right, self.scale) + \
                          F.interpolate(x_right, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                right.append(pre_right)
                # forward
                accumulate_right_dif = 0
                if i < self.past_future_num:
                    for j in range(1, i+1):
                        accumulate_right_dif = right[-1*j-1] + accumulate_right_dif
                        warp_forward.append(out_0[-1*j-1] - accumulate_right_dif)
                    for k in range(self.past_future_num-i):
                        warp_forward.append(warp_forward[-1])
                else: 
                    for j in range(1, self.past_future_num+1):
                        accumulate_right_dif = right[-1*j-1] + accumulate_right_dif
                        warp_forward.append(out_0[-1*j-1] - accumulate_right_dif) 
              
                warp_forward = torch.cat(warp_forward, dim=1)
                warp_res = torch.cat((warp_forward, pre_0, pre_left, h), dim=1)
                warp_res = F.relu(self.conv_fusion_1_f(warp_res))
                warp_res = self.conv_fusion_1_f_(warp_res)
                warp_res = warp_res + h
                warp_res = self.recon_temporal_f(warp_res)
                h_buf.append(warp_res)

                pre_1 = F.relu(self.conv_fusion_2_f_s(warp_res))
                pre_1 = self.conv_fusion_3_f_s(pre_1) + pre_0
                out_1.append(pre_1) 
                s_t_1.append( F.pixel_shuffle(pre_1, self.scale) + \
                          F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False) )

                pre_left_ = F.relu(self.conv_fusion_2_f_d(warp_res))
                pre_left = self.conv_fusion_3_f_d(pre_left_) + pre_left
                left.append(pre_left)
                pre_left_total_1.append( F.pixel_shuffle(pre_left, self.scale) + \
                          F.interpolate(x_left, scale_factor=self.scale, mode='bilinear', align_corners=False) )
                # backward 
                if i >= self.past_future_num: # i start at 0.
                    accumulate_left_dif = 0
                    for j in range(self.past_future_num, 0, -1):
                        accumulate_left_dif = left[-1*j] + accumulate_left_dif
                        warp_backward.append(out_1[-1*j] - accumulate_left_dif)
                    warp_backward = torch.cat(warp_backward, dim=1)
                    warp_res = torch.cat((warp_backward, out_1[i - self.past_future_num], h_buf[i - self.past_future_num]), dim=1)
                    warp_res = F.relu(self.conv_fusion_1_b(warp_res))
                    warp_res = self.conv_fusion_1_b_(warp_res)
                    warp_res = warp_res + h_buf[i - self.past_future_num]
                    warp_res = self.recon_temporal_b(warp_res)

                    pre_2 = F.relu(self.conv_fusion_2_b(warp_res))
                    pre_2 = self.conv_fusion_3_b(pre_2) + out_1[i - self.past_future_num]
                    s_t_2.append(F.pixel_shuffle(pre_2, self.scale) + \
                          F.interpolate(input[i - self.past_future_num], scale_factor=self.scale, mode='bilinear', align_corners=False) )
                    

            GT_0.append(0.2*target[:,:,i+1,:,:] + 0.8*F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False) )
            GT_1.append(0.5*target[:,:,i+1,:,:] + 0.5*F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False) )
            hr_right.append(0.2*(target[:,:,i+1,:,:] - target[:,:,i+2,:,:]) + 0.8*F.interpolate(x_right, scale_factor=self.scale, mode='bilinear', align_corners=False))
            hr_left_0.append(0.2*(target[:,:,i+1,:,:] - target[:,:,i,:,:]) + 0.8*F.interpolate(x_left, scale_factor=self.scale, mode='bilinear', align_corners=False))
            hr_left_1.append(0.5*(target[:,:,i+1,:,:] - target[:,:,i,:,:]) + 0.5*F.interpolate(x_left, scale_factor=self.scale, mode='bilinear', align_corners=False))
    
        for i in range(self.past_future_num-1, 0, -1): # do not process the last frame due to it is the padded frame.
            warp_backward = []
            left_dif = 0
            for j in range(self.past_future_num-1, i-1, -1): #[-2], [-2,-1]
                left_dif = left[-1*j] + left_dif
                warp_backward.append(out_1[-1*j] - left_dif)
            for k in range(self.past_future_num - len(warp_backward)):
                warp_backward.append(warp_backward[-1])
            warp_backward = torch.cat(warp_backward, dim=1)
            warp_res = torch.cat((warp_backward, out_1[-1*i-1], h_buf[-1*i-1]), dim=1) # pad the frame that is not included in out_1
            warp_res = F.relu(self.conv_fusion_1_b(warp_res))
            warp_res = self.conv_fusion_1_b_(warp_res)
            warp_res = warp_res + h_buf[-1*i-1]
            warp_res = self.recon_temporal_b(warp_res)

            pre_2 = F.relu(self.conv_fusion_2_b(warp_res))
            pre_2 = self.conv_fusion_3_b(pre_2) + out_1[-1*i-1]
            s_t_2.append( F.pixel_shuffle(pre_2, self.scale) + \
                      F.interpolate(input[-1*i-1], scale_factor=self.scale, mode='bilinear', align_corners=False) )

        warp_backward = []
        for i in range(self.past_future_num):
            warp_backward.append(out_1[-1])
        warp_backward = torch.cat(warp_backward, dim=1)
        warp_res = torch.cat((warp_backward, out_1[-1],  h_buf[-1]), dim=1) # pad the frame that is not included in out_1
        warp_res = F.relu(self.conv_fusion_1_b(warp_res))
        warp_res = self.conv_fusion_1_b_(warp_res)
        warp_res = warp_res + h_buf[-1]
        warp_res = self.recon_temporal_b(warp_res)

        pre_2 = F.relu(self.conv_fusion_2_b(warp_res))
        pre_2 = self.conv_fusion_3_b(pre_2) + out_1[-1]
        s_t_2.append( F.pixel_shuffle(pre_2, self.scale) + \
                      F.interpolate(input[-1], scale_factor=self.scale, mode='bilinear', align_corners=False))

        pre_left_total_0 = torch.stack(pre_left_total_0, dim=2)
        pre_left_total_1 = torch.stack(pre_left_total_1, dim=2)
        pre_right_total = torch.stack(pre_right_total, dim=2)
        hr_left_0 = torch.stack(hr_left_0, dim=2)
        hr_left_1 = torch.stack(hr_left_1, dim=2)
        hr_right = torch.stack(hr_right, dim=2)
        s_t_0 = torch.stack(s_t_0, dim=2)
        s_t_1 = torch.stack(s_t_1, dim=2)
        s_t_2 = torch.stack(s_t_2, dim=2)

        GT_0 = torch.stack(GT_0, dim=2)
        GT_1 = torch.stack(GT_1, dim=2)

        if ref == 'train':
            return s_t_0, s_t_1, s_t_2, pre_left_total_0, pre_left_total_1, pre_right_total, hr_left_0, hr_left_1, hr_right, GT_0, GT_1
        else:
            return s_t_2

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

from __future__ import absolute_import
import os
import sys
import math
import errno
import shutil
import json
import os.path as osp
import cv2
import torch
import warnings
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.
    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.
    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.
    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))



def _smooth_l1_loss(input,target,delta):
    # type: (Tensor, Tensor) -> Tensor
    t = torch.abs(input - target)
    return torch.where(t <= delta, 0.5 * t ** 2, delta*t - 0.5*delta**2)

def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = _smooth_l1_loss(input, target)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.smooth_l1_loss(expanded_input, expanded_target, get_enum(reduction))
    return ret

def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(reduction + " is not a valid value for reduction")
    return ret


def legacy_get_string(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> str
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    size_average = torch.jit._unwrap_optional(size_average)
    reduce = torch.jit._unwrap_optional(reduce)
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret

def checkpoint(rrn, epoch, systime, opt, best=False):
    #save_model_path = os.path.join(opt.save_model_path, systime)
    #isExists = os.path.exists(opt.save_model_path)
    #if not isExists:
    try:
        os.makedirs(opt.save_model_path)
    except OSError:
        pass
    if best:
        model_name  = 'X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_best.pth'
    else:
        model_name  = 'X'+str(opt.scale)+'_{}L'.format(opt.layer)+'_{}'.format(opt.patch_size)+'_epoch_{}.pth'.format(epoch)
    #torch.save(rrn, os.path.join(opt.save_model_path, model_name))
    torch.save(rrn.state_dict(), os.path.join(opt.save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(opt.save_model_path, model_name)))

def set_random_seed():
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    random.seed(opt.seed)

def crop_border_Y(prediction, shave_border=0):
    prediction = prediction[shave_border:-shave_border, shave_border:-shave_border]
    return prediction

def crop_border_RGB(target, shave_border=0):
    target = target[:,shave_border:-shave_border, shave_border:-shave_border,:]
    return target

def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def save_img(opt, prediction, test_name):
    if not os.path.exists(opt.image_out):
        os.makedirs(opt.image_out)
    image_dir = os.path.join(opt.image_out, test_name) + '.png'
   
    cv2.imwrite(image_dir, prediction * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

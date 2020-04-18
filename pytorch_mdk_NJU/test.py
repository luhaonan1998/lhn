import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from glob import glob
from PIL import Image
from utilis import torch_msssim, dali
from modules import model
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from gti.param_parser import TrainParser 

#[0-1] float -> [0-31] int
class FloatTo5Bit:
    def __call__(self, x):
        # out = (((x * 255).int() >> 2) + 1) >> 1
        # out = (x * 255).int()
        # return torch.clamp(out.float(), 0, 31)
        return x

def test(args, im_dir):

    TRAINING = False
    CONTEXT = False                              # no context model
    
    # read image
    precise = 16
    print('====> Encoding Image:', im_dir)
    
    img = Image.open(im_dir)
    img = np.array(img)/255.0
    H, W, _ = img.shape
    
    C = 3

    H_PAD = int(16.0 * np.ceil(H / 16.0))
    W_PAD = int(16.0 * np.ceil(W / 16.0))
    im = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im[:H, :W, :] = img[:,:,:3]
    im = torch.FloatTensor(im)

    # model initalization
    image_comp = model.Image_Coder_Context(args)
    # image_comp = nn.DataParallel(image_comp, device_ids=[0,1])
    image_comp = torch.load('/checkpoints/ae.pt')
    
    GPU = True
    if GPU:
        image_comp.cuda()
        #msssim_func = msssim_func.cuda()
        im = im.cuda()

    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H_PAD, W_PAD)

    mssim_func = torch_msssim.MS_SSIM(max_val=1).cuda()

    rec, y_main_q, y_hyper, p_main, p_hyper = image_comp(args, TRAINING, CONTEXT)

    # rate of hyper and main
    bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
    bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
    bpp = bpp_main + bpp_hyper

    output_ = torch.clamp(output, min=0., max=1.0)
    out = output_.data[0].cpu().numpy()
    out = np.round(out * 255.0) 
    out = out.astype('uint8')

    #ms-ssim
    mssim = msssim_func(im.cuda(),output_.cuda())
    
    #psnr float
    mse =  torch.mean((im - output_) * (im - output_))
    psnr = 10. * np.log(1.0/mse.item())/ np.log(10.)
    
    
    print(im_dir, "bpp(main/hyper):%f (%f / %f)"%(bpp, bpp_main, bpp_hyper), "PSNR:", psnr)
    
    #psnr uint8
    mse_i =  torch.mean((im - torch.Tensor([out/255.0]).cuda()) * (im - torch.Tensor([out/255.0]).cuda()))
    psnr_i = 10. * np.log(1.0/mse_i.item())/ np.log(10.)
    # print("bpp: %f PSNR: %f")

if __name__ == "__main__":
    args = TrainParser().parse_args()
    dirs = glob('./kodak/kodim*.png')
    for dir in dirs:
        test(args, dir)
    print("test1")

import torch
import torch.nn as nn
import numpy as np
from modules.factorized_entropy_model import Entropy_bottleneck
from modules.gaussian_entropy_model import Gaussian_Entropy_2D
from modules.basic_module import ResBlock, Non_local_Block, CConv2D
from modules.fast_context_model import Context4
from gti.models.gtinet import GtiNet
from gti.models.encoder_decoder import *

class Enc(GtiNet):
    def __init__(self, args):
        super(Enc, self).__init__()
        # Main encoder 
        self.host_layer0 = make_main_encoder_host0(args)
        self.chip_layer0 = make_main_encoder_chip0(args)  
        self.host_layer1 = make_main_encoder_host1(args)

        # attention
        self.atten = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Hyper encoder 
        self.host_layer2 = make_hyper_encoder_host0(args)
        self.chip_layer1 = make_hyper_encoder_chip0(args)  
        self.host_layer3 = make_hyper_encoder_host1(args)
        
        self._initialize_weights()

    def forward(self, x):
        # Main encoder 
        x = self.host_layer0(x)
        x_ = self.chip_layer0(x)

        x_main = nn.functional.sigmoid(self.atten(x_)) * self.host_layer1(x_)
        #x_main = self.host_layer1(x_)

        # Hyper encoder 
        x = self.host_layer2(x_main)
        x = self.chip_layer1(x)
        x_hyper = self.host_layer3(x)

        return x_main, x_hyper

class Hyper_Dec(GtiNet):
    def __init__(self, args):
        super(Hyper_Dec, self).__init__()
        self.host_layer0 = make_hyper_decoder_host0(args)
        self.chip_layer0 = make_hyper_decoder_chip0(args)
        self.host_layer1 = make_hyper_decoder_host1(args)

        self._initialize_weights()

    def forward(self, x):
        x = self.host_layer0(x)
        x = self.chip_layer0(x)
        x = self.host_layer1(x)
        return x 


class Main_Dec(GtiNet):
    def __init__(self, args):
        super(Main_Dec, self).__init__()
        self.host_layer0 = make_main_decoder_host0(args)
        self.chip_layer0 = make_main_decoder_chip0(args)
        self.host_layer1 = make_main_decoder_host1(args)

        self._initialize_weights()

    def forward(self, x):
        x = self.host_layer0(x)
        x = self.chip_layer0(x)
        x = self.host_layer1(x)
        return x 


class Image_Coder_Context(GtiNet):
    def __init__(self, args):
        super(Image_Coder_Context, self).__init__()
        self.encoder = Enc(args)
        self.factorized_entropy_func = Entropy_bottleneck(256)
        self.hyper_dec = Hyper_Dec(args)
        self.gaussian_entropy_func = Gaussian_Entropy_2D()
        self.context = Context4(256)
        self.decoder = Main_Dec(args)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, if_training, CONTEXT):
        y_main, y_hyper = self.encoder(x)

        if if_training:
            y_main_q = self.add_noise(y_main)
        else:
            y_main_q = torch.round(y_main)
        
        output = self.decoder(y_main_q)

        y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper, if_training) #Training = True
        p_main = self.hyper_dec(y_hyper_q)
        if CONTEXT:
            p_main, _ = self.context(y_main_q, p_main)
            #p_main = self.gaussian_entropy_context(y_main_q, p_main)
        else:
            p_main = self.gaussian_entropy_func(y_main_q, p_main)

        return output, y_main_q, y_hyper_q, p_main, p_hyper

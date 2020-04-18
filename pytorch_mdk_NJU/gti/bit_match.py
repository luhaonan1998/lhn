"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
import os
import logging
import numpy as np
import json
import filecmp
import shutil
from PIL import Image
import torch
import gti.layers
from gti.chip import driver
from gti.param_parser import ConversionParser
from gti.utils import (
    load_checkpoint,
    make_call
)

if __name__ == "__main__":
    _logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
else:
    _logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def load_image(image_path, image_size=224):
    img = Image.open(image_path).resize((image_size, image_size))
    img = np.transpose(np.asarray(img), (2, 0, 1)) #CHW
    return img[:3] #dump alpha channel if present

def clip_image(img):
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = (((img * 255).int() >> 2) + 1) >> 1
    img = torch.clamp(img.float(), 0, 31)
    return img.to("cuda")

class BitMatch():
    def __init__(self, args):
        torch.set_grad_enabled(False)
        self.chip = args.chip
        self.checkpoint = args.checkpoint
        self.use_cpu = args.use_cpu
        self.image_path = args.image_path
        path_prefix = "_".join([args.chip, args.net])
        self.chip_model = os.path.join(args.net_dir, path_prefix + ".model")

        self.save_dir = os.path.join(args.net_dir, "bit_match")
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        self.pytorch_output_dir = os.path.join(self.save_dir, "pytorch")
        self.chip_output_dir = os.path.join(self.save_dir, "chip")
        if not os.path.isdir(self.pytorch_output_dir):
            os.makedirs(self.pytorch_output_dir)
        if not os.path.isdir(self.chip_output_dir):
            os.makedirs(self.chip_output_dir)

        # extracting parameters from json
        self.IMAGE_SIZE = args.image_size
        #input channels not needed

        self.net = load_checkpoint(
             self.chip,
             args.net,
             self.checkpoint,
             self.use_cpu
        )
        if not args.net_config_lst or len(args.net_config_lst) < 1:
            _logger.exception("Network configuration not provided or error!")
        self.net_config = args.net_config_lst[-1]
        self.OUTPUT_CHANNELS = self.net_config['layer'][-1]['output_channels']
        self.OUTPUT_IMAGE_SIZE = self.net_config['layer'][-1]['image_size']
        if 'pooling' in self.net_config['layer'][-1] and \
                self.net_config['layer'][-1]['pooling']:
            self.OUTPUT_IMAGE_SIZE >>= 1
        elif 'upsample_enable' in self.net_config['layer'][-1] and \
                self.net_config['layer'][-1]['upsample_enable']:
            self.OUTPUT_IMAGE_SIZE <<= 1
    
    def get_chip_layers(self):
        chip_layers = []
        prefix = "dump_sublayer"
        for layer in self.net_config['layer']:
            major_layer = str(layer['major_layer'])
            for i in range(layer['sublayer_number']):
                sub_layer = str(i+1)
                chip_layers.append(prefix + major_layer + '-' + sub_layer)
        return chip_layers

    def get_pytorch_layers(self):
        pytorch_layers = []
        pytorch_conv_layers = []
        for major_idx, major in enumerate(self.net.module.chip_layer0):
            for minor_idx, minor in enumerate(major):
                if type(minor) == gti.layers.ConvBlock:
                    pytorch_layers.append(
                        ("{}_{}".format(major_idx+1, minor_idx+1), minor)
                    )
                    pytorch_conv_layers.append(
                        "{}_{}".format(major_idx+1, minor_idx+1)
                    )
                else:
                    pytorch_layers.append(("pool", minor))
        return pytorch_layers, pytorch_conv_layers

    def forward_pytorch(self, img):
        out = clip_image(img)
        for name, layer in self.net.module.named_children():
            if "chip" in name:
                out = layer(out)
        return np.squeeze(out.cpu().numpy(), axis=(0,))

    def forward_chip(self, img):
        img = np.array(img * 255).astype(np.uint8)
        img = np.transpose(img, [1,2,0])
        chip_out = self.gti_model.evaluate(img)
        chip_out = np.reshape(
            chip_out,
            (
             self.OUTPUT_CHANNELS,
             self.OUTPUT_IMAGE_SIZE,
             self.OUTPUT_IMAGE_SIZE
            )
        )
        return chip_out

    def dump_pytorch_activations(self):
        self.pytorch_layers, self.pytorch_conv_layers = self.get_pytorch_layers()
        img = load_image(self.image_path, self.IMAGE_SIZE)
        out = clip_image(img)
        for name, layer in self.pytorch_layers:
            out = layer(out)
            if name is not "pool":
                layer_feature_flatten = out.flatten().cpu().numpy().astype(np.uint8)
                if self.is_fc_mode() and self.chip != '2801': layer_feature_flatten *= 8
                bin_file = os.path.join(self.pytorch_output_dir, name + ".bin")
                layer_feature_flatten.tofile(bin_file)

    def dump_chip_activations(self):
        self.chip_layers = self.get_chip_layers()
        img = load_image(self.image_path, self.IMAGE_SIZE)
        img_bin = np.array(img * 255).astype(np.uint8)
        img_bin_file = os.path.join(self.save_dir, "image_input.bin")
        img_bin.tofile(img_bin_file)
        litedemo = os.path.join(
            os.path.dirname(os.path.realpath(os.path.abspath(__file__))),
            'liteDemo'
            )
        call = make_call()
        call("""
        cd {} && \
        GTI_LOG_LEVEL=9 \
        {} \
        {} \
        {}
        """.format(
                os.path.abspath(self.chip_output_dir),
                litedemo,
                os.path.abspath(self.chip_model),
                os.path.abspath(img_bin_file)
            )
        )

    def match_image(self):
        #check net_config for the learning modes
        for layer in self.net_config['layer']:
            if 'learning' not in layer or not layer['learning']:
                 _logger.exception("Please use the model with 'learning=true' enabled \
                    #for all the layers in order to match layer by layer!")

        self.dump_pytorch_activations()
        self.dump_chip_activations()
        if len(self.pytorch_conv_layers) != len(self.chip_layers):
            _logger.exception("pytorch layers and chip layers not match! Please convert \
                the model with 'learning=true' for all the layers")
        all_match = True
        for i in range(len(self.pytorch_conv_layers)):
            caffefile = os.path.join(
                self.pytorch_output_dir,
                self.pytorch_conv_layers[i] + ".bin"
            )
            chipfile = os.path.join(
                self.chip_output_dir,
                self.chip_layers[i] + ".bin"
            )
            if not filecmp.cmp(caffefile, chipfile):
                all_match = False
                _logger.info(self.pytorch_conv_layers[i] + " does not match chip output!")
        if all_match:
            _logger.info("caffe output matches chip output for all the layers!")

    def match_folder(self):
        #check net_config for the learning modes
        for layer in self.net_config['layer']:
            if 'learning' in layer and layer['learning']:
                 _logger.exception("Please use the model with 'learning=false' for \
                    all the layers in order to run batch testing!")
        self.gti_model = driver.GtiModel(self.chip_model)
        match_count = 0
        image_count = 0
        for image_name in os.listdir(self.image_path):
            image_count += 1
            image_path = os.path.join(self.image_path, image_name)
            img = load_image(image_path, self.IMAGE_SIZE)
            pytorch_out = self.forward_pytorch(img)
            chip_out = self.forward_chip(img)
            bit_diff = (pytorch_out == chip_out)
            if bit_diff.all():
                match_count += 1
                _logger.info("Comparing image %s: %r"%(image_path, bit_diff.all()))
            else:
                output_size = self.OUTPUT_CHANNELS * self.OUTPUT_IMAGE_SIZE * self.OUTPUT_IMAGE_SIZE
                bit_match_ratio = np.sum(bit_diff)/float(output_size)
                _logger.info(
                    "Comparing image %s: %r(%3f match)"  \
                    %(image_path, bit_diff.all(), bit_match_ratio)
                )

        _logger.info("Total Images: {:5d}, Match Count: {:.3f}, Match Ratio: {:.3f}"\
            .format(
                image_count,
                match_count,
                float(match_count)/image_count
             )
        )

    #same as self.OUTPUT_IMAGE_SIZE==7 ?
    def is_fc_mode(self):
        return self.net_config['layer'][-1]['image_size'] == 14 \
               and self.net_config['layer'][-1]['pooling']

def bit_match(args):
    bitMatch = BitMatch(args)
    if os.path.isdir(args.image_path):
        bitMatch.match_folder()
    elif os.path.isfile(args.image_path):
        bitMatch.match_image()
    else:
        _logger.exception("Image path error: either a single image or an image folder!")

if __name__ == "__main__":
    args = ConversionParser().parse_args()
    bit_match(args)

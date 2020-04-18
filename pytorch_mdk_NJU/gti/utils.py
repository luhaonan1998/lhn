'''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

"""Model util functions."""

from collections import namedtuple
import os
import sys
import logging
import subprocess
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from importlib import import_module
from gti.param_parser import QuantizationParams
_logger = logging.getLogger(__name__)

#training related operations
def train_epoch(net, criterion, optimizer, train_loader):
    """run training over one epoch"""
    torch.set_grad_enabled(True)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.to(outputs.device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    train_loss/=(batch_idx+1)
    acc = 100.*float(correct)/float(total)
    if train_loss<1e-3:
        _logger.info('Training loss: %.4e | accuracy: %.3f%% (%d/%d)'
            % (train_loss, acc, correct, total))
    else:
        _logger.info('Training loss: %.4f | accuracy: %.3f%% (%d/%d)'
            % (train_loss, acc, correct, total))
    return acc, train_loss

def val_epoch(net, criterion, val_loader):
    """run evaluation over one epoch"""
    torch.set_grad_enabled(False)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        outputs = net(inputs)
        targets = targets.to(outputs.device)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    test_loss/=(batch_idx+1)
    acc = 100.*float(correct)/float(total)
    _logger.info('Validation loss: %.4f | accuracy: %.3f%% (%d/%d)'
            % (test_loss, acc, correct, total))
    return acc, test_loss

#checkpoint related operations
# TODO: check GPU/CPU compatibility
def load_checkpoint(chip, net, checkpoint, use_cpu=False):
    """Load checkpoint and construct net given chip, net name and checkpoint.
    This function is not used for training, so net is loaded in evaluation
    mode by default.

    Args:
        chip (str): name of GTI chip for checkpoint to be deployed on
        net (str): architecture (vgg16/mobilenet/etc) to evaluate
        checkpoint (str): full file path to checkpoint
        use_cpu (bool): True -> use cpu

    Returns:
        net (subclass of torch.nn.Module): net with forward function
    """
    _logger.info('Loading checkpoint.. %s' % checkpoint)

    device = "cuda" if not use_cpu else "cpu"
    state_dict = torch.load(
        checkpoint,
        map_location=device
    )
    _logger.info("Last validation accuracy: %.3f" % state_dict['best_acc'])
    state_dict = state_dict['model_state_dict']
    arch = get_architecture(net)
    num_classes = arch.get_num_classes(state_dict)
    if num_classes == -1:
        _logger.warning("Checkpoint either does not support classification"
            " or checkpoint is using old variable names.  It is advised to"
            " verify checkpoint before proceeding further.  num_classes is"
            " defaulting to 1000."
        )
        num_classes = 1000
    train_args = wrap_args(
        chip = chip,
        quant_w = True,
        quant_act = True,
        fuse = True,
        num_classes = num_classes
    )
    net = arch(train_args)

    if use_cpu:
        torch.cuda.is_available = return_false
    net = nn.DataParallel(net, [0])  # TODO: if this is OK for CPU
    if not use_cpu:
        net.cuda()
        cudnn.benchmark = True
    check_consistency(net.state_dict().keys(), state_dict.keys())
    net.load_state_dict(state_dict, strict=False)
    # fix batchnorm, dropout and others
    net.eval()
    return net

def save_checkpoint(save_file, epoch, net, optimizer, best_acc):
    """Save checkpoint to a given file path"""
    _logger.info('Saving checkpoint to %s' % save_file)
    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(state, save_file)

def check_consistency(netKeys, ckptKeys):
    netKeys = set(netKeys)
    ckptKeys = set(ckptKeys)
    missing = 0
    extra = 0
    for key in netKeys:
        if key in ckptKeys:
            ckptKeys.discard(key)
        else:
            _logger.warning(key + " is missing!")
            missing+=1
    for key in ckptKeys:
        _logger.warning(key + " is discarded!")
        extra+=1
    if missing>0 or extra>0:
        _logger.warning(
            "missing: {}, extra: {}"
            .format(
                missing,
                extra
            )
        )

def get_step(quant_w, quant_act, fuse):
    """Get which step according to quant_w, quant_act and fuse"""
    if not quant_w and not quant_act and not fuse:
        return 1
    elif quant_w and not quant_act and not fuse:
        return 2
    elif quant_w and quant_act and not fuse:
        return 3
    elif quant_w and quant_act and fuse:
        return 4
    else:
        _logger.warning(
            "Checkpoint not originated from standard GTI MDK training flow! \
            Proceed training with quant_w: %s, quant_act: %s, and \
            fuse: %s" %(quant_w, quant_act, fuse)
        )
        return -1

def get_checkpoint_name(args):
    """Given args (which specifies quantization schemes & chip),
    returns default name to resume from, as well as name to save checkpoint to.
    If currently training on step 1, resume_from is the name specified in args
    """

    prefix = "%s_%s"%(args.chip, args.net)
    step = get_step(args.quant_w, args.quant_act, args.fuse)
    if args.resume_from:
       resume_from = args.resume_from
    else:
        if step in [-1, 1]:
            resume_from = None
        else:
            resume_from = os.path.join(
                args.best_checkpoint_dir,
                prefix + "_step%s.pt"%(str(step-1))
            )
    save_to = prefix + "_step%s.pt"%(str(step))
    return resume_from, save_to

#Misc operations
def get_architecture(arch):
    """Get model architecture based on model name"""
    arch=arch.lower() #arch is str
    if "vgg" in arch:
        module = import_module("gti.models.vgg")
    elif "resnet" in arch:
        module = import_module("gti.models.resnet")
    elif arch in ["mobilenet", "deconv", "tenbit", "gnetfc"]:
        module = import_module("gti.models." + arch)
    else:
        raise NotImplementedError(arch)
    return eval("module."+arch)

def train_step_msg(args):
    """Tag checkpoint to indicate quantization schemes & chip"""
    step = get_step(args.quant_w, args.quant_act, args.fuse)
    if step == 1:
        train_msg = "Step1-training floating-point model"
    elif step == 2:
        train_msg = "Step2-training weight-quantized model"
    elif step == 3:
        train_msg = "Step3-training activation-quantized model"
    elif step == 4:
        train_msg = "Step4-Finetuning fully-quantized model"
    else:
        train_msg = "Step unknown-GTI training flow not followed"
    return train_msg

def wrap_args(chip, quant_w, quant_act, fuse, num_classes=1000):
    """Wrap training-related parameters into wrap"""
    args = QuantizationParams().parse_args([])
    args.chip = chip
    args.quant_w = quant_w
    args.quant_act = quant_act
    args.fuse = fuse
    args.num_classes = num_classes
    return args

def get_sorted_classes(folder_name):
    class_list = [d for d in os.listdir(folder_name)
        if os.path.isdir(os.path.join(folder_name, d))
        and any(os.scandir(os.path.join(folder_name, d)))]
    class_list.sort()
    return class_list

#used for preventing torch from seeing GPUs
def return_false():
    return False

def make_call():
    def call(s):
        if subprocess.call(s, shell=True) != 0:
            _logger.exception("Error occurred during processing!")
    return call

TaskPath = namedtuple("TaskPath", ["dat_json", "model_json", "model"])
def get_conversion_path(net, chip, save_dir="nets"):
    """Helper function to get default paths to files associated with conversion and inference.
    Args:
        net (str): network name
        chip (str): GTI chip series

    Returns:
        task path (namedtuple): paths to associated files
    """
    model_file_name = "_".join([chip, net]) + ".model"
    if net[-4:]=="nobn":
        net = net[:-4]
    path_prefix = "_".join([chip, net])

    return TaskPath(
        dat_json=os.path.join(save_dir, path_prefix + "_dat"),
        model_json=os.path.join(save_dir, path_prefix + "_model.json"),
        model=os.path.join(save_dir, model_file_name)
    )

#does not touch the other vars -> assumes they're already correct
def update_model_json(
        net_config_lst,
        model_json,
        data_files,
        model_json_out,
        dump_mode=False
    ):
    """Update full MODEL JSON with newly generated data file paths:
        dat0, dat1... (chip layers)

    Args:
        net_config_lst (list of dicts): list of dat jsons (as python dicts)
        model_json (str): path of model definition JSON
        data_files (dict str:str): name:file path
        dat_json_out (str): path to write modified model JSON
        dump_mode (bool): if True, chip will dump activations of all minor layers

    Returns:
        None"""

    with open(model_json, "r+") as f:
        model_def = json.load(f)

    #dump all host layers (because SDK 5.0)
    #TODO: clean all the model.jsons?
    tmp = []
    for layer in model_def["layer"]:
        if layer["operation"] in ["GTICNN", "IMAGEREADER"]:
            tmp.append(layer)
    model_def["layer"] = tmp

    count_dat = 0
    for layer in model_def["layer"]:
        if layer["operation"] == "GTICNN":
            layer["data file"] = data_files["dat" + str(count_dat)]
            count_dat += 1
            if layer["device"]["chip"] == "2801" and dump_mode:
                layer["mode"] = 1

    #add SDK 5.0 support
    chip_nums = len(net_config_lst)
    chip_type = net_config_lst[0]['model'][0]['ChipType']

    fullmodel = FullModelConfig7802(model_def) if chip_type == 7802 else FullModelConfig(model_def)

    for idx, net_config in enumerate(net_config_lst):
        model_def = fullmodel.update_fullmodel(idx+1, net_config, chip_nums)

    with open(model_json_out, "w") as outf:
        json.dump(model_def, outf, indent=4, separators=(',', ': '), sort_keys=True)

#does not touch the other vars -> assumes they're already correct
#most of the other vars can be easily read/computed from the checkpoint
#image_size for each layer can be computed (knowing input size), but is annoying
#pooling information is not in the checkpoint
def update_dat_json(dat_json, new_shifts, dat_json_out, dump_mode):
    """Update DAT JSON with newly calculated bit shifts/scaling factors from checkpoint.

    Args:
        dat_json (str): path of DAT definition JSON
        new_shifts (list(int)): list of new shifts
        dat_json_out (str): path to write modified DAT JSON
        dump_mode (bool): if True, chip will dump activations of all minor layers

    Returns:
        net_config (dict): updated dat_json in dict form
    """

    with open(dat_json) as f:
        net_config = json.load(f)
    # add MajorLayerNumber
    net_config['model'][0]['MajorLayerNumber'] = len(net_config['layer'])

    # add major_layer and shift values to net.json
    idx = 0
    for i, layer in enumerate(net_config['layer']):
        layer['major_layer'] = i + 1
        layer['scaling'] = []
        for j in range(layer['sublayer_number']):
            layer['scaling'].append(int(new_shifts[idx]))
            idx += 1

        #change net.json learning mode to do the conversion
        if dump_mode:
            layer['learning'] = True
        else:
            if 'learning' in layer:
                layer['learning'] = False
            #else not present -> using default false

    with open(dat_json_out, 'w') as f:
        json.dump(net_config, f, indent=4, separators=(',', ': '), sort_keys=True)
    return net_config

class FullModelConfig(object):
    def __init__(self, fullmodel_config):
        self.fullmodel_config = fullmodel_config
        if "version" not in self.fullmodel_config:
            self.fullmodel_config['version'] = 100

        self.net_confg = None
        self.layer_idx = 0
        self.chip_type = 0

    def update_fullmodel(self, layer_idx, net_config, chip_nums):
        self.net_config = net_config
        self.layer_idx = layer_idx
        self.chip_nums = chip_nums

        cnn_layer = self.fullmodel_config['layer'][self.layer_idx]
        #add inputs array to cnn layer
        cnn_layer['inputs'] = [{
            "format": "byte",
            "prefilter": "interlace_tile_encode",
            "shape": [
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['input_channels']
            ]
        }]
        #add outputs array to cnn layer, consider learning mode, the implementation vary by chip type
        cnn_layer['outputs'] = []
        DEFAULT_TILE_SIZE = 14
        NUM_ENGINES = 16

        for idx, layer in enumerate(self.net_config['layer']):
            image_size = layer['image_size']
            output_channels = layer['output_channels']
            layer_scaledown = 0
            if self.chip_nums > 1 and self.layer_idx < self.chip_nums:
                layer_scaledown = -3
            upsample_mode = 0
            if 'upsample_enable' in layer and layer['upsample_enable']:
                image_size <<= 1
                output_channels = ((NUM_ENGINES - 1 + output_channels) / NUM_ENGINES) * NUM_ENGINES
                upsample_mode = 1
            output_format = 'byte'
            filter_type = "interlace_tile_decode"
            if 'ten_bits_enable' in layer and layer['ten_bits_enable']:
                output_format = 'float'
                filter_type = 'interlace_tile_10bits_decode'
            tile_size = image_size if image_size < DEFAULT_TILE_SIZE else DEFAULT_TILE_SIZE
            output_size = image_size * image_size * output_channels * 32 // 49
            if 'learning' in layer and layer['learning']:
                # check fake layer
                sublayers = layer['sublayer_number'] + 1 if self.need_fake_layer(layer) else layer['sublayer_number']
                for i in range(sublayers):
                    sub_output_channels = output_channels
                    #handle mobilenet one by one convolution
                    if i == 0 and self.depth_enabled(layer):
                        sub_output_channels = layer['input_channels']
                        sub_output_size = image_size * image_size * sub_output_channels * 32 // 49
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                sub_output_channels,
                                tile_size * tile_size,
                                sub_output_size
                            ],
                            "layer scaledown": layer_scaledown,
                            "upsampling": upsample_mode
                        })
                    else:
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                output_channels,
                                tile_size * tile_size,
                                output_size
                            ],
                            "layer scaledown": layer_scaledown,
                            "upsampling": upsample_mode
                        })
            elif 'last_layer_out' in layer and layer['last_layer_out']:
                cnn_layer["outputs"].append({
                    "format": output_format,
                    "postfilter": filter_type,
                    "shape": [
                        image_size,
                        image_size,
                        output_channels,
                        tile_size * tile_size,
                        output_size
                    ],
                    "layer scaledown": layer_scaledown,
                    "upsampling": upsample_mode
                })
            elif idx + 1 == len(self.net_config['layer']): # add the last layer output
                if 'pooling' in layer and layer['pooling']:
                    image_size >>= 1
                    tile_size = DEFAULT_TILE_SIZE >> 1
                    if image_size == 7: #fc_mode
                        filter_type = "fc77_decode"

                if filter_type == "fc77_decode":
                    output_size = image_size * image_size * output_channels * 64 // 49
                    layer_scaledown = 3
                else:
                    output_size = image_size * image_size * output_channels * 32 // 49
                cnn_layer["outputs"].append({
                    "format": output_format,
                    "postfilter": filter_type,
                    "shape": [
                        image_size,
                        image_size,
                        output_channels,
                        tile_size * tile_size,
                        output_size
                    ],
                    "layer scaledown": layer_scaledown,
                    "upsampling": upsample_mode
                })
        return self.fullmodel_config

    def need_fake_layer(self, layer):
        return 'resnet_shortcut_start_layers' in layer and 'pooling' in layer and layer['pooling'] and \
            layer['sublayer_number'] == layer['resnet_shortcut_start_layers'][-1] + 1

    def depth_enabled(self, layer):
        return 'depth_enable' in layer and layer['depth_enable'] \
            and 'one_coef' in layer and len(layer['one_coef']) > 0 \
            and layer['one_coef'][0] == 0

class FullModelConfig7802(object):
    def __init__(self, fullmodel_config):
        self.fullmodel_config = fullmodel_config
        # Update version, TODO what to do with version, need decide later!
        if "version" not in self.fullmodel_config:
            self.fullmodel_config['version'] = 100

        self.net_confg = None
        self.layer_idx = 0
        self.chip_type = 0

    def updat_fullmodel(self, layer_idx, net_config):
        self.net_config = net_config
        self.layer_idx = layer_idx

        cnn_layer = self.fullmodel_config['layer'][self.layer_idx]
        #add inputs array to cnn layer
        cnn_layer['inputs'] = [{
            "format": "byte",
            "prefilter": "interlace_tile_encode",
            "shape": [
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['input_channels']
            ]
        }]
        #add outputs array to cnn layer, consider learning mode, the implementation vary by chip type
        cnn_layer['outputs'] = []
        DEFAULT_TILE_SIZE = 20
        NUM_ENGINES = 8

        if 'InputImageIOFormat' in self.net_config['model'][0] and self.net_config['model'][0]['InputImageIOFormat'] == 2:
            cnn_layer['inputs'][0]['prefilter'] = None

        filter_type_prefix = 'interlace_tile_'
        if 'OutputCompression' not in self.net_config['model'][0] or not self.net_config['model'][0]['OutputCompression']:
            filter_type_prefix += 'byte_'

        for idx, layer in enumerate(self.net_config['layer']):
            image_size = layer['image_size']
            if 'upsample_enable' in layer and layer['upsample_enable']:
                image_size <<= 1
            tile_size = image_size if image_size < DEFAULT_TILE_SIZE else DEFAULT_TILE_SIZE

            output_format = 'byte'
            filter_type = filter_type_prefix
            output_channels = layer['output_channels']
            if 'ten_bits_enable' in layer and layer['ten_bits_enable']:
                output_format = 'float'
                filter_type += '10bits_'
                output_channels <<= 1
            filter_type += 'decode' #finalize filter type
            if 'byte' in filter_type:
                output_size = image_size * image_size * output_channels
            else:
                output_size = image_size * image_size * output_channels * 64 / 100
            if 'learning' in layer and layer['learning']:
                # check fake layer
                sublayers = layer['sublayer_number'] + 1 if self.need_fake_layer(layer) else layer['sublayer_number']
                for i in range(sublayers):
                    #handle mobilenet one by one convolution
                    if i == 0 and self.depth_enabled(layer):
                        sub_output_channels = layer['input_channels']
                        sub_output_size = image_size * image_size * sub_output_channels * 32 / 49
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                sub_output_channels,
                                tile_size * tile_size,
                                sub_output_size
                            ]
                        })
                    else:
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                output_channels,
                                tile_size * tile_size,
                                output_size
                            ]
                        })
            elif 'last_layer_out' in layer and layer['last_layer_out']:
                cnn_layer["outputs"].append({
                        "format": output_format,
                        "postfilter": filter_type,
                        "shape": [
                            image_size,
                            image_size,
                            output_channels,
                            tile_size * tile_size,
                            output_size
                        ]
                    })
            elif idx + 1 == len(self.net_config['layer']): # add the last layer output
                if 'pooling' in layer and layer['pooling']:
                    image_size >>= 1
                    tile_size = DEFAULT_TILE_SIZE >> 1
                    output_size >>= 2
                rotation_count = 4 if 'InputImageRotation' in self.net_config['model'][0] and self.net_config['model'][0]['InputImageRotation'] else 1
                for i in range(rotation_count):
                    cnn_layer["outputs"].append({
                        "format": output_format,
                        "postfilter": filter_type,
                        "shape": [
                            image_size,
                            image_size,
                            output_channels,
                            tile_size * tile_size,
                            output_size
                        ]
                    })
        return self.fullmodel_config

    def need_fake_layer(self, layer):
        return 'resnet_shortcut_start_layers' in layer and 'pooling' in layer and layer['pooling'] and \
            layer['sublayer_number'] == layer['resnet_shortcut_start_layers'][-1] + 1

    def depth_enabled(self, layer):
        return 'depth_enable' in layer and layer['depth_enable'] \
            and 'one_coef' in layer and len(layer['one_coef']) > 0 \
            and layer['one_coef'][0] == 0



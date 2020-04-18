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

import torch
from torchvision import transforms, datasets

#[0-1] float -> [0-31] int
class FloatTo5Bit:
    def __call__(self, x):
        out = (((x * 255).int() >> 2) + 1) >> 1
        return torch.clamp(out.float(), 0, 31)

class FloatTo8Bit:
    def __call__(self, x):
        out = (x * 255).int()
        return torch.clamp(out.float(), 0, 255)

#if train_data_dir is falsy, return only val_loader
#else, returns both
#(assuming both dirs are valid image directories)
def load_data(train_data_dir, val_data_dir,
            train_batch_size, val_batch_size,
            ten_bits=False, image_size=224):
    quantizer = FloatTo8Bit() if ten_bits else FloatTo5Bit()
    val_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        quantizer
    ])
    val_dataset = datasets.ImageFolder(
        val_data_dir,
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=2
    )
    if not train_data_dir:
        return val_loader

    train_transform = transforms.Compose([
        transforms.Resize(int(image_size*8/7)),
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        quantizer
    ])
    train_dataset = datasets.ImageFolder(
        train_data_dir,
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    return train_loader, val_loader
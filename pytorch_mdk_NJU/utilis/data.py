import time
import logging
import numpy as np
from PIL import Image
from glob import glob

dirs = "../train/*"
dirs = glob(dirs)
print(len(dirs))
#transformer = gluon.data.vision.transforms.RandomResizedCrop(size=(256,256),scale=(0.25,1),ratio=(1.0,1.0))

img_size = 256
dataset_len = 100000
coun = 35000
tic = time.time()
while coun < dataset_len:
    for dir in dirs:
        if coun < dataset_len:
            #read image
            #print(dir)
            img = Image.open(dir)
            h,w = img.size
            scale = np.random.uniform(0.5,1)
            sz = [int(scale * h), int(scale * w)]
            img = img.resize(sz)

            img = np.asarray(img)
            #print(img.shape)
            #random resize & crop
            if img.shape[0] - img_size <=0 or img.shape[1] - img_size <=0:
                break
            x = np.random.randint(0, img.shape[0] - img_size)
            y = np.random.randint(0, img.shape[1] - img_size)
            out = img[x:x + img_size, y:y + img_size, :]
            out.astype("uint8")
            im = Image.fromarray(out)

            #save image
            im.save("../dataset/%d.png"%coun)
            coun += 1
            if coun%1000 == 0:
                print(coun, "spend", time.time()-tic, "s")
                tic = time.time()

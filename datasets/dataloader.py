'''
author: huhq
'''
import os
import torchvision
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
class MyDataset(Dataset):
    def __init__(self, images_dirs, transform=None, shuffle=True):
        if not isinstance(images_dirs, list):
            images_dirs = [images_dirs]

        self.images_dirs = images_dirs

        self.images = []
        for images_dir in images_dirs:
            images = glob.glob(os.path.join(images_dir,'**', '*.jpg'), recursive=True)
            for f in images:
                if 'mask' not in f:
                    self.images.append(f)

        print('Training samples:', len(self.images))
        if shuffle:
            np.random.shuffle(self.images)
        if transform:
            self.tx = transform
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_path = os.path.join(self.images[idx])

        im = Image.open(im_path)
        im1 = self.tx(im.copy())
        im2 = self.tx(im.copy())  # 转化为numpy

        return im1, im2

if __name__ == '__main__':
    import cv2
    import numpy as np
    train_file = r'X:\chest_reconstruction\penu\gram_pos'
    train_dataset = MyDataset(images_dir=train_file, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=2, persistent_workers=True)
    for step, (im1, im2) in enumerate(train_loader):
        if step == 0:            
            print(im1.shape)
            normal_dist = torch.distributions.Normal(0, 0.1)
           
            im = im1[0,...].numpy()
          
            cv2.imwrite('1_3.png', np.uint8(im*255))

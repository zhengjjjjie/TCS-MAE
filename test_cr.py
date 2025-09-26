'''
author: changrj
'''
from models.crnet import CRModel
import torch
from datasets.dataloader import MyDataset
import torchvision.transforms as transforms
from cfgs.config import cfg
from trainers.cr_trainer import Trainer
import argparse
import warnings
import os
import cv2
import numpy as np
def get_args():
    parser = argparse.ArgumentParser(description='Test the CRNet for chest reconstruction')
    parser.add_argument('--encoder', '-encoder', type=str, default='mit_b0', help='The backbone for feature extraction')
    parser.add_argument('--input_size', '-input_size', type=int, default=256, help='the feed size of image')
    parser.add_argument('--data_aug', '-data_aug', type=str, default='', help='the augmentation for dataset')
    parser.add_argument('--load', '-load', type=str, default=r'D:\projects\ChestReconstruction\checkpoints\mit_b0\best_weights.pth', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='test_results/mit_b0_test', help='the path to save weights')
   
    return parser.parse_args()

def random_mask_images_by_patch(images, patch_size=16, mask_ratio=0.75):
        
    N, C, H, W = images.shape
    num_patches = (H // patch_size) * (W // patch_size) # Total number of patches in each image

    # Reshape images to (N, C, num_patches, patch_size, patch_size)
    reshaped_images = images.view(N, C, H // patch_size, patch_size, W // patch_size, patch_size)
    reshaped_images = reshaped_images.permute(0, 2, 4, 1, 3,5).contiguous() # (N, H_patches, W_patches, C, patch_size, patch_size)

    # Generate a random mask for patches
    mask = torch.rand((N, H // patch_size, W // patch_size)) < mask_ratio
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, C, patch_size, patch_size)

    # Apply the mask
    masked_images = reshaped_images.clone()
    masked_images[mask_expanded] = 0

    # Reshape back to original shape
    masked_images = masked_images.permute(0, 3, 1, 4, 2, 5).contiguous()
    masked_images = masked_images.view(N, C, H, W)

    # Also reshape mask back to original image shape for loss calculation
    mask_expanded = mask_expanded.permute(0, 3, 1, 4, 2, 5).contiguous()
    mask_expanded = mask_expanded.view(N, C, H, W)

    return masked_images, mask_expanded[:,0,:,:]

def random_mask_images_by_intensity(images, n_ranges=5, mask_ratio=0.70):
       
    # 将输入的images从形状(N, 3, H, W)转换为(N, H, W, 3)
    images = images.permute(0, 2, 3, 1)
    
    ims_masked = torch.zeros_like(images)
    for i in range(images.shape[0]):
        image = images[i, ...]
        
        # 将图像从RGB转换为灰度
        r,g,b = image[...,0],image[...,1],image[...,2]
        im_gray = 0.2989*r+0.5870*g+0.1140*b
            
        range_width = 1.0 / n_ranges
        mask_ranges = [[i * range_width, (i + 1) * range_width] for i in range(n_ranges)]
        im_to_mask = image.clone()
      
        to_mask_ranges = torch.randint(0, n_ranges, (int(n_ranges * mask_ratio),))
        
        for mask_range_ind in to_mask_ranges:
            mask_low = mask_ranges[mask_range_ind][0]
            mask_high = mask_ranges[mask_range_ind][1]
            
            im_to_mask[(im_gray >= mask_low) & (im_gray < mask_high)] = torch.tensor([0, 0, 0],dtype=torch.float32).to('cuda')
        
        ims_masked[i, ...] = im_to_mask
    
    # 将输出的ims_masked从形状(N, H, W, 3)转换回(N, 3, H, W)
    ims_masked = ims_masked.permute(0, 3, 1, 2)
    return ims_masked

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_args()

    cfg.INPUT_SHAPE = (args.input_size, args.input_size)

    # 读取train数据集
    test_dataset = MyDataset(images_dir=cfg.TEST_DATA_FILE, transform=None, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, persistent_workers=True)

    model = CRModel(encoder_name=args.encoder, num_classes=3)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.load_state_dict(torch.load(args.load, map_location=device))
    
    model.to(device=device)
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    total = 0
    for step, (images1, images2, images3) in enumerate(test_loader):
           
            images1 = images1.unsqueeze(1).to(device)#lung
            images2 = images2.unsqueeze(1).to(device)#gradient
            images3 = images3.unsqueeze(1).to(device)#mediastinum

            inputs = torch.cat((images1, images3, images2), dim=1)
            #inputs,_ = random_mask_images_by_patch(inputs)
            #inputs = random_mask_images_by_intensity(inputs)
            with torch.no_grad():                
                recons, _, _ = model(inputs)              
               
                N, C, H, W = recons.shape
                for n in range(N):
                     recon = recons[n,...].cpu().numpy()
                     input = inputs[n,...].cpu().numpy()
                     recon = recon.transpose((1,2,0))
                     input = input.transpose((1,2,0))
                     recon=np.flip(recon,axis=-1)
                     input=np.flip(input,axis=-1)
                     cv2.imwrite(os.path.join(args.save_dir, str(total+n+1)+'_input.jpg'), np.uint8(input*255))
                     cv2.imwrite(os.path.join(args.save_dir, str(total+n+1)+'_output.jpg'), np.uint8(recon*255))                    
                    
                total+=N
                



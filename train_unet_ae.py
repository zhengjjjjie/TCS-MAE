from models.unet_ae import UnetAE
import torch
from datasets.dataloader import MyDataset
import torchvision.transforms as transforms
from cfgs.config import cfg
from trainers.unet_ae_trainer import Trainer
import argparse
import warnings
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser(description='Train the CRNet for chest reconstruction')
    parser.add_argument('--encoder', '-encoder', type=str, default='resnet50', help='The backbone for feature extraction')
    parser.add_argument('--input_size', '-input_size', type=int, default=256, help='the feed size of image')    
    parser.add_argument('--intensity_mask_size', '-intensity_mask_size', type=int, default=10, help='the masking type, 0:patch, 1:intensity')
    parser.add_argument('--intensity_mask_ratio', '-intensity_mask_ratio', type=float, default=0.7, help='the mask ratio of image for contrastive')
    parser.add_argument('--spatial_mask_size', '-spatial_mask_size', type=int, default=16, help='the masking type, 0:patch, 1:intensity')
    parser.add_argument('--spatial_mask_ratio', '-spatial_mask_ratio', type=float, default=0, help='the mask ratio of image for contrastive')
    parser.add_argument('--load', '-load', type=str, default='weights_best.h5', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='resnet_unet', help='the path to save weights')
    parser.add_argument('--epochs', '-epochs', type=int, default=120, help='Epochs for training')
    parser.add_argument('--steps_per_epoch', '-steps_per_epoch', type=int, default=0, help='iterations for each epoch')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Base learning rate for training')
   
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_args()

    cfg.INPUT_SHAPE = (args.input_size, args.input_size)
    transforms = transforms.Compose([transforms.RandomAffine(degrees=(-20,20), translate=(0.05,0.05), scale=(0.9, 1.1), fill=0),\
                                     transforms.Resize((cfg.INPUT_SHAPE)), transforms.ToTensor()])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # 读取train数据集
    train_dataset = MyDataset(images_dirs=cfg.TRAIN_DATA_FILE, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=10, persistent_workers=True)

    #定义并初始化模型
    project_heads = []
    project_outdims =  [128, 128, 128]
    for outdim in project_outdims:
        project_heads.append(dict(
            pooling='max',
            dropout=None,
            activation=None,
            out_dims= outdim
        ))
    model = UnetAE(encoder_name=args.encoder,  classes=3, activation='sigmoid', aux_params= project_heads)
    if osp.exists(args.load):
        model.load_state_dict(torch.load(args.load, map_location=device))
   
    cfg.INTENSITY_MASK_SIZE =args.intensity_mask_size
    cfg.INTENSITY_MASK_RATIO = args.intensity_mask_ratio
    cfg.SPATIALA_MASK_SIZE = args.spatial_mask_size
    cfg.SPATIALA_MASK_RATIO = args.spatial_mask_ratio
    cfg.STEPS_PER_EPOCH = len(train_dataset)//cfg.BATCH_SIZE if args.steps_per_epoch==0 else args.steps_per_epoch
    cfg.DECAY_STEPS = cfg.STEPS_PER_EPOCH
    cfg.LR = args.lr
    cfg.EPOCHS = args.epochs
    # 初始化训练器
    
    trainer = Trainer(model, cfg, device)
    trainer.start_train(train_loader, args.save_dir, pretrained_file=args.load)
    




'''
input : 1024
output : 2048
'''

# python native
import os

# external library
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# torch
from torchsummary import summary

import torch.nn as nn

import torch.optim as optim
from torchvision import models
import torchvision.transforms.functional as fn
from utils import set_seed, save_model, validation_mo, increment_path, dice_loss

from models import UNet_3Plus_2048_eff_hr as UNet_3Plus

import datetime
import pytz
import torch, gc
import argparse
import wandb
gc.collect()
torch.cuda.empty_cache()

### 저장 폴더명 생성
kst = pytz.timezone('Asia/Seoul')
now = datetime.datetime.now(kst)
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S')

##############-------------------train----------------------
def train(args):
    print(f'Start training..')

    set_seed(args.seed)

    SAVED_DIR = increment_path(os.path.join(args.model_dir, folder_name))
    SAVED_DIR += '_' + args.model
    print(SAVED_DIR)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    train_tf = A.Compose([A.GridDropout(ratio=0.1, random_offset=True, holes_number_x=4, holes_number_y=4, mask_fill_value=0,p=1), A.Rotate(limit=10, p=0.5)])
    valid_tf = None
    train_dataset = XRayDataset_all_2048(is_train=True, transforms=train_tf)
    valid_dataset = XRayDataset_all_2048(is_train=False, transforms=valid_tf)


    wandb.init(project='project3_Mike', entity='cv-06')
    wandb.config.update(args)
    wandb.epochs = args.epochs
    wandb.run.name = folder_name + '_' + args.model

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    ##----------define model-----------------------
    #pre_model = torch.load(args.pre_weight)
    model =  UNet_3Plus.UNet_3Plus_DeepSup()
    '''
    if args.pre_hr !='':
        pre_model = torch.load(args.pre_hr)
        
        model_st = pre_model.state_dict()
        model.load_state_dict(model_st, strict=False)
    '''
    if args.pre_weight !='':
        pre_model = torch.load(args.pre_weight)
        
        model_st = pre_model.state_dict()
        model.load_state_dict(model_st, strict=False)

    model.cuda()
    summary(model, (3, 1024, 1024))
    
    # Loss function 정의
    criterion = nn.BCELoss()#nn.BCEWithLogitsLoss()
    criterion_b = nn.BCELoss()#reduction='mean')
    criterion_i = dice_loss#IOU()

    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(args.epochs):
        model.train()

        for step, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # gpu 연산을 위해 device 할당
            images = fn.resize(images,1024)

            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            outputs =  model(images)
                
            # inference
            if step==0:
                print(images[0].shape)
                print(outputs[0].shape)

            # loss 계산
            loss1_i = criterion_i(outputs[0], masks)#2048 #2048
            loss1_b = criterion_b(outputs[0], masks)
            
            loss2_i = criterion_i(outputs[1], fn.resize(masks,256*4))#1024
            loss2_b = criterion_b(outputs[1], fn.resize(masks,256*4))
            loss3_i = criterion_i(outputs[2], fn.resize(masks,64*4))#256
            loss3_b = criterion_b(outputs[2], fn.resize(masks,64*4))
            loss4_i = criterion_i(outputs[3], fn.resize(masks,16*4))#64
            loss4_b = criterion_b(outputs[3], fn.resize(masks,16*4))
            loss5_i = criterion_i(outputs[4], fn.resize(masks,4*4))#16
            loss5_b = criterion_b(outputs[4], fn.resize(masks,4*4))
            

            optimizer.zero_grad()

            loss = (loss1_i + loss1_b +loss2_i +loss2_b +loss3_i +loss3_b +loss4_i +loss4_b)/5
            loss.backward()
            optimizer.step()
            
            wandb.log({'train_bce_iou_loss': loss})
            
        if not os.path.isdir(SAVED_DIR):
            print("폴더 생성")
            os.makedirs(SAVED_DIR)

        if (epoch) % 1 == 0:
            save_model(model, SAVED_DIR, file_name=f'{epoch}.pt')
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch) % args.interval == 0:
            dice = validation_mo(epoch, model, valid_loader, criterion, args.thr)
            wandb.log({'val_dice': dice[0]})
            wandb.log({'val_TN': dice[1]})
            wandb.log({'val_FP': dice[2]})
            
            if best_dice < dice[0]:
                print(f"Best performance at epoch: {epoch}, {best_dice:.4f} -> {dice[0]:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice[0]
                save_model(model, SAVED_DIR,file_name='best.pt')
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='XRayDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='half_filter', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='eff_hr', help='model type')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='BCEWithLogitsLoss', help='criterion type (default: focal)')
    parser.add_argument('--interval', type=int, default=1, help='batch intervals for validation')
    parser.add_argument('--thr', type=float, default=0.5, help='valid thr')
    parser.add_argument('--pre_weight', type=str, default="/content/drive/MyDrive/input/code/trained_model/2023-06-22-02-19-03_eff_hr/11.pt", help='for transfer learning')
    parser.add_argument('--pre_hr', type=str, default="/content/drive/MyDrive/input/code/trained_model/2023-06-22-02-19-03_high_resol/best.pt", help='back_bone')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/DCM'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './trained_model'))
    args = parser.parse_args()

    print(args)
    train(args)

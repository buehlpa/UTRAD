import argparse
import os
import numpy as np
import time
import datetime
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Create_nets
from datasets import get_dataloader
from utils.dataloader import get_paths_mvtec
from datasets import ImageDataset_mvtec
from torch.utils.data import DataLoader


from configurations.options import TrainOptions
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label
import json



### this script gets all scores from the trainings set with a already trained model

def main():
    args = TrainOptions().parse()
    
    EXPERIMENT_PATH = os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}')

            
    print('step')
    
    if not args.fixed_seed_bool:
        args.test_seed = int(time.time())
        args.seed = int(time.time())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    SAVE_DIR= os.path.join(EXPERIMENT_PATH, args.model_result_dir, 'checkpoint.pth')
    
    
    if os.path.exists(os.path.join(EXPERIMENT_PATH,"experiment_paths.json")) and not args.development:
        with open(os.path.join(EXPERIMENT_PATH,"experiment_paths.json"), "r") as file:
            experiment_paths = json.load(file)
            print("sucessfully loaded experiment paths")  
    
    
    train_paths=experiment_paths['train']
    train_leftout_paths=experiment_paths['leftout_train_paths']
    test_paths=experiment_paths['test']
    
    
    if args.data_set == 'mvtec':

        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        # allpaths
        
        train_path_combined=train_paths+train_leftout_paths
        dataset_train=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = train_path_combined, test_paths = None)
        train_dataloader = DataLoader(dataset_train, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # load experiment dataset pahts
        _, valid_loader ,test_dataloader = get_dataloader(args)
    
    
    else:
        print("Not implemented for other datasets than MVTEC yet")
        sys.exit()
    

    
    start_epoch = 0
    transformer = Create_nets(args)
    transformer = transformer.to(device)
    transformer.cuda()
    optimizer = torch.optim.Adam( transformer.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    best_loss = 1e10

    backbone = models.resnet18(pretrained=True).to(device)
    backbone.eval()
    outputs = []
    
    def hook(module, input, output):
        outputs.append(output)
    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)
    
    
    
    if os.path.exists(SAVE_DIR):
        checkpoint = torch.load(SAVE_DIR)
        transformer.load_state_dict(checkpoint['transformer'])
        start_epoch = checkpoint['start_epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        del checkpoint
        
        
    def embedding_concat(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    flag = 0
    torch.set_printoptions(profile="full")
    
    
    transformer.eval()
    score_map = []
    for i,(filename, batch) in enumerate(train_dataloader):
        with torch.no_grad():
            inputs = batch.to(device)
            outputs = []
            _ = backbone(inputs)
            outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
            recon, std = transformer(outputs)
            
            
            batch_size, channels, width, height = recon.size()
            dist = torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).div(std.abs())
            dist = dist.view(batch_size, 1, width, height)
            patch_normed_score = []
            for j in range(4):
                patch_size = pow(4, j)
                patch_score = F.conv2d(input=dist, 
                    weight=(torch.ones(1,1,patch_size,patch_size) / (patch_size*patch_size)).to(device), 
                    bias=None, stride=patch_size, padding=0, dilation=1)
                patch_score = F.avg_pool2d(dist,patch_size,patch_size)
                patch_score = F.interpolate(patch_score, (width,height), mode='bilinear', align_corners=False)
                patch_normed_score.append(patch_score)
            score = torch.zeros(batch_size,1,64,64).to(device)
            for j in range(4):
                score = embedding_concat(score, patch_normed_score[j])
            
            score = F.conv2d(input=score, 
                    weight=torch.tensor([[[[0.0]],[[0.25]],[[0.25]],[[0.25]],[[0.25]]]]).to(device), 
                    bias=None, stride=1, padding=0, dilation=1)
            score = F.interpolate(score, (inputs.size(2),inputs.size(3)), mode='bilinear', align_corners=False)
            heatmap = score.repeat(1,3,1,1)
            score_map.append(score.cpu())
            
    scores = torch.cat(score_map,dim=0)
    img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
    
    
    with open( os.path.join(EXPERIMENT_PATH,'all_trainscores.pkl'), 'wb') as file:
        pickle.dump(img_scores, file)
        
        
        
if __name__ == '__main__':
    main()
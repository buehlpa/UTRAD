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
from configurations.options import TrainOptions
from utils.dataloader import open_args_log
from datasets import *
from torch.utils.data import DataLoader
import pickle

from torchvision.utils import save_image
from torchvision import models
from PIL import Image
import json

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label

def main():
    args = TrainOptions().parse()
    
    EXPERIMENT_PATH = os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}')

    params=open_args_log(os.path.join(EXPERIMENT_PATH,'args.log'))
    for key in params:
        setattr(args, key, params[key])
    

    print(f'args.seed {args.seed} {args.fixed_seed_bool}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)   

    with open(os.path.join(EXPERIMENT_PATH,'args_cleaned.log') ,"a") as args_log:
        for k, v in sorted(vars(args).items()):
            print('%s: %s ' % (str(k), str(v)))
            args_log.write('%s: %s \n' % (str(k), str(v)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    SAVE_DIR= os.path.join(EXPERIMENT_PATH, args.model_result_dir, 'checkpoint_clean.pth')
    
    start_epoch = 0
    transformer = Create_nets(args)
    transformer = transformer.to(device)
    transformer.cuda()
    optimizer = torch.optim.Adam( transformer.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    best_loss = 1e10

    backbone = models.resnet18(pretrained=True).to(device)
    backbone.eval()
    
    outputs = []
    # hook adds directly to a list calles outputs if a forward pass is done
    def hook(module, input, output):
        outputs.append(output)
    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)
    #backbone.layer4[-1].register_forward_hook(hook)
    layer = 3

    criterion = nn.MSELoss()
    
    
    if os.path.exists(os.path.join(EXPERIMENT_PATH,"experiment_paths.json")) and not args.development:
        with open(os.path.join(EXPERIMENT_PATH,"experiment_paths.json"), "r") as file:
            experiment_paths = json.load(file)
            train_paths=experiment_paths['train']
            clean_train_paths=[path for path in train_paths if 'good' in path]
            
            print("sucessfully loaded experiment paths and added clean train paths")  
    
            experiment_paths['clean_train'] = clean_train_paths

        with open(os.path.join(EXPERIMENT_PATH, "experiment_paths.json"), "w") as file:
            json.dump(experiment_paths, file, indent=4)

        print("Successfully saved the updated experiment paths")   
    else:
        print("Could not load experiment paths")
        sys.exit()
    
    
    
    if args.data_set == 'mvtec':
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        # allpaths
        dataset_clean_train=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = experiment_paths['clean_train'], test_paths = None)
        clean_traindataloader = DataLoader(dataset_clean_train, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # allpaths
        dataset_clean_test=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = experiment_paths['test'], test_paths = None)
        clean_testdataloader = DataLoader(dataset_clean_test, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # load experiment dataset pahts
        _ , _ , test_dataloader = get_dataloader(args)
        
        
        
    if args.data_set == 'mvtec_loco':
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        # allpaths
        dataset_clean_train=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = experiment_paths['clean_train'], test_paths = None)
        clean_traindataloader = DataLoader(dataset_clean_train, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # allpaths
        dataset_clean_test=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = experiment_paths['test'], test_paths = None)
        clean_testdataloader = DataLoader(dataset_clean_test, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # load experiment dataset pahts
        _ , _ , test_dataloader = get_dataloader(args)   
    
    if args.data_set == 'beantec':
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        # allpaths
        dataset_clean_train=ImageDataset_beantec(args,DATA_PATH,mode='train',train_paths = experiment_paths['clean_train'], test_paths = None)
        clean_traindataloader = DataLoader(dataset_clean_train, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # allpaths
        dataset_clean_test=ImageDataset_beantec(args,DATA_PATH,mode='train',train_paths = experiment_paths['test'], test_paths = None)
        clean_testdataloader = DataLoader(dataset_clean_test, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # load experiment dataset pahts
        _ , _ , test_dataloader = get_dataloader(args)   

    if args.data_set == 'visa':
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        # allpaths
        dataset_clean_train=ImageDataset_visa(args,DATA_PATH,mode='train',train_paths = experiment_paths['clean_train'], test_paths = None)
        clean_traindataloader = DataLoader(dataset_clean_train, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # allpaths
        dataset_clean_test=ImageDataset_visa(args,DATA_PATH,mode='train',train_paths = experiment_paths['test'], test_paths = None)
        clean_testdataloader = DataLoader(dataset_clean_test, batch_size=2,shuffle=False,num_workers=8,drop_last=False)

        # load experiment dataset pahts
        _ , _ , test_dataloader = get_dataloader(args)   
    
    else:
        print("Not implemented for other datasets")
        sys.exit()
    
    #train_dataloader, valid_loader ,test_dataloader = get_dataloader(args)

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

    for epoch in range(start_epoch, args.epoch_num):
        avg_loss = 0
        avg_loss_scale = 0
        total = 0
        transformer.train()
        for i,(filename, batch) in enumerate(clean_traindataloader):
            inputs = batch.to(device)
            outputs = []
            optimizer.zero_grad()

            with torch.no_grad():
                _ = backbone(inputs)
                #outputs = outputs[layer-1]
                #outputs = embedding_concat(embedding_concat(embedding_concat(inputs,outputs[0]),outputs[1]),outputs[2])
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
        
            recon, std = transformer(outputs)
            torch.cuda.empty_cache()

            loss = criterion(recon, outputs)
            loss_scale = criterion(std, torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).detach())
            (loss+loss_scale).backward()

            optimizer.step()
            torch.cuda.empty_cache()

            avg_loss += loss * inputs.size(0)
            avg_loss_scale += loss_scale * inputs.size(0)
            total += inputs.size(0)
            print(("\r[Epoch%d/%d]-[Batch%d/%d]-[Loss:%f]-[Loss_scale:%f]" %
                                                            (epoch+1, args.epoch_num,
                                                            i, len(clean_traindataloader),
                                                            avg_loss / total,
                                                            avg_loss_scale / total)))


        if best_loss > avg_loss and best_loss > loss:
            best_loss = avg_loss
            state_dict = {
                        'start_epoch':epoch,
                        #'optimizer':optimizer.state_dict(),
                        'transformer':transformer.state_dict(),
                        'args':args,
                        'best_loss':best_loss
                }
            torch.save(state_dict, SAVE_DIR)
            
        ## TODO add validation with data not from testset , loop 
        
        
        
        print("start evaluation on test set!")
        transformer.eval()
        score_map = []
        gt_list = []
        gt_mask_list = []
        for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader): # this is already the reduced testset
            with torch.no_grad():
                inputs = batch.to(device)
                ground_truth = ground_truth.to(device)
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
                score = F.interpolate(score, (ground_truth.size(2),ground_truth.size(3)), mode='bilinear', align_corners=False)
                heatmap = score.repeat(1,3,1,1)
                score_map.append(score.cpu())
                gt_mask_list.append(ground_truth.cpu())
                gt_list.append(gt)
        
        score_map = torch.cat(score_map,dim=0)
        
        gt_mask_list = torch.cat(gt_mask_list,dim=0)
        gt_list = torch.cat(gt_list,dim=0)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
        gt_list = gt_list.numpy()
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        # calculate per-pixel level ROCAUC
        gt_mask = gt_mask_list.numpy().astype('int')
        scores = scores.numpy().astype('float32')
        fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten()) 
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten()) 
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        with open(os.path.join(EXPERIMENT_PATH,'args_cleaned.log') ,"a") as train_log:
            train_log.write("\r[Epoch%d]-[Loss:%f]-[Loss_scale:%f]-[image_AUC:%f]-[pixel_AUC:%f]" %
                                                        (epoch+1, avg_loss / total, avg_loss_scale / total, img_roc_auc, per_pixel_rocauc))


    ################### after training and saving the clean model - do infrence on test and trainset
    
    
    transformer.eval()
    score_map = []
    for i,(filename, batch) in enumerate(clean_testdataloader):
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
    
    
    with open( os.path.join(EXPERIMENT_PATH,'all_testscores_clean.pkl'), 'wb') as file:
        pickle.dump(img_scores, file)
    
    transformer.eval()
    score_map = []
    for i,(filename, batch) in enumerate(clean_traindataloader):
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
    
    
    with open( os.path.join(EXPERIMENT_PATH,'all_trainscores_clean.pkl'), 'wb') as file:
        pickle.dump(img_scores, file)





if __name__ == '__main__':
    main()
    
    
    
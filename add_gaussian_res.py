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
from torchvision.utils import save_image
from torchvision import models
from PIL import Image

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib





def apply_gaussian_blur(image):
    image_np = image.numpy()
    blurred_image = cv2.GaussianBlur(image_np, (9, 9), 0)
    return torch.from_numpy(blurred_image)



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = TrainOptions().parse()
    
    # CONTROOL BOTH
    if not args.fixed_seed_bool:
        args.test_seed = int(time.time()/2)
        args.seed = int(time.time()/3)
    
    # CONTROL INDICIAUL SEED , TEST_SEED = HOW TO  ADD DATA TO THE TRAINING SET IN CASE OF CONTAMINATION    
    if not args.fixed_seed_TEST_bool:
        args.test_seed = int(time.time()/2)
        
    if not args.fixed_seed_TRAIN_bool:        
         args.seed = int(time.time()/3)

    EXPERIMENT_PATH = os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}')
    SAVE_DIR= os.path.join(EXPERIMENT_PATH, args.model_result_dir, 'checkpoint.pth')


    start_epoch = 0
    transformer = Create_nets(args)
    transformer = transformer.to(device)
    transformer.cuda()
    '''
    if os.path.exists('resnet18_pretrained.pth'):
        backbone = models.resnet18(pretrained=False).to(device)
        backbone.load_state_dict(torch.load('resnet18_pretrained.pth'))
    else:
        backbone = models.resnet18(pretrained=True).to(device)
        '''

    backbone = models.resnet18(pretrained=True).to(device)

    if os.path.exists(SAVE_DIR):
        checkpoint = torch.load(SAVE_DIR)
        transformer.load_state_dict(checkpoint['transformer'])
        start_epoch = checkpoint['start_epoch']

    backbone.eval()
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)
    #backbone.layer4[-1].register_forward_hook(hook)
    layer = 3


    #TODO add paths from smaller testset

    _, _, test_dataloader = get_dataloader(args)


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


        
    score_map = []
    gt_list = []
    gt_mask_list = []
    if True:
        transformer.eval()
        for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader):                       
            with torch.no_grad():
                
                num = 4
                norm_range = [(0.9,1.3),(0.9,1.3),(0.9,1.3),(0.9,1.3),(1.1,1.5),]

                inputs = batch.to(device)
                ground_truth = ground_truth.to(device)
                
                outputs = []
                _ = backbone(inputs)                 ## with the hook funciton in the backbone the feature maps of the backbone are put in the list outputs
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
                    patch_score = F.interpolate(patch_score, (width,height), mode='bilinear')
                    patch_normed_score.append(patch_score)
                    
                score = torch.zeros(batch_size,1,64,64).to(device)
                for j in range(4):
                    score = embedding_concat(score, patch_normed_score[j])
                
                score = F.conv2d(input=score, 
                        weight=torch.tensor([[[[0.0]],[[0.25]],[[0.25]],[[0.25]],[[0.25]]]]).to(device), 
                        bias=None, stride=1, padding=0, dilation=1)

                score = F.interpolate(score, (ground_truth.size(2),ground_truth.size(3)), mode='bilinear')
                
                heatmap = score.repeat(1,3,1,1)
                score_map.append(score.cpu())
                gt_mask_list.append(ground_truth.cpu())
                gt_list.append(gt)
                

    #assert 0

    score_map = torch.cat(score_map,dim=0)
    
    
    #blur extra 
    blurred_images = []
    for i in range(score_map.size(0)):
        image = score_map[i, 0] 
        blurred_image = apply_gaussian_blur(image)
        blurred_images.append(blurred_image.unsqueeze(0))  

    # Stack the list to get a tensor of shape [160, 1, 256, 256]
    score_map = torch.stack(blurred_images)
    
    gt_mask_list = torch.cat(gt_mask_list,dim=0)
    gt_list = torch.cat(gt_list,dim=0)
    
    print(score_map.size())
    
    
    print('dataset: ', args.data_category)
    if True:
        if True:
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

            
            # get optimal threshold
            gt_mask = gt_mask_list.numpy().astype('int')
            scores = scores.numpy().astype('float32')
            '''
            precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            threshold = thresholds[np.argmax(f1)]
            '''
            '''
            from sklearn.utils.multiclass import type_of_target
            print(type_of_target(gt_mask))
            print(type_of_target(scores))
            '''
            # calculate per-pixel level ROCAUC
            fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten()) 
            per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten()) 
            print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
            

            
            
            if args.unalign_test:
                with open(os.path.join(EXPERIMENT_PATH,'Additional_Gausskernel.log') ,"a") as log:
                    log.write('epochs:%ss\n' % (str(start_epoch)))
                    log.write('class_name:%s\n' % (args.data_category))
                    log.write('unalign image ROCAUC: %.3f\n' % (img_roc_auc))
                    log.write('unalign pixel ROCAUC: %.3f\n\n' % (per_pixel_rocauc))
            else:
                with open(os.path.join(EXPERIMENT_PATH,'Additional_Gausskernel.log') ,"w") as log:
                    log.write('epochs:%ss\n' % (str(start_epoch)))
                    log.write('class_name:%s\n' % (args.data_category))
                    log.write('image ROCAUC: %.3f\n' % (img_roc_auc))
                    log.write('pixel ROCAUC: %.3f\n\n' % (per_pixel_rocauc))
            #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (args.data_category, per_pixel_rocauc))

if __name__ == '__main__':
    main()
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

### DEV FOR Refinement strategy
alpha=0.1 # assumed püortion of training data which is conatminated

def weighted_loss(recon, outputs, alpha, epoch):
    individual_losses = F.mse_loss(recon, outputs, reduction='none').mean([2, 3])  # Shape is (batch_size, channels)
    flattened_losses = individual_losses.view(-1)

    quantile = torch.quantile(flattened_losses, alpha)
    lower = flattened_losses[flattened_losses <= quantile]
    upper = flattened_losses[flattened_losses > quantile]

    # exponential weight decay other functions possible here
    b=0.5
    lossweight=np.round(alpha * np.exp(-epoch*b),12)  
    
    ## linaer weight decay
    # a=0.09
    # lossweight = np.maximum(0, np.round((1 - alpha) -  epoch * a, 12))
    
    # weighted loss
    total_loss_exp=(1-lossweight) * lower.mean() +  lossweight * upper.mean()
    return total_loss_exp



def main():
    args = TrainOptions().parse()
    
    EXPERIMENT_PATH = os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}')
        
    with open(os.path.join(EXPERIMENT_PATH,'args.log') ,"a") as args_log:
        for k, v in sorted(vars(args).items()):
            print('%s: %s ' % (str(k), str(v)))
            args_log.write('%s: %s \n' % (str(k), str(v)))
    print('step')
    if args.fixed_seed_bool:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.cuda.manual_seed(int(time.time()))
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    SAVE_DIR= os.path.join(EXPERIMENT_PATH, args.model_result_dir, 'checkpoint.pth')
    
    start_epoch = 0
    transformer = Create_nets(args)
    transformer = transformer.to(device)
    transformer.cuda()
    optimizer = torch.optim.Adam( transformer.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    best_loss = 1e10

    backbone = models.resnet18(pretrained=True).to(device)

    if os.path.exists(SAVE_DIR):
        checkpoint = torch.load(SAVE_DIR)
        transformer.load_state_dict(checkpoint['transformer'])
        start_epoch = checkpoint['start_epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        del checkpoint

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
    
    
    
    train_dataloader, valid_loader ,test_dataloader = get_dataloader(args)

    # get save objs for all losses
    trainsave_ind_loss={} # individual losses
    trainsave_loss_orig={} # original by paper loss
    
    testsave_ind_loss={} # individual losses
    testsave_loss_orig={} # original by paper loss
    testsave_score_after_interpolation={} # score after interpolation
    testsave_score_end={}                 # score after end of network final score
    
    
    for _,(filename, _) in enumerate(train_dataloader):
        for name_ in filename:
            trainsave_ind_loss[name_] = []
            trainsave_loss_orig[name_] = []
            
    for _,(filename, _, _, _) in enumerate(test_dataloader):
        for name_ in filename:
            testsave_ind_loss[name_] = []    
            testsave_loss_orig[name_] = []  
            testsave_score_after_interpolation[name_] = []
            testsave_score_end[name_] = []




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
        for i,(filename, batch) in enumerate(train_dataloader):
            inputs = batch.to(device)
            outputs = []
            optimizer.zero_grad()

            with torch.no_grad():
                _ = backbone(inputs)
                # outputs = outputs[layer-1]
                # outputs = embedding_concat(embedding_concat(embedding_concat(inputs,outputs[0]),outputs[1]),outputs[2])
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
                
            recon, std = transformer(outputs)
            torch.cuda.empty_cache()
            
            ### weighted loss
            #loss = weighted_loss(recon, outputs, alpha, epoch)
            
            
            # TODO Investigate this backpropagation..
            loss = criterion(recon, outputs)
            loss_scale = criterion(std, torch.norm(recon - outputs, p = 2, dim = 1, keepdim = True).detach())   
        
            (loss+loss_scale).backward()    
            
            
            # ## save individual losses
            individual_losses = F.mse_loss(recon, outputs, reduction='none').mean([2, 3]).detach().cpu()  # Shape is (batch_size, channels)                 
            for p in range(len(filename)):
                trainsave_ind_loss[filename[p]].append(individual_losses[p])
            # save original loss
            for l in range(len(filename)):
                ind_loss=criterion(recon[l], outputs[l]).detach().cpu()
                trainsave_loss_orig[filename[l]].append(ind_loss)
            
            
            optimizer.step()
            torch.cuda.empty_cache()
            
            avg_loss += loss * inputs.size(0)
            avg_loss_scale += loss_scale * inputs.size(0)
            total += inputs.size(0)
            print(("\r[Epoch%d/%d]-[Batch%d/%d]-[Loss:%f]-[Loss_scale:%f]" %
                                                            (epoch+1, args.epoch_num,
                                                            i, len(train_dataloader),
                                                            avg_loss / total,
                                                            avg_loss_scale / total)))
            
        ## uncomment for saving
        
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
        # train end per epoch 
        
        ## TODO add validation with data not from testset , loop 
        ## ucomment for eval
        print("start evaluation on test set!")
        transformer.eval()
        
        
        names=[]
        score_map = []
        gt_list = []
        gt_mask_list = []
        for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader):
            with torch.no_grad():
                inputs = batch.to(device)
                ground_truth = ground_truth.to(device)
                outputs = []
                _ = backbone(inputs)
                outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
                recon, std = transformer(outputs)
                
                # SAVE the losses
                individual_losses_test = F.mse_loss(recon, outputs, reduction='none').mean([2, 3]).detach().cpu()  # Shape is (batch_size, channels)
                
                for pi in range(len(name)):
                    testsave_ind_loss[name[pi]].append(individual_losses_test[pi])
                    
                    
                for li in range(len(name)):
                    ind_loss=criterion(recon[li], outputs[li]).detach().cpu()
                    testsave_loss_orig[name[li]].append(ind_loss)
                    
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
                for name_i in name:
                    names.append(name_i)
        # test end per epoch
        
        
        
        score_map = torch.cat(score_map,dim=0)
        gt_mask_list = torch.cat(gt_mask_list,dim=0)
        gt_list = torch.cat(gt_list,dim=0)


        unnormalized_scores = score_map.view(score_map.size(0),-1).max(dim=1)[0]
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # FINAL SCORE
        
        
        # calculate image-level ROC AUC score
        img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
        
        for i in range(len(names)):
                testsave_score_after_interpolation[names[i]].append(unnormalized_scores[i]) 
                testsave_score_end[names[i]].append(img_scores[i]) 
            
            
        
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

        with open(os.path.join(EXPERIMENT_PATH,'args.log') ,"a") as train_log:
            train_log.write("\r[Epoch%d]-[Loss:%f]-[Loss_scale:%f]-[image_AUC:%f]-[pixel_AUC:%f]" %
                                                        (epoch+1, avg_loss / total, avg_loss_scale / total, img_roc_auc, per_pixel_rocauc))
        #epoch end
        
        
    # Saving the dictionary
    torch.save(trainsave_ind_loss, os.path.join(EXPERIMENT_PATH,'trainsave_ind_loss.pth'))
    torch.save(trainsave_loss_orig,  os.path.join(EXPERIMENT_PATH,'trainsave_loss_orig.pth'))    
    torch.save(testsave_ind_loss, os.path.join(EXPERIMENT_PATH,'testsave_ind_loss.pth'))
    torch.save(testsave_loss_orig,  os.path.join(EXPERIMENT_PATH,'testsave_loss_orig.pth')) 
    torch.save(testsave_score_end, os.path.join(EXPERIMENT_PATH,'testsave_score_end.pth'))
    torch.save(testsave_score_after_interpolation, os.path.join(EXPERIMENT_PATH,'testsave_score_after_interpolation.pth'))       
    
        
if __name__ == '__main__':
    main()
import torch
import torch.nn.functional as F
from models import Create_nets
from datasets import get_dataloader
#from options import TrainOptions
from torchvision import models
import os
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import cv2
from utils.results import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils.dataloader import get_paths_mvtec
from datasets import ImageDataset_mvtec
from torch.utils.data import DataLoader
import copy
from configurations.options import TrainOptions


def main():
    args = TrainOptions().parse()

    torch.manual_seed(args.seed)


    normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test = get_paths_mvtec(args,verbose=True)
    DATA_PATH=os.path.join(args.data_root,args.data_category)
    # combine good and anomalies
    train_data=normal_images+sampled_anomalies_for_train
    labels_train=[0]*len(normal_images)+[1]*len(sampled_anomalies_for_train)


    np.random.seed(args.seed)
    N_samples = len(train_data)
    print(N_samples)
    idx = np.arange(N_samples)
    np.random.shuffle(idx)

    stride =45
    window_size = 120

    # Create train sets
    shifts = np.floor(N_samples / stride)
    train_sets = int(shifts)
    print('train_sets: ', train_sets)
    print('window_size: ', window_size)

    train_ind_ls = []
    for i in range(train_sets):
        train_ind_ls.append(idx[np.arange(i * stride, i * stride + window_size) % N_samples])

    all_data_paths=[train_data[id]for id in idx]
    all_data_set=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = all_data_paths,test_paths = None)
    all_dataloader = DataLoader(all_data_set,batch_size=args.batch_size,shuffle=False,num_workers=args.n_cpu,drop_last=False)
    recon_error_ls = []

    ## Preset model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet18(pretrained=True).to(device)
    backbone.eval()
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)
    criterion = nn.MSELoss()

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


    recon_error_ls = []



    for j in range(train_sets):
        
        model_result_m=args.model_result_dir+'_j_'+str(j)
        
        EXPERIMENT_PATH = os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}')
        SAVE_DIR= os.path.join(EXPERIMENT_PATH, model_result_m, 'checkpoint.pth')
        
        if not os.path.exists(os.path.join(EXPERIMENT_PATH, model_result_m)):
            os.makedirs(os.path.join(EXPERIMENT_PATH, model_result_m))

        paths_j = [train_data[idx] for idx in train_ind_ls[j]]
        dataset_j=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = paths_j, test_paths = None)
        train_j_dataloader = DataLoader(dataset_j, batch_size=2,shuffle=False,num_workers=8,drop_last=False)
        
        #train_dataloader, valid_loader ,test_dataloader = get_dataloader(args)

        ## Train model
        
        # #model....
        start_epoch = 0
        transformer = Create_nets(args)
        transformer = transformer.to(device)
        transformer.cuda()
        optimizer = torch.optim.Adam( transformer.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        best_loss = 1e10
        
        #### TRAIN MODEL
        for epoch in range(start_epoch, args.epoch_num):
            avg_loss = 0
            avg_loss_scale = 0
            total = 0
            transformer.train()
            for i,(filename, batch) in enumerate(train_j_dataloader):
                
                inputs = batch.to(device)
                outputs = []
                optimizer.zero_grad()
                with torch.no_grad():
                    _ = backbone(inputs)
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
                                                                i, len(train_j_dataloader),
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
                
        # EVALUATE ON TRAINING SET
        print("start evaluation on training set!")
        transformer.eval()
        score_map = []
        for i,(filename, batch) in enumerate(all_dataloader):
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
        
        # max_score = scores.max()
        # min_score = scores.min()
        # scores = (scores - min_score) / (max_score - min_score)
        
        img_scores = scores.view(scores.size(0),-1).max(dim=1)[0]
        recon_error_ls.append(img_scores)
            

    scores = np.vstack(recon_error_ls)

    for i in range(train_sets):
        scores[i, :] = (scores[i, :] - scores[i, train_ind_ls[i]].mean()) / scores[i, train_ind_ls[i]].std()
        
    # Create boolean matrix for train and test points
    scores_bool = np.zeros_like(scores, dtype=bool)

    for i in range(train_sets):
        scores_bool[i, train_ind_ls[i]] = True
        
        
    scores_infer = scores.copy()
    scores_train = scores.copy()

    scores_infer[scores_bool] = np.nan
    scores_train[~scores_bool] = np.nan


    # Check how much point is varying in train and test
    scores_test_mean = np.nanmean(scores_infer,axis = 0)
    scores_test_std = np.nanstd(scores_infer,axis = 0)
    scores_train_mean = np.nanmean(scores_train,axis = 0)
    scores_train_std = np.nanstd(scores_train,axis = 0)


    indicator = np.abs(scores_test_mean - scores_train_mean)


    resdict={'indicator':indicator,'idx':idx,'scores_test_mean':scores_test_mean,'scores_test_std':scores_test_std,'scores_train_mean':scores_train_mean,'scores_train_std':scores_train_std}
    pd.DataFrame(resdict).to_pickle(os.path.join(EXPERIMENT_PATH,f'USDR_window:{window_size}_stride:{stride}.pkl'))
    
if __name__ == '__main__':
    main()
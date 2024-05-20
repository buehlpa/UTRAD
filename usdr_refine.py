
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
import re
import pandas as pd
import pickle
from configurations.options import TrainOptions
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.measure import label


def get_sorted_indexes(lst):
    indexed_lst = list(enumerate(lst))
    sorted_indexed_lst = sorted(indexed_lst, key=lambda x: x[1])
    sorted_indexes = [index for index, value in sorted_indexed_lst]
    return sorted_indexes

def get_matching_files(EXPERIMENT_PATH):
    pattern = re.compile(r'^USDR.*\.pkl$')
    files = os.listdir(EXPERIMENT_PATH)
    matching_files = [f for f in files if pattern.match(f)]
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No files matching the pattern found in {EXPERIMENT_PATH}")
    if len(matching_files) > 1:
        raise ValueError(f"Multiple files matching the pattern found in {EXPERIMENT_PATH}")
    else:
        return os.path.join(EXPERIMENT_PATH, matching_files[0])


def main():
    args = TrainOptions().parse()
    

    
    EXPERIMENT_PATH = os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}')
        
    with open(os.path.join(EXPERIMENT_PATH,'args.log') ,"a") as args_log:
        for k, v in sorted(vars(args).items()):
            print('%s: %s ' % (str(k), str(v)))
            args_log.write('%s: %s \n' % (str(k), str(v)))
            
    print('step')
    
    # TODO read args and set random seed etc
    
    
    
    if args.fixed_seed_bool:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.cuda.manual_seed(int(time.time()))
    



    usdr_df_path=get_matching_files(EXPERIMENT_PATH)
    usdr_df=pd.read_pickle(usdr_df_path)

    # TODO load saved idx and train model withz refined paths





    if args.data_set == 'mvtec':
        
        train_data=normal_images+sampled_anomalies_for_train
        labels_train=[0]*len(normal_images)+[1]*len(sampled_anomalies_for_train)
        
        
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        print(img_scores.tolist())
        # cut off the images with the highest reconstruction error
        refined_idx=get_sorted_indexes(img_scores.tolist())[:int(len(img_scores.tolist())*(1-args.assumed_contamination_rate))]
        paths_j = [train_data[idx] for idx in refined_idx]
        
        with open( os.path.join(EXPERIMENT_PATH,'trainpaths_refined.pkl'), 'wb') as file:
            pickle.dump(paths_j, file)
        
        print(paths_j)
        
        dataset_refined_simple=ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = paths_j, test_paths = None)
        train_refined_simple = DataLoader(dataset_refined_simple, batch_size=2,shuffle=True,num_workers=8,drop_last=False)
    
    else:
        print("Not implemented for other datasets than MVTEC yet")
        sys.exit()
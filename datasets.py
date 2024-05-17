import glob
import random
import torch
import math
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils.dataloader import get_paths_mvtec , get_paths_mvtec_loco , get_paths_beantec



_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
#*2

# TODO loader for, visa 

########### mvtec and mvtc loco
class ImageDataset_mvtec(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train', train_paths=None, test_paths=None):
        
        self.img_size = 280 * args.factor
        self.crop_size = 256 * args.factor
        self.args = args
        self.mode = mode
        if train_paths is None and test_paths is None:
            raise ValueError("either test or train paths must be provided depending on the mode")
        
        self.train_paths = train_paths
        self.test_paths = test_paths
        
        self.transform_train = transforms.Compose([ transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                                transforms.Pad(int(self.crop_size/10),fill=0,padding_mode='constant'),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225 ]) ])
        
        self.resize_transform_loco = transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC)

        if mode == 'train':
            self.files = train_paths
        elif mode == 'test':
            self.files = test_paths
            
    def _align_transform(self, img, mask):
        #resize to 224
        img = TF.resize(img, self.crop_size, Image.BICUBIC)
        mask = TF.resize(mask, self.crop_size, Image.NEAREST)
        #toTensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        #normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
        return img, mask
    
    def _unalign_transform(self, img, mask):
        #resize to 256
        img = TF.resize(img, self.img_size, Image.BICUBIC)
        mask = TF.resize(mask, self.img_size, Image.NEAREST)
        #random rotation
        angle = transforms.RandomRotation.get_params([-10, 10])
        img = TF.rotate(img, angle, fill=(0,))
        mask = TF.rotate(mask, angle, fill=(0,))
        #random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        #toTensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        #normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
        return img, mask

    def __getitem__(self, index):
        filename = self.files[index]
        img = Image.open(filename)
        img = img.convert('RGB')
        

        if self.args.mode=='mvtec_loco':
            img = self.resize_transform_loco(img)       

        
        if self.mode == 'train':
            img = self.transform_train(img)            
            return filename, img
        
        elif self.mode == 'test':
            transform_test = self._unalign_transform if self.args.unalign_test else self._align_transform
            img_size = (img.size[0], img.size[1])
            
            
            if 'good' in filename:    
                ground_truth =Image.new('L',(img_size[0],img_size[1]),0)
                img, ground_truth = transform_test(img, ground_truth)
                
                return filename, img, ground_truth, 0
            else:   
                # different ground truth schema for mvtec_loco
                if self.args.mode=='mvtec_loco':
                    ground_truth = Image.open(filename.replace("test", "ground_truth").replace(".png", "/000.png"))                        
                if self.args.mode=='mvtec':
                    ground_truth = Image.open(filename.replace("test", "ground_truth").replace(".png", "_mask.png"))
                # TODO check for mask if resized correctly for loco

                ground_truth = self.resize_transform_loco(ground_truth)  
                img, ground_truth = transform_test(img, ground_truth)
                
                return filename, img, ground_truth, 1


    def __len__(self):
        return len(self.files)
    
###--------------------------------------------------------------- Beantech
class ImageDataset_beantec(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train', train_paths=None, test_paths=None):
        
        self.img_size = 280 * args.factor
        self.crop_size = 256 * args.factor
        self.args = args
        self.mode = mode
        if train_paths is None and test_paths is None:
            raise ValueError("either test or train paths must be provided depending on the mode")
        
        self.train_paths = train_paths
        self.test_paths = test_paths
        
        self.transform_train = transforms.Compose([ transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                                transforms.Pad(int(self.crop_size/10),fill=0,padding_mode='constant'),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225 ]) ])
        
        self.resize_transform_bean = transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC)

        if mode == 'train':
            self.files = train_paths
        elif mode == 'test':
            self.files = test_paths
            
    def _align_transform(self, img, mask):
        #resize to 224
        img = TF.resize(img, self.crop_size, Image.BICUBIC)
        mask = TF.resize(mask, self.crop_size, Image.NEAREST)
        #toTensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        #normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
        return img, mask
    
    def _unalign_transform(self, img, mask):
        #resize to 256
        img = TF.resize(img, self.img_size, Image.BICUBIC)
        mask = TF.resize(mask, self.img_size, Image.NEAREST)
        #random rotation
        angle = transforms.RandomRotation.get_params([-10, 10])
        img = TF.rotate(img, angle, fill=(0,))
        mask = TF.rotate(mask, angle, fill=(0,))
        #random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        #toTensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        #normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
        return img, mask

    def __getitem__(self, index):
        filename = self.files[index]
        img = Image.open(filename)
        img = img.convert('RGB')
        img = self.resize_transform_bean(img)

        if self.mode == 'train':
            img = self.transform_train(img)            
            return filename, img
        
        elif self.mode == 'test':
            transform_test = self._unalign_transform if self.args.unalign_test else self._align_transform
            img_size = (img.size[0], img.size[1])
            
            if 'ok' in os.path.split(filename)[0].split(os.sep):    
                ground_truth =Image.new('L',(img_size[0],img_size[1]),0)
                img, ground_truth = transform_test(img, ground_truth)
                return filename, img, ground_truth, 0
            else:   
                ground_truth = Image.open(filename.replace("test", "ground_truth").replace(".bmp", ".png"))
                ground_truth = self.resize_transform_bean(ground_truth)  
                img, ground_truth = transform_test(img, ground_truth) 
                # TODO check for mask if resized correctly
                return filename, img, ground_truth, 1
            
    def __len__(self):
        return len(self.files)   
    
## Original Loader
class ImageDataset(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train'):
        self.img_size = 280 * args.factor
        self.crop_size = 256 * args.factor
        
        self.transform_train = transforms.Compose([ transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                                transforms.Pad(int(self.crop_size/10),fill=0,padding_mode='constant'),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225 ])
                                                ])

        self.args = args
        self.mode = mode
        if mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, mode) + '/good/*.*'))
        elif mode == 'test':
            self.files = sorted(glob.glob(os.path.join(root, mode) + '/*/*.*'))

    def _align_transform(self, img, mask):
        #resize to 224
        img = TF.resize(img, self.crop_size, Image.BICUBIC)
        mask = TF.resize(mask, self.crop_size, Image.NEAREST)
        #toTensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        #normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
        return img, mask

    def _unalign_transform(self, img, mask):
        #resize to 256
        img = TF.resize(img, self.img_size, Image.BICUBIC)
        mask = TF.resize(mask, self.img_size, Image.NEAREST)
        #random rotation
        
        angle = transforms.RandomRotation.get_params([-10, 10])
        img = TF.rotate(img, angle, fill=(0,))
        mask = TF.rotate(mask, angle, fill=(0,))

        #random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        #toTensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        #normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225 ])
        return img, mask


    def __getitem__(self, index):
        filename = self.files[index]
        img = Image.open(filename)
        img = img.convert('RGB')

        if self.mode == 'train':
            img = self.transform_train(img)
            return filename, img

        elif self.mode == 'test':
            transform_test = self._unalign_transform if self.args.unalign_test else self._align_transform
            img_size = (img.size[0], img.size[1])

            if 'good' in filename:
                ground_truth = Image.new('L',(img_size[0],img_size[1]),0)
                img, ground_truth = transform_test(img, ground_truth)
                return filename, img, ground_truth, 0
            else:
                ground_truth = Image.open(filename.replace("test", "ground_truth").replace(".png", "_mask.png"))
                img, ground_truth = transform_test(img, ground_truth)
                return filename, img, ground_truth, 1


    def __len__(self):
        return len(self.files)

class JsonDataset(Dataset):
    def __init__(self, args, meta_dir, mode='train'):
        self.img_size = 280 * args.factor
        self.crop_size = 256 * args.factor
        
        self.transform_train = transforms.Compose([ transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                                transforms.Pad(int(self.crop_size/10),fill=0,padding_mode='constant'),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225 ])
                                                ])
        self.transform_test = transforms.Compose([transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

        self.files = []
        self.mode = mode
        self.args = args
        with open(meta_dir, mode='r') as reader:
            for json_data in reader.readlines():
                dict_data = json.loads(json_data)
                self.files.append(dict_data)


    def __getitem__(self, index):
        instance = self.files[index]
        filename = instance["filename"]
        label = instance["label"]
        filepath = os.path.join(self.args.img_dir, filename)
        img = Image.open(filepath)
        img = img.convert('RGB')
        if self.mode == 'train':
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        return img, label


    def __len__(self):
        return len(self.files)


def get_dataloader(args):

    if args.mode == 'mvtec' or args.mode == 'mvtec_loco':
        
        EXPERIMENT_LOG_PATH = os.path.join(args.results_dir,args.data_set ,
                                    f'contamination_{int(args.contamination_rate*100)}',
                                    f'{args.exp_name}-{args.data_category}')
        
        if os.path.exists(os.path.join(EXPERIMENT_LOG_PATH,"experiment_paths.json")) and not args.development:
            with open(os.path.join(EXPERIMENT_LOG_PATH,"experiment_paths.json"), "r") as file:
                experiment_paths = json.load(file)
                print("sucessfully loaded experiment paths")   
                
        else:    
            if args.mode == 'mvtec':    
                normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test = get_paths_mvtec(args,verbose=True)
                
            if args.mode == 'mvtec_loco':
                normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test = get_paths_mvtec_loco(args,verbose=True)

            # sample from train paths if less trainming data should be used
            train_paths = normal_images + sampled_anomalies_for_train
            train_paths=random.sample(train_paths,int(len(train_paths)*args.data_ratio))
                    
            valid_paths = validation_images + sampled_anomalies_for_val
            test_paths = good_images_test + remaining_anomalies_test
            experiment_paths={'train':train_paths,'test':test_paths,'valid':valid_paths,'contamination_rate':args.contamination_rate,'seed':args.seed}
            
            if not args.development:
                os.makedirs(EXPERIMENT_LOG_PATH, exist_ok=True)
                with open(os.path.join(EXPERIMENT_LOG_PATH,"experiment_paths.json"), "w") as file:
                    json.dump(experiment_paths, file)
        
        
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        train_dataloader = DataLoader(ImageDataset_mvtec(args,DATA_PATH,mode='train',train_paths = experiment_paths['train'],test_paths = None),
                                                        batch_size=args.batch_size,shuffle=True,num_workers=args.n_cpu,drop_last=False)
        if len(experiment_paths['valid']) > 0:
            valid_dataloader = DataLoader(ImageDataset_mvtec(args,DATA_PATH,mode='test',train_paths = None,test_paths = experiment_paths['valid']),
                                                batch_size=args.batch_size,shuffle=False,num_workers=1,drop_last=False)
        else:
            valid_dataloader = None
            
        test_dataloader = DataLoader(ImageDataset_mvtec(args,DATA_PATH,mode='test',train_paths = None,test_paths = experiment_paths['test']),
                                                        batch_size=args.batch_size,shuffle=False,num_workers=1,drop_last=False)
    

    if args.mode == 'beantec':
        
        EXPERIMENT_LOG_PATH = os.path.join(args.results_dir,args.data_set ,
                                    f'contamination_{int(args.contamination_rate*100)}',
                                    f'{args.exp_name}-{args.data_category}')
        
        if os.path.exists(os.path.join(EXPERIMENT_LOG_PATH,"experiment_paths.json")) and not args.development:
            with open(os.path.join(EXPERIMENT_LOG_PATH,"experiment_paths.json"), "r") as file:
                experiment_paths = json.load(file)     
                
        else:    
            normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test = get_paths_beantec(args,verbose=True)    
            
            train_paths = normal_images + sampled_anomalies_for_train
            train_paths=random.sample(train_paths,int(len(train_paths)*args.data_ratio))
            valid_paths = validation_images + sampled_anomalies_for_val
            test_paths = good_images_test + remaining_anomalies_test
            experiment_paths={'train':train_paths,'test':test_paths,'valid':valid_paths,'contamination_rate':args.contamination_rate,'seed':args.seed}
            
            if not args.development:
                os.makedirs(EXPERIMENT_LOG_PATH, exist_ok=True)
                with open(os.path.join(EXPERIMENT_LOG_PATH,"experiment_paths.json"), "w") as file:
                    json.dump(experiment_paths, file)
        
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        train_dataloader = DataLoader(ImageDataset_beantec(args,DATA_PATH,mode='train',train_paths = experiment_paths['train'],test_paths = None),
                                                        batch_size=args.batch_size,shuffle=True,num_workers=args.n_cpu,drop_last=False)
        if len(experiment_paths['valid']) > 0:
            valid_dataloader = DataLoader(ImageDataset_beantec(args,DATA_PATH,mode='test',train_paths = None,test_paths = experiment_paths['valid']),
                                                batch_size=args.batch_size,shuffle=False,num_workers=1,drop_last=False)
        else:
            valid_dataloader = None
        test_dataloader = DataLoader(ImageDataset_beantec(args,DATA_PATH,mode='test',train_paths = None,test_paths = experiment_paths['test']),
                                                        batch_size=args.batch_size,shuffle=False,num_workers=1,drop_last=False)
        
        
    
    if args.mode == 'visa':
        pass        
    if args.mode == 'utrad_mvtec':
        
        if args.contamination_rate != 0.0:
            raise ValueError("Contamination rate should be 0.0 for clean original implementation")
        DATA_PATH=os.path.join(args.data_root,args.data_category)
        
        train_dataloader = DataLoader(ImageDataset(args,DATA_PATH, mode='train'),
                                                    batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=False)
        test_dataloader = DataLoader(ImageDataset(args, DATA_PATH, mode='test'),
                            batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)


    return train_dataloader,valid_dataloader, test_dataloader


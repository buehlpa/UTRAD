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
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
#*2

# TODO loader for , loco , visa , beantec


# TODO loader for mvtec with contamination rate ,provide paths to image dataset
class ImageDataset_mvtec(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train', train_paths=None, test_paths=None):
        
        self.img_size = 280 * args.factor
        self.crop_size = 256 * args.factor
        self.args = args
        self.mode = mode
        if train_paths is None or test_paths is None:
            raise ValueError("train_paths and test_paths must be provided")
        
        self.train_paths = train_paths
        self.test_paths = test_paths
        
        self.transform_train = transforms.Compose([ transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                                transforms.Pad(int(self.crop_size/10),fill=0,padding_mode='constant'),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225 ]) ])
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

# Configure dataloaders
def Get_dataloader(args,mode='original_mvtec'):


    if mode == 'contam_mvtec':
        ## get dataset paths for test and train 
        
        # TODO change dataloader from  dev_ file
        train_paths 
        test_paths
        
        
        
        train_dataloader = DataLoader(ImageDataset_mvtec(args, "%s/%s" % (args.data_root,args.dataset_name), mode='train', train_paths = train_paths, test_paths = test_paths),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=False)

        test_dataloader = DataLoader(ImageDataset_mvtec(args, "%s/%s" % (args.data_root,args.dataset_name), mode='test', train_paths = train_paths, test_paths = test_paths),
                            batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)


    if mode == 'original_mvtec':
        train_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), mode='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=False)

        test_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), mode='test'),
                            batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    # train_dataloader = DataLoader(JsonDataset(args, "train.jsonl", mode='train'),
    #                     batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=False)

    # test_dataloader = DataLoader(JsonDataset(args, "test.jsonl", mode='test'),
    #                         batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    return train_dataloader, test_dataloader

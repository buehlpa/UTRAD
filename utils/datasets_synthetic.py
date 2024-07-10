import os
import torch
import math
import numpy as np
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as TF


# from data.perlin import rand_perlin_2d_np

texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                'Class6', 'Class7', 'Class8', 'Class9', 'Class10']



def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):  #shape (256 256) res (16,2))
    delta = (res[0] / shape[0], res[1] / shape[1]) #(1/16,1,128)
    d = (shape[0] // res[0], shape[1] // res[1])  #(16,128)
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1   #delta 为间隔 0:res[0]为上下界。 (256,256,2)

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)    #(17,3)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)  #(17,3,2)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1) # (272,384,2)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]) #(256,256)
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]]) #(256,256,2)
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]) #(256,256)

"""
def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    #grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing='ij'), dim=-1) % 1
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(
        0, res[1], delta[1])), dim=-1) % 1    
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
"""



texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']

class MVTecSynthAnoDataset(Dataset):

    def __init__(self, args, root, transforms_=None, mode='train', train_paths=None, test_paths=None):


        #data_path,classname,img_size,args

        ##################### origianl loader
        self.img_size = 280 * args.factor
        self.crop_size = 256 * args.factor
        self.args = args
        self.mode = mode
        if train_paths is None and test_paths is None:
            raise ValueError("either test or train paths must be provided depending on the mode")
        
        self.train_paths = train_paths
        self.test_paths = test_paths
        
        if mode == 'train':
            self.files = train_paths
            
        elif mode == 'test':
            self.files = test_paths
            
        
        print(f"Number of images in {mode} mode: {len(self.files)}")
        
        self.transform_train = transforms.Compose([ transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC),
                                                transforms.Pad(int(self.crop_size/10),fill=0,padding_mode='constant'),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225 ]) ])
        ########################################

        self.resize_transform_loco = transforms.Resize((self.crop_size, self.crop_size), Image.BICUBIC)

        

        self.classname=args.data_category
        
        self.root_dir = os.path.join(root,'train','good')
        self.resize_shape = [self.crop_size, self.crop_size]
        self.anomaly_source_path = args.synthetic_anomaly_root
        
        

        self.image_paths = train_paths
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        

        #foreground path of textural classes
        foreground_path = os.path.join(args.data_root,'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

    def __len__(self):
        return len(self.files)
    
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

    def random_choice_foreground_path(self):
        foreground_path_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_path_id]
        return foreground_path


    def get_foreground_mvtec(self,image_path):
        classname = self.classname
        if classname in texture_list:
            foreground_path = self.random_choice_foreground_path()
        else:
            foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path



    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

    # no_anomaly = torch.rand(1).numpy()[0]
    # if no_anomaly > 0.5:
    #     image = image.astype(np.float32)
    #     return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

    # else:
        perlin_scale = 6  
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale,perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale,perlin_scale, (1,)).numpy()[0])

        has_anomaly = 0
        try_cnt = 0
        while(has_anomaly == 0 and try_cnt<50):  
            perlin_noise = rand_perlin_2d_np(
                (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            
            object_perlin = thresh*perlin_thr

            object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)  

            msk = (object_perlin).astype(np.float32) 
            if np.sum(msk) !=0: 
                has_anomaly = 1        
            try_cnt+=1
            
        
        if self.classname in texture_list: # only DTD
            #print('texture selected')
            aug = self.randAugmenter()
            anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            anomaly_img_augmented = anomaly_source_img# aug(image=anomaly_source_img)# no aug
            img_object_thr = anomaly_img_augmented.astype(
                np.float32) * object_perlin/255.0
            
        else: # DTD and self-augmentation
            texture_or_patch = torch.rand(1).numpy()[0]
            if texture_or_patch > 0.5:  # >0.5 is DTD 
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = anomaly_source_img#aug(image=anomaly_source_img)
                anomaly_img_augmented = anomaly_source_img# aug(image=anomaly_source_img)# no aug
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0

            else: #self-augmentation
                aug = self.randAugmenter()
                anomaly_image = cv2_image#aug(image=cv2_image)#no aug
                high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                gird_high, gird_width = int(high/8), int(width/8)
                wi = np.split(anomaly_image, range(
                    gird_width, width, gird_width), axis=1)
                wi1 = wi[::2]
                random.shuffle(wi1)
                wi2 = wi[1::2]
                random.shuffle(wi2)
                width_cut_image = np.concatenate(
                    (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                hi = np.split(width_cut_image, range(
                    gird_high, high, gird_high), axis=0)
                random.shuffle(hi)
                hi1 = hi[::2]
                random.shuffle(hi1)
                hi2 = hi[1::2]
                random.shuffle(hi2)
                mixer_cut_image = np.concatenate(
                    (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                img_object_thr = mixer_cut_image.astype(
                    np.float32) * object_perlin/255.0

        beta = torch.rand(1).numpy()[0] * 0.6 + 0.2
        augmented_image = image * \
            (1 - object_perlin) + (1 - beta) * \
            img_object_thr + beta * image * (object_perlin)

        augmented_image = augmented_image.astype(np.float32)

        return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, index):
        
        
        
        filename = self.files[index]
        

        
        
        if self.mode == 'train':
            
            gets_anomaly_rand = torch.rand(1).numpy()[0]
            
            
            if gets_anomaly_rand > self.args.synthetic_ratio:
                has_anomaly=1
                
                image = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
                
                cv2_image=image
                thresh_path = self.get_foreground_mvtec(filename)
                
                #print(f'thresh_path: {thresh_path}')
                
                thresh=cv2.imread(thresh_path,0)
                thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

                thresh = np.array(thresh).astype(np.float32)/255.0 
                image = np.array(image).astype(np.float32)/255.0

                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
                
                augmented_image, anomaly_mask, has_anomaly_per  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
                
                augmented_image = np.transpose(augmented_image, (2, 0, 1))
                image = np.transpose(image, (2, 0, 1))
                anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
                
                
                #print(np.shape(augmented_image))
                augmented_image=torch.from_numpy(augmented_image)
                
                #print(f'type of img: {type(augmented_image)}, shape of img: {augmented_image.shape} dtype of img: {augmented_image.dtype}')
                
                
                
                
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(augmented_image)
                augmented_image = self.transform_train(pil_image)   
                return filename, augmented_image, has_anomaly
            
            
            # if there is no anomaly created
            else:
                has_anomaly = 0
                #original resize and transform 
                # img = self.transform_train(img)      
                # with rand augmenter
                image = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
                # aug = self.randAugmenter()
                # image=aug(image=image)
                
                img = np.array(image).astype(np.float32)/255.0
                img=img.transpose(2, 0, 1)
                img=torch.from_numpy(img)
                
                
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(img)
                img = self.transform_train(pil_image)  
                 
                 
                 
                return filename, img, has_anomaly
        
        elif self.mode == 'test':
            # img = Image.open(filename)
            # img = img.convert('RGB')
            
            image = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            img = np.array(image).astype(np.float32)/255.0
            img=img.transpose(2, 0, 1)
            img=torch.from_numpy(img)
            to_pil = transforms.ToPILImage()
            img = to_pil(img)
            
            #print(filename)
            
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
                
                ground_truth = self.resize_transform_loco(ground_truth)  
                img, ground_truth = transform_test(img, ground_truth)
                
                return filename, img, ground_truth, 1
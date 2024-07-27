import torch
import json
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from transformers import ViTFeatureExtractor, ViTModel
from transformers import AutoFeatureExtractor, AutoModel
from torchvision import models, transforms
import torch.nn.functional as F
import torchvision.transforms as T

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'







def refine_paths(experiment,args):



    path_list = experiment['train']

    def load_image(filename, crop_size=256, aligned=True, img_size=280):
        img = Image.open(filename)
        img = img.convert('RGB')
        
        if aligned:
            img = TF.resize(img, crop_size, Image.BICUBIC)
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            img = TF.resize(img, img_size, Image.BICUBIC)
            angle = transforms.RandomRotation.get_params([-10, 10])
            img = TF.rotate(img, angle, fill=(0,))
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop_size, crop_size))
            img = TF.crop(img, i, j, h, w)
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = img.to(torch.float32)
        return img

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    featuremaps = []
    backbone = models.resnet18(pretrained=True).to(device)
    backbone.eval()
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)

    img_list = [load_image(path, aligned=True) for path in path_list]

    with torch.no_grad():
        for img in img_list:
            img = img.unsqueeze(0).to(device)
            outputs = []
            _ = backbone(img)
            outputs = embedding_concat(embedding_concat(outputs[0], outputs[1]), outputs[2])
            featuremaps.append(outputs.cpu().numpy())


    shuffled_final = np.array(featuremaps).squeeze()

    isolation_outliers = []
    lof_outliers = []

    for i in range(shuffled_final.shape[1]):
        feature_maps = shuffled_final[:, i, :, :]
        flattened_feature_maps = feature_maps.reshape(feature_maps.shape[0], -1)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(flattened_feature_maps)

        clf_isolation = IsolationForest(contamination=args.assumed_contamination_rate)
        outliers_isolation = clf_isolation.fit_predict(features_scaled)
        isolation_outliers.append(outliers_isolation)

        clf_lof = LocalOutlierFactor(n_neighbors=20, contamination=args.assumed_contamination_rate)
        outliers_lof = clf_lof.fit_predict(features_scaled)
        lof_outliers.append(outliers_lof)

    isolation_outliers = np.array(isolation_outliers)
    lof_outliers = np.array(lof_outliers)

    isolation_column_sums = isolation_outliers.sum(axis=0)
    lof_column_sums = lof_outliers.sum(axis=0)


    remove_inds=np.array(lof_column_sums).argsort()[:int(args.assumed_contamination_rate * len(lof_column_sums))]
    selected_paths = [path_list[i] for i in remove_inds]
    refined_paths= [path for path in path_list if path not in selected_paths]

    experiment['isoforest_scores'] = isolation_column_sums.tolist()
    experiment['lof_scores'] = lof_column_sums.tolist()
    experiment['lof_refined_paths'] = refined_paths

    return experiment

    # with open(experiment_path, 'w') as file:
    #     json.dump(experiment, file)
        


    
def load_image(filename, crop_size=256, aligned=True, img_size=280):
    img = Image.open(filename)
    img = img.convert('RGB')
    
    if aligned:
        img = TF.resize(img, crop_size, Image.BICUBIC)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        img = TF.resize(img, img_size, Image.BICUBIC)
        angle = transforms.RandomRotation.get_params([-10, 10])
        img = TF.rotate(img, angle, fill=(0,))
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop_size, crop_size))
        img = TF.crop(img, i, j, h, w)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img.to(torch.float32)
    return img


def feature_extraction(experiment,args):



    path_list = experiment['train']

    def load_image(filename, crop_size=256, aligned=True, img_size=280):
        img = Image.open(filename)
        img = img.convert('RGB')
        
        if aligned:
            img = TF.resize(img, crop_size, Image.BICUBIC)
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            img = TF.resize(img, img_size, Image.BICUBIC)
            angle = transforms.RandomRotation.get_params([-10, 10])
            img = TF.rotate(img, angle, fill=(0,))
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop_size, crop_size))
            img = TF.crop(img, i, j, h, w)
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img = img.to(torch.float32)
        return img

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    featuremaps = []
    backbone = models.resnet18(pretrained=True).to(device)
    backbone.eval()
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    backbone.layer1[-1].register_forward_hook(hook)
    backbone.layer2[-1].register_forward_hook(hook)
    backbone.layer3[-1].register_forward_hook(hook)

    img_list = [load_image(path, aligned=False) for path in path_list]

    with torch.no_grad():
        for img in img_list:
            img = img.unsqueeze(0).to(device)
            outputs = []
            _ = backbone(img)
            outputs = embedding_concat(embedding_concat(outputs[0], outputs[1]), outputs[2])
            featuremaps.append(outputs.cpu().numpy())


    shuffled_final = np.array(featuremaps).squeeze()
    return shuffled_final















import os

import argparse





def main():



    parser = argparse.ArgumentParser(description="Process ")

    # Add the arguments
    parser.add_argument('--mode', type=str, help='Your name')
    parser.add_argument('--category', type=str, help='Your age')
    parser.add_argument('--run', type=int, help='Your age')

    argu = parser.parse_args()





    if argu.mode=='own':




        # Define the TrainOptions class
        class TrainOptions:
            def __init__(self, category):
                self.exp_name = "Exp0-r18"
                self.epoch_start = 0
                self.epoch_num = 150
                self.factor = 1
                self.seed = 233
                self.num_row = 4
                self.activation = 'gelu'
                self.unalign_test = False
                self.data_root = '/home/bule/projects/datasets/visa_anomaly_detection/'
                self.dataset_name = category
                self.batch_size = 2
                self.lr = 1e-4
                self.b1 = 0.5
                self.b2 = 0.999
                self.n_cpu = 8
                self.image_result_dir = 'result_images'
                self.model_result_dir = 'saved_models'
                self.validation_image_dir = 'validation_images'
                self.assumed_contamination_rate = 0.1

        # Initialize the options
        category = argu.category
        args = TrainOptions(category)

        data = {'paths':[],'gt_labels':[],'own; ISO':[],'own; LOF':[],'TSNE; ISO':[],'TSNE; LOF':[],'PCA; ISO':[],'PCA; LOF':[],'Original; ISO':[],'Original; LOF':[],'FMAPS; ISO':[],"Vit_beans; LOF":[],"Vit_beans; Cosine":[]}
        paths=[f'/home/bule/projects/UTRAD/results/visa/contamination_10/Exp_11_07_24_run_{i}-{category}/experiment_paths.json' for i in range(1,6)]

        for path in paths:
            with open(path, 'r') as file:
                experiment = json.load(file)
            ############################################################################## OWN REFINEMENT FUNCTION
            experiment_refined=refine_paths(experiment,args) 
            labels=[0 if 'Normal' in path else 1 for path in experiment_refined['train']]
            data['paths'].append(experiment_refined['train'])
            data['gt_labels'].append(labels)
            data['own; ISO'].append(np.array(experiment_refined['isoforest_scores']))
            data['own; LOF'].append(np.array(experiment_refined['lof_scores']))
            
            ############################################################################ ON ORIGINAL IMAGE DATA PCA AND TSNE LOF AND ISO
            
            img_list = [load_image(path, aligned=True) for path in experiment_refined['train']]
            img_array = np.array([img.numpy().flatten() for img in img_list])
            
            #PCA
            pca = PCA(n_components=2, random_state=args.seed)
            pca_results = pca.fit_transform(img_array)
            lof_pca = LocalOutlierFactor(n_neighbors=20, contamination=args.assumed_contamination_rate)
            lof_pca_preds = lof_pca.fit_predict(pca_results)
            lof_pca_scores = -lof_pca.negative_outlier_factor_
            
            isf_pca = IsolationForest(contamination=args.assumed_contamination_rate, random_state=args.seed)
            isf_pca.fit(pca_results)
            isf_pca_preds = isf_pca.predict(pca_results)
            iso_scores = -isf_pca.decision_function(pca_results)
            

            
            #TSNE
            tsne = TSNE(n_components=2, random_state=args.seed)
            tsne_results = tsne.fit_transform(img_array)
            
            lof_tsne = LocalOutlierFactor(n_neighbors=20, contamination=args.assumed_contamination_rate)
            lof_tsne_preds = lof_tsne.fit_predict(tsne_results)
            lof_tsne_scores = -lof_tsne.negative_outlier_factor_
            isf_tsne = IsolationForest(contamination=args.assumed_contamination_rate, random_state=args.seed)
            isf_tsne.fit(tsne_results)
            isf_tsne_preds = isf_tsne.predict(tsne_results)
            isf_tsne_scores = -isf_tsne.decision_function(tsne_results)
            
            scaler = MinMaxScaler()
            data['TSNE; ISO'].append(scaler.fit_transform(np.array(isf_tsne_scores).reshape(-1, 1)).flatten())
            data['TSNE; LOF'].append(scaler.fit_transform(np.array(lof_tsne_scores).reshape(-1, 1)).flatten())
            data['PCA; ISO'].append(scaler.fit_transform(np.array(iso_scores).reshape(-1, 1)).flatten())
            data['PCA; LOF'].append( scaler.fit_transform(np.array(lof_pca_scores).reshape(-1, 1)).flatten())
            
            ############################################################################# on orig images
            # LOF
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof_pred = lof.fit_predict(img_array)
            lof_scores = -lof.negative_outlier_factor_
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=args.seed)
            iso_forest.fit(img_array)
            iso_scores = -iso_forest.decision_function(img_array)

            data['Original; ISO'].append( scaler.fit_transform(np.array(iso_scores).reshape(-1, 1)).flatten())
            data['Original; LOF'].append(scaler.fit_transform(np.array(lof_scores).reshape(-1, 1)).flatten())
            # print('run done')
            
            
            #################################### on the features
            fmaps_images=feature_extraction(experiment_refined,args)

            img_array = np.array([img.flatten() for img in fmaps_images])
            iso_forest = IsolationForest(contamination=0.1, random_state=args.seed)
            iso_forest.fit(img_array)
            iso_scores = -iso_forest.decision_function(img_array)
            
            data['FMAPS; ISO'].append(scaler.fit_transform(np.array(iso_scores).reshape(-1, 1)).flatten())

            #############################################################
            
            
            # images=[]
            # for filename in experiment_refined['train']:
            #     img = Image.open(filename)
            #     if img.mode == 'L':  # Check if the image is grayscale (single channel)
            #         img = img.convert('RGB')  # Convert grayscale to RGB (three channels)
            #     images.append(img)
            
            # images=[transformation_chain(image) for image in images]
            # images=torch.stack(images)
            # embeddings = model(images).last_hidden_state[:, 0].cpu()
            # embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # cosine_similarity_matrix = torch.mm(embeddings, embeddings.t())
            # col_sums = np.sum(cosine_similarity_matrix.detach().numpy(), axis=0)
            # colsums_sorted = scaler.fit_transform(col_sums.reshape(-1, 1)).flatten()
            # colsums_sorted=1-colsums_sorted
            # data["Vit_beans; Cosine"].append(colsums_sorted)
            # embeddings_=embeddings.detach().numpy()

            # lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            # outlier_labels = lof.fit_predict(embeddings_)
            # anomaly_scores = -lof.negative_outlier_factor_
            
            # data["Vit_beans; LOF"].append(scaler.fit_transform(np.array(anomaly_scores).reshape(-1, 1)).flatten())
            print("run completed")
            
        filename = f'/home/bule/projects/UTRAD/results/visa/refinement/Refinement_Comparison_{category}.pkl'


        # Dump the dictionary to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
            
    ####################################################################vit_beans        
    if argu.mode=='vit_beans':


        # Define the TrainOptions class
        class TrainOptions:
            def __init__(self, category):
                self.exp_name = "Exp0-r18"
                self.epoch_start = 0
                self.epoch_num = 150
                self.factor = 1
                self.seed = 233
                self.num_row = 4
                self.activation = 'gelu'
                self.unalign_test = False
                self.data_root = '/home/bule/projects/datasets/visa_anomaly_detection/'
                self.dataset_name = category
                self.batch_size = 2
                self.lr = 1e-4
                self.b1 = 0.5
                self.b2 = 0.999
                self.n_cpu = 8
                self.image_result_dir = 'result_images'
                self.model_result_dir = 'saved_models'
                self.validation_image_dir = 'validation_images'
                self.assumed_contamination_rate = 0.1

        # Initialize the options
        category = argu.category
        
        model_ckpt = "nateraw/vit-base-beans"
        extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)
        hidden_dim = model.config.hidden_size
        # Data transformation chain.
        transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * extractor.size["height"])),
                T.CenterCrop(extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
            ]
        )
        
        
        
        
        

        args = TrainOptions(category)
        data = {'paths':[],'gt_labels':[],"Vit_beans; LOF":[],"Vit_beans; Cosine":[]}
        path=f'/home/bule/projects/UTRAD/results/visa/contamination_10/Exp_11_07_24_run_{argu.run}-{category}/experiment_paths.json' 

        with open(path, 'r') as file:
            experiment = json.load(file)
        ############################################################################## OWN REFINEMENT FUNCTION
        labels=[0 if 'Normal' in path else 1 for path in experiment['train']]


        scaler = MinMaxScaler()

        images=[]
        for filename in experiment['train']:
            img = Image.open(filename)
            if img.mode == 'L':  # Check if the image is grayscale (single channel)
                img = img.convert('RGB')  # Convert grayscale to RGB (three channels)
            images.append(img)
        
        images=[transformation_chain(image) for image in images]
        images=torch.stack(images)
        embeddings = model(images).last_hidden_state[:, 0].cpu()
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        cosine_similarity_matrix = torch.mm(embeddings, embeddings.t())
        col_sums = np.sum(cosine_similarity_matrix.detach().numpy(), axis=0)
        colsums_sorted = scaler.fit_transform(col_sums.reshape(-1, 1)).flatten()
        colsums_sorted=1-colsums_sorted
        data["Vit_beans; Cosine"].append(colsums_sorted)
        embeddings_=embeddings.detach().numpy()

        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_labels = lof.fit_predict(embeddings_)
        anomaly_scores = -lof.negative_outlier_factor_
        
        data["Vit_beans; LOF"].append(scaler.fit_transform(np.array(anomaly_scores).reshape(-1, 1)).flatten())
        print("run completed")
    
    
    
        filename = f'/home/bule/projects/UTRAD/results/visa/refinement/Refinement_Comparison_SOTA_{category}.pkl'

        if os.path.exists(filename):
            
            with open(filename, 'rb') as file:
                data_load = pickle.load(file)    
                
                data_load["Vit_beans; LOF"].append(data["Vit_beans; LOF"][0])
                data_load["Vit_beans; Cosine"].append(data["Vit_beans; Cosine"][0])
                
            with open(filename, 'wb') as file:
                pickle.dump(data_load, file)    
                
                
        else:    
            
            # Dump the dictionary to a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(data, file)



    ####################################################################vit image_net
    if argu.mode=='vit_imagenet': 
        
        # Define the TrainOptions class
        class TrainOptions:
            def __init__(self, category):
                self.exp_name = "Exp0-r18"
                self.epoch_start = 0
                self.epoch_num = 150
                self.factor = 1
                self.seed = 233
                self.num_row = 4
                self.activation = 'gelu'
                self.unalign_test = False
                self.data_root = '/home/bule/projects/datasets/visa_anomaly_detection/'
                self.dataset_name = category
                self.batch_size = 2
                self.lr = 1e-4
                self.b1 = 0.5
                self.b2 = 0.999
                self.n_cpu = 8
                self.image_result_dir = 'result_images'
                self.model_result_dir = 'saved_models'
                self.validation_image_dir = 'validation_images'
                self.assumed_contamination_rate = 0.1

        # Initialize the options
        category = argu.category
        

        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


        #embeddings = last_hidden_states[:, 0].cpu()


        transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * feature_extractor.size["height"])),
                T.CenterCrop(feature_extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
        )
        
        
        
        
        
        args = TrainOptions(category)
        data = {'paths':[],'gt_labels':[],"Vit_imagenet; LOF":[],"Vit_imagenet; Cosine":[]}
        path=f'/home/bule/projects/UTRAD/results/visa/contamination_10/Exp_11_07_24_run_{argu.run}-{category}/experiment_paths.json' 

        with open(path, 'r') as file:
            experiment = json.load(file)
        ############################################################################## OWN REFINEMENT FUNCTION



        images=[]
        for filename in experiment['train']:
            img = Image.open(filename)
            if img.mode == 'L':  # Check if the image is grayscale (single channel)
                img = img.convert('RGB')  # Convert grayscale to RGB (three channels)
            images.append(img)
        
        images=[transformation_chain(image) for image in images]
        images=torch.stack(images)
        embeddings = model(images).last_hidden_state[:, 0].cpu()
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        embeddings_=embeddings.detach().numpy()
        scaler = MinMaxScaler()
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_labels = lof.fit_predict(embeddings_)
        anomaly_scores = -lof.negative_outlier_factor_
        
        
        
        data["Vit_imagenet; LOF"].append(scaler.fit_transform(np.array(anomaly_scores).reshape(-1, 1)).flatten())
        print("run completed")
        
        
        cosine_similarity_matrix = torch.mm(embeddings, embeddings.t())
        col_sums = np.sum(cosine_similarity_matrix.detach().numpy(), axis=0)
        colsums_sorted = scaler.fit_transform(col_sums.reshape(-1, 1)).flatten()
        colsums_sorted=1-colsums_sorted
        data["Vit_imagenet; Cosine"].append(colsums_sorted)
        
        
        
        

    
    
    
        filename = f'/home/bule/projects/UTRAD/results/visa/refinement/Refinement_Comparison_SOTA_imagenet_{category}.pkl'

        if os.path.exists(filename):
            
            with open(filename, 'rb') as file:
                data_load = pickle.load(file)    
                
                data_load["Vit_imagenet; LOF"].append(data["Vit_imagenet; LOF"][0])
                data_load["Vit_imagenet; Cosine"].append(data["Vit_imagenet; Cosine"][0])
                
            with open(filename, 'wb') as file:
                pickle.dump(data_load, file)    
                
                
        else:    
            
            # Dump the dictionary to a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(data, file)
                

        
        
if __name__ == "__main__":
    main()
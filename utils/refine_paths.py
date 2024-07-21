import json
import torch
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import torch.nn.functional as F


# experiment_path = "/home/bule/projects/UTRAD/results/mvtec/contamination_10/Exp_07_06_run_1-leather/experiment_paths.json"

def refine_paths(experiment,args):

    # with open(experiment_path, 'r') as file:
    #     experiment = json.load(file)

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

    isolation_outliers = []
    lof_outliers = []

    for i in range(shuffled_final.shape[1]):
        feature_maps = shuffled_final[:, i, :, :]
        flattened_feature_maps = feature_maps.reshape(feature_maps.shape[0], -1)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(flattened_feature_maps)

        clf_isolation = IsolationForest(contamination=0.2)
        outliers_isolation = clf_isolation.fit_predict(features_scaled)
        isolation_outliers.append(outliers_isolation)

        clf_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
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
        

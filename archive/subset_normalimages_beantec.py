import os
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import mean_squared_error, structural_similarity
import matplotlib.pyplot as plt
import json

def load_images_from_directory(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png')):
            filepath = os.path.join(directory, filename)
            images.append(img_as_float(io.imread(filepath)))
            filenames.append(filename)
    return images, filenames

def calculate_mse_matrix(images):
    num_images = len(images)
    mse_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i+1, num_images):
            mse_matrix[i, j] = mean_squared_error(images[i], images[j])
    return mse_matrix

def calculate_ssim_matrix(images):
    num_images = len(images)
    ssim_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i+1, num_images):
            min_dim = min(images[i].shape[:2])
            win_size = min(7, min_dim // 2 * 2 + 1)  # Ensure win_size is odd and <= min_dim
            ssim_matrix[i, j] = structural_similarity(images[i], images[j], multichannel=True, win_size=win_size)
    return ssim_matrix

def plot_matrix(matrix, title, filenames):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(filenames)))
    ax.set_yticks(np.arange(len(filenames)))
    ax.set_xticklabels(filenames, rotation=90)
    ax.set_yticklabels(filenames)
    plt.title(title)
    plt.show()

def save_matrices_and_paths(mse_matrix, filenames, output_dir, ssim_matrix=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, 'mse_matrix.npy'), mse_matrix)
    #np.save(os.path.join(output_dir, 'ssim_matrix.npy'), ssim_matrix)
    
    with open(os.path.join(output_dir, 'filenames.json'), 'w') as f:
        json.dump(filenames, f)

def main():
    directory = '/home/bule/projects/datasets/BTech_Dataset_transformed/03/train/ok'
    output_dir = '/home/bule/projects/UTRAD/results/beantec/allresults'
    
    images, filenames = load_images_from_directory(directory)
    
    mse_matrix = calculate_mse_matrix(images)
    plot_matrix(mse_matrix, 'MSE Matrix', filenames)
    
    
    # ssim_matrix = calculate_ssim_matrix(images)


    # plot_matrix(ssim_matrix, 'SSIM Matrix', filenames)
    
    save_matrices_and_paths(mse_matrix,filenames, output_dir)
main()

import os
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd



def stratified_sample(file_paths, class_list, total_samples,seed=123):
    
    # Categorize file paths by class
    categorized_paths = defaultdict(list)
    for path in file_paths:
        for class_name in class_list:
            if class_name in os.path.split(path)[0].split(os.sep):
                categorized_paths[class_name].append(path)
                break
    
    # Compute total files and sample sizes per class
    total_files = sum(len(paths) for paths in categorized_paths.values())
    sample_sizes = {class_name: int(len(paths) / total_files * total_samples)
                    for class_name, paths in categorized_paths.items()}
    
    # Sample file paths
    sampled_paths = []
    for class_name, size in sample_sizes.items():
        sampled_paths.extend(random.sample(categorized_paths[class_name], min(size, len(categorized_paths[class_name]))))
    
    # Determine remaining paths
    remaining_paths = [path for path in file_paths if path not in sampled_paths]
    
    return sampled_paths, remaining_paths


def count_files_by_class(file_paths, class_list):
    # Initialize counts for each class
    class_counts = {class_name: 0 for class_name in class_list}
    
    # Count occurrences of each class in file paths
    for path in file_paths:
        for class_name in class_list:
            if class_name in os.path.split(path)[0].split(os.sep):
                class_counts[class_name] += 1
                break  # Stop checking other classes if the current one matches
    return class_counts

# mvtec
def get_paths_mvtec(args,verbose=True):
    
    anomaly_categories=args.dataset_parameters['anomaly_categories']
    category=args.data_category
    validation=args.dataset_parameters['use_validation']
    validation_split=args.dataset_parameters['validation_split']
    DATA_PATH=args.data_root
    
    
    NORMAL_PATH = os.path.join(DATA_PATH, f'{category}/train/good')
    ANOMALY_PATH = os.path.join(DATA_PATH , f'{category}/test')
    
        
    normal_images=[os.path.join(NORMAL_PATH,item) for item in os.listdir(NORMAL_PATH)]
    file_path = []
    for root, _, files in os.walk(ANOMALY_PATH):
        for file in files:
            file_path.append( os.path.join(root, file))
        
    anomaly_images_test=[item for item in file_path if "good" not in item]
    good_images_test=[item for item in file_path if "good" in item]
    
    n_samples = int(len(normal_images)*args.contamination_rate)
    
    sampled_anomalies_for_train, remaining_anomalies_test = stratified_sample(anomaly_images_test, anomaly_categories[category], n_samples, args.test_seed)

    if validation_split > 0:
        if validation!= True:
            raise ValueError('validation_split is set to > 0 but use_validation is set to False')
        normal_images,validation_images = train_test_split(normal_images, test_size=validation_split, random_state=args.seed)
        sampled_anomalies_for_train, sampled_anomalies_for_val = train_test_split(sampled_anomalies_for_train, test_size=validation_split, random_state=args.seed)
    else:
        sampled_anomalies_for_val = []
        validation_images = []

    if verbose:
        print(f'category: {category}, normals train:  {len(normal_images)}, anomalies test: {len(anomaly_images_test)}, normal test: {len(good_images_test)}')       
        print(f'anomalies test total:     {count_files_by_class(anomaly_images_test, anomaly_categories[category])}')
        print(f'anomalies test sampled:   {count_files_by_class(sampled_anomalies_for_train, anomaly_categories[category])}')
        print(f'anomalies test remaining: {count_files_by_class(remaining_anomalies_test, anomaly_categories[category])}')
    
    return normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test

def get_paths_mvtec_loco(args,verbose=True):
    

    anomaly_categories=args.dataset_parameters['anomaly_categories']
    category=args.data_category
    validation=args.dataset_parameters['use_validation']
    validation_split=args.dataset_parameters['validation_split']
    DATA_PATH=args.data_root
    
    NORMAL_PATH = os.path.join(DATA_PATH, f'{category}/train/good')
    VALIDATION_PATH = os.path.join(DATA_PATH, f'{category}/validation/good')
    ANOMALY_PATH = os.path.join(DATA_PATH , f'{category}/test')
    
    file_path = []
    for root, dirs, files in os.walk(ANOMALY_PATH):
        for file in files:
            file_path.append( os.path.join(root, file))
        
    anomaly_images_test=[item for item in file_path if "good" not in item]
    good_images_test=[item for item in file_path if "good" in item]
    
    normal_images=[os.path.join(NORMAL_PATH,item) for item in os.listdir(NORMAL_PATH)]
    validation_images=[os.path.join(VALIDATION_PATH,item) for item in os.listdir(VALIDATION_PATH)]

    
    n_samples = int((len(normal_images)+len(validation_images))*args.contamination_rate)
    valid_train_ratio=float(len(validation_images)/(len(normal_images)+len(validation_images)))
    
    
    
    sampled_anomalies_for_train, remaining_anomalies_test = stratified_sample(anomaly_images_test,anomaly_categories[category], n_samples, args.test_seed)

    if validation:
        warnings.warn(f"Vaidation split is set to {validation_split}, but the dataset is already split into train and validation sets by publisher. Ignoring validation split ratio.")
        sampled_anomalies_for_train, sampled_anomalies_for_val = train_test_split(sampled_anomalies_for_train, test_size=valid_train_ratio, random_state=args.seed)
    else:
        sampled_anomalies_for_val = []

    if verbose:
        print(f'category: {category}, normals train:  {len(normal_images)}, anomalies test: {len(anomaly_images_test)}, normal test: {len(good_images_test)}')       
        print(f'anomalies test total:     {count_files_by_class(anomaly_images_test, anomaly_categories[category])}')
        print(f'anomalies test sampled:   {count_files_by_class(sampled_anomalies_for_train, anomaly_categories[category])}')
        print(f'anomalies test remaining: {count_files_by_class(remaining_anomalies_test, anomaly_categories[category])}')
        
    return normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test


def get_paths_beantec(args,verbose=True):
    
    anomaly_categories=args.dataset_parameters['anomaly_categories']
    category=args.data_category
    validation=args.dataset_parameters['use_validation']
    validation_split=args.dataset_parameters['validation_split']
    DATA_PATH=args.data_root
    
    NORMAL_PATH = os.path.join(DATA_PATH, f'{category}/train/ok')
    ANOMALY_PATH = os.path.join(DATA_PATH , f'{category}/test')
    
        
        
    if category=="03":
        normal_images=args.dataset_parameters["reduced_train_paths"]["03"]
        
    else:   
        normal_images=[os.path.join(NORMAL_PATH,item) for item in os.listdir(NORMAL_PATH)]
    
    
    
    file_path = []
    for root, _, files in os.walk(ANOMALY_PATH):
        for file in files:
            file_path.append( os.path.join(root, file))
        
    anomaly_images_test=[item for item in file_path if "ok" not in os.path.split(item)[0].split(os.sep)]
    good_images_test=[item for item in file_path if "ok" in os.path.split(item)[0].split(os.sep)]
    

    
    n_samples = int(len(normal_images)*args.contamination_rate)
    sampled_anomalies_for_train, remaining_anomalies_test = stratified_sample(anomaly_images_test, anomaly_categories[category], n_samples, args.seed)

    if validation_split > 0:
        if validation!= True:
            raise ValueError('validation_split is set to > 0 but use_validation is set to False')
        normal_images,validation_images = train_test_split(normal_images, test_size=validation_split, random_state=args.seed)
        sampled_anomalies_for_train, sampled_anomalies_for_val = train_test_split(sampled_anomalies_for_train, test_size=validation_split, random_state=args.seed)
    else:
        sampled_anomalies_for_val = []
        validation_images = []
        
    if verbose:
        print(f'category: {category}, normals train:  {len(normal_images)}, anomalies test: {len(anomaly_images_test)}, normal test: {len(good_images_test)}')       
        print(f'anomalies test total:     {count_files_by_class(anomaly_images_test, anomaly_categories[category])}')
        print(f'anomalies test sampled:   {count_files_by_class(sampled_anomalies_for_train, anomaly_categories[category])}')
        print(f'anomalies test remaining: {count_files_by_class(remaining_anomalies_test, anomaly_categories[category])}')
    
    return normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test


def get_paths_visa(args,verbose=True):
    
    DATA_PATH=args.data_root
    
    
    df = pd.read_csv(os.path.join(DATA_PATH,'split_csv/1cls.csv')) # use split 1
    
    
    anomaly_categories=args.dataset_parameters['anomaly_categories']
    category=args.data_category
    validation=args.dataset_parameters['use_validation']
    validation_split=args.dataset_parameters['validation_split']
    
    
    category=args.data_category

    df_cat=df[df['object']==category]
    df_test=df_cat[df_cat['split']=='test']

    normal_train=df_cat[df_cat['split']=='train']['image'].values.tolist()
    normal_test=df_test[df_test['label']=='normal']['image'].values.tolist()
    anomaly_test=df_test[df_test['label']=='anomaly']['image'].values.tolist()
    
    normal_images=[os.path.join(DATA_PATH,item) for item in normal_train]
    good_images_test=[os.path.join(DATA_PATH,item) for item in normal_test]
    anomaly_images_test=[os.path.join(DATA_PATH,item) for item in anomaly_test]
    
    if args.fixed_n_normals != None and args.fixed_n_normals <= len(normal_images):
        normal_images= random.sample(normal_images, args.fixed_n_normals)
    
    n_samples = int(len(normal_images)*args.contamination_rate)
        
    
    
    sampled_anomalies_for_train, remaining_anomalies_test = stratified_sample(anomaly_images_test, anomaly_categories[category], n_samples, args.test_seed)

    if validation_split > 0:
        if validation!= True:
            raise ValueError('validation_split is set to > 0 but use_validation is set to False')
        normal_images,validation_images = train_test_split(normal_images, test_size=validation_split, random_state=args.seed)
        sampled_anomalies_for_train, sampled_anomalies_for_val = train_test_split(sampled_anomalies_for_train, test_size=validation_split, random_state=args.seed)
    else:
        sampled_anomalies_for_val = []
        validation_images = []
        
    if verbose:
        print(f'category: {category}, normals train:  {len(normal_images)}, anomalies test: {len(anomaly_images_test)}, normal test: {len(good_images_test)}')       
        print(f'anomalies test total:     {count_files_by_class(anomaly_images_test, anomaly_categories[category])}')
        print(f'anomalies test sampled:   {count_files_by_class(sampled_anomalies_for_train, anomaly_categories[category])}')
        print(f'anomalies test remaining: {count_files_by_class(remaining_anomalies_test, anomaly_categories[category])}')
    
    return normal_images, validation_images, sampled_anomalies_for_train, sampled_anomalies_for_val, good_images_test, remaining_anomalies_test



def open_args_log(log_file_path):
    # Initialize an empty dictionary to store the parameters
    params = {}
    # Read the log file
    with open(log_file_path, "r") as file:
        for line in file:
            if line.startswith("["):
                # Stop reading when the log part starts
                break
            
            # Process only lines containing ': '
            if ": " in line:
                # Split the line into key and value
                key, value = line.split(": ", 1)
                
                # Clean up the key and value
                key = key.strip()
                value = value.strip()
                
                # Convert certain values to their respective data types
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                elif value.startswith("{") and value.endswith("}"):
                    value = eval(value)  # Use eval to convert dictionary strings to actual dictionaries
                
                # Add the key-value pair to the dictionary if the key is not already present
                if key not in params:
                    params[key] = value
    return params

# usage!
# params=open_args_log(log_file_path)
# for key in params:
#     setattr(args, key, params[key])
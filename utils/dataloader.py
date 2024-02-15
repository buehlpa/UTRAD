
import os
from collections import defaultdict
import random




def stratified_sample(file_paths, class_list, total_samples,seed=123):
    
    random.seed(seed)
    # Categorize file paths by class
    categorized_paths = defaultdict(list)
    for path in file_paths:
        for class_name in class_list:
            if class_name in path:
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
            if class_name in path:
                class_counts[class_name] += 1
                break  # Stop checking other classes if the current one matches
    
    return class_counts

# mvtec
def get_paths_mvtec(contamination=0.0,category='bottle',DATA_PATH='/home/bule/projects/datasets/mvtec_anomaly_detection',verbose=True,seed=123):
    
    anomaly_categories = {
    'bottle': ['broken_large', 'broken_small', 'contamination'],
    'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation'],
    'capsule': ['crack', 'faulty_imprint', 'poke', 'scratch','squeeze'],
    'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
    'grid': ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
    'hazelnut': ['crack', 'cut', 'hole', 'print'],
    'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
    'metal_nut': ['bent', 'color', 'flip', 'scratch'],
    'pill': ['color', 'combined','contamination', 'crack', 'faulty_imprint', 'pill_type','scratch'],
    'screw': ['manipulated_front', 'scratch_head', 'scratch_neck','thread_side', 'thread_top'],
    'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil','rough'],
    'toothbrush': ['defective'],
    'transistor': ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
    'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
    'zipper': ['broken_teeth', 'combined','fabric_border', 'fabric_interior','split_teeth','rough', 'squeezed_teeth']}
    
    NORMAL_PATH = os.path.join(DATA_PATH, f'{category}/train/good')
    ANOMALY_PATH = os.path.join(DATA_PATH , f'{category}/test')
    
    
    print(NORMAL_PATH)
    
    normal_images=[os.path.join(NORMAL_PATH,item) for item in os.listdir(NORMAL_PATH)]
    file_path = []
    for root, dirs, files in os.walk(ANOMALY_PATH):
        for file in files:
            file_path.append( os.path.join(root, file))
        
    anomaly_images_test=[item for item in file_path if "good" not in item]
    good_images_test=[item for item in file_path if "good" in item]
    

    
    n_samples = int(len(normal_images)*contamination)
    
    sampled_anomalies_for_train, remaining_anomalies_test = stratified_sample(anomaly_images_test, anomaly_categories[category], n_samples, seed)

    if verbose:
        print(f'category: {category}, normals train:  {len(normal_images)}, anomalies test: {len(anomaly_images_test)}, normal test: {len(good_images_test)}')       
        print(f'anomalies test total:     {count_files_by_class(anomaly_images_test, anomaly_categories[category])}')
        print(f'anomalies test sampled:   {count_files_by_class(sampled_anomalies_for_train, anomaly_categories[category])}')
        print(f'anomalies test remaining: {count_files_by_class(remaining_anomalies_test, anomaly_categories[category])}')
    
    return normal_images, sampled_anomalies_for_train, good_images_test, remaining_anomalies_test
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score

import pandas as pd
import re
import os
import numpy as np
import seaborn as sns


def read_train_loss(PATH):
    "reads all losses from the args.log file and returns a list of floats"
    with open(PATH, 'r') as file:
        lines = file.readlines()
        
        lines=[line for line in lines if line.startswith("[")]
        lines = [line.strip() for line in lines]   
    lines_new=[float(re.findall(r"Loss:(\d+\.\d+)", item)[0]) for item in lines]
    return lines_new

def read_training_scores(PATH):
    """
    reads the validation_result.log file and returns a dictionary with the metrics
    """
    resdict={"Loss":[],"Loss_scale":[],"image_AUC":[],"pixel_AUC":[]}
    with open(PATH, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]     

    for key in resdict.keys():
        pattern = f"{key}"+r':\s*(\d+\.\d+)'
        lines_new=[]
        for line in lines:
            match = re.search(pattern, line)  
            if match:
                lines_new.append(match.group(1))       
        lines_new=[float(item) for item in lines_new]          
        resdict[key]=lines_new
    return resdict

def read_validation_score(PATH):
    """
    reads the validation_result.log file and returns a dictionary with the metrics
    """
    resdict={"image ROCAUC":[],"unalign image ROCAUC":[],"pixel ROCAUC":[],"unalign pixel ROCAUC":[]}
    with open(PATH, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]     

    for key in resdict.keys():
        pattern = f"{key}"+r':\s*(\d+\.\d+)'
        lines_new=[]
        for line in lines:
            match = re.search(pattern, line)  
            if match:
                lines_new.append(match.group(1))       
        lines_new=[float(item) for item in lines_new]          
        resdict[key]=lines_new
    return resdict

def get_categories_from_run_path(PATH):
    """
     "/results/mvtec/contamination_0/Exp_11_02_24-bottle" 
    gets all the unique categories "-category" at the end of path  
    """
    regex_pattern = r'-(\w+)$'
    extracted_categories = [re.search(regex_pattern, category).group(1) for category in list(os.listdir(PATH))]
    extracted_categories=list(set(extracted_categories))
    return extracted_categories

def plot_losses(RESPATH,dataset,run,experiment):
    
    """_summary_
    Plots the training loss for each category in the run e.g. contamination_0 
    """
    
    RUN_PATH= os.path.join(RESPATH, dataset,run)
    categories=get_categories_from_run_path(RUN_PATH)
    for category in categories:
        ARGS_PATH= os.path.join(RUN_PATH, experiment+ f'{category}', "args.log")
        losses=read_train_loss(ARGS_PATH)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.axhline(y=0.01, color='r', linestyle='dotted')
        plt.title(f'Training Loss {category},{experiment}')
        plt.show()


def get_vals_per_category(RESPATH,dataset,run,experiment):
    
    """
    gets a dictionary with the validation scores for each category in the run e.g. contamination_0

    """
    
    RUN_PATH= os.path.join(RESPATH, dataset,run)
    categories=get_categories_from_run_path(RUN_PATH)
    print(categories)
    resdict={}
    for category in categories:
        VALS_PATH = os.path.join(RESPATH, dataset, run, experiment + f'{category}', "validation_result.log")
        if not os.path.exists(VALS_PATH):
            continue
        valdict = read_validation_score(VALS_PATH)
        resdict[category]=valdict
    return resdict

def plot_vals_per_category(RESPATH,dataset,run,experiment):
    """_summary_
    Plots the validation scores for each category in the run e.g. contamination_0
    """

    resdict=get_vals_per_category(RESPATH,dataset,run,experiment)
    data_for_df = []
    for cls, metrics in resdict.items():
        for metric, values in metrics.items():
            value = values[0] if values else np.nan  # Assign NaN if the list is empty
            data_for_df.append({'Class': cls, 'Metric': metric, 'Value': value})
    df = pd.DataFrame(data_for_df)

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Class', y='Value', hue='Metric', data=df)
    for p in bar_plot.patches:
        height = p.get_height()
        if height > 0:  # Only annotate bars with a height greater than 0
            bar_plot.annotate(format(height, '.2f'),  # Format the value
                                (p.get_x() + p.get_width() / 2., height),  # Position
                                ha = 'center', va = 'center',  # Alignment
                                xytext = (0, 9),  # Distance from the bar
                                textcoords = 'offset points')

    plt.title(f' {experiment}  ,{run}'),plt.xlabel('Class'),plt.ylabel('Value'),plt.legend(title='Metric', loc='lower right'),plt.show()

def plot_vals_per_category_and_contamination(RESPATH,dataset,experiment,category,contam_dir_list=["contamination_0","contamination_2","contamination_4","contamination_6","contamination_8","contamination_10"],metric="image ROCAUC"):
    """
    plots the validation scores for each category and contamination level
    """
    
    contam_list=[int(item.split("_")[-1]) for item in contam_dir_list]
    metric_list=[]
    for run in contam_dir_list:
        PATH= os.path.join(RESPATH, dataset,run,experiment+ f'{category}', "validation_result.log")
        resdict=read_validation_score(PATH)
        metric_list.append(resdict[metric][0])
    plt.ylim(0.5,1.1)
    plt.grid()
    plt.plot(contam_list,metric_list)
    plt.xlabel('Contamination in %' ),plt.ylabel(f'{metric}'),plt.title(f'{experiment}  ,{category}'),plt.show()
    
def extract_auc_values(log_file_path):
    
    """extracts the image and pixel AUC values from the val log file"""
    auc_values = {}
    
    with open(log_file_path, 'r') as file:
        for line in file:
            if "image ROCAUC" in line:
                image_auc_match = re.search(r'image ROCAUC:\s*([0-9.]+)', line)
                if image_auc_match:
                    auc_values['image_AUC'] = [float(image_auc_match.group(1))]
            elif "pixel ROCAUC" in line:
                pixel_auc_match = re.search(r'pixel ROCAUC:\s*([0-9.]+)', line)
                if pixel_auc_match:
                    auc_values['pixel_AUC'] = [float(pixel_auc_match.group(1))]
    return auc_values  
    
def plot_vals_per_category_and_contamination_multirun(RESPATH,dataset,experiment,category,contam_dir_list=["contamination_0","contamination_2","contamination_4","contamination_6","contamination_8","contamination_10"],metric="image ROCAUC"):
    """
    plots the validation scores for each category and contamination level
    """
    
    contam_list=[int(item.split("_")[-1]) for item in contam_dir_list]
    metric_list=[]
    for run in contam_dir_list:
        PATH= os.path.join(RESPATH, dataset,run,experiment+ f'{category}', "validation_result.log")
        resdict=read_validation_score(PATH)
        metric_list.append(resdict[metric][0])
        
        
        
    plt.ylim(0.5,1.1)
    plt.grid()
    plt.plot(contam_list,metric_list)
    plt.xlabel('Contamination in %' ),plt.ylabel(f'{metric}'),plt.title(f'{experiment}  ,{category}'),plt.show()
    
    
def plot_vals_per_category_and_contamination_multirun(RESPATH,dataset,experiment,category,reps=["_run_1","_run_2","_run_3","_run_4","_run_5"],contam_dir_list=["contamination_0","contamination_2","contamination_4","contamination_6","contamination_8","contamination_10"],metric="image ROCAUC"):
    """
    plots the validation scores for each category and contamination level
    
    experiment : str
        experiment name e.g. "Exp_15_02_24"  # no "-" at the end
    """
    
    contam_list=[int(item.split("_")[-1]) for item in contam_dir_list]
    metric_dict={contam:[]  for contam in contam_dir_list}
    
    for contam in contam_dir_list:
        for rep in reps:
            exp_rep_category= experiment+rep + f'-{category}' # e.g. Exp_11_02_24_run_1-bottle
            PATH= os.path.join(RESPATH, dataset,contam,exp_rep_category, "validation_result.log")
            resdict=read_validation_score(PATH)
            metric_dict[contam].append(resdict[metric][0])
            
            
        
    df=pd.DataFrame(metric_dict) 
    fig, ax = plt.subplots()

    for column in df.columns:
        ax.boxplot(df[column], positions=[df.columns.get_loc(column) + 1])#, notch=True)
        
    ax.set_xticklabels(df.columns)
    ax.set_xlabel('Contamination')
    ax.set_ylabel('Value')
    ax.set_title(f'mean over: {len(reps)} {experiment}  ,{category}')
    ax.grid(True)  # Add grid

    plt.show()
    
    
def plot_curves_with_metrics(true_classes, predicted_probs):
    # Compute precision, recall, and thresholds for the precision-recall curve
    precision, recall, _ = precision_recall_curve(true_classes, predicted_probs)
    ap = average_precision_score(true_classes, predicted_probs)

    # Compute false positive rate and true positive rate for ROC curve
    fpr, tpr, _ = roc_curve(true_classes, predicted_probs)
    auroc = roc_auc_score(true_classes, predicted_probs)

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.', label=f'(AP={ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    # Plot ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', label=f' (AUROC={auroc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.tight_layout()

    # Add legend with metrics
    plt.subplot(1, 2, 1)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.legend()

    plt.show()


# splits

def plot_splits(train_ind_ls, idx, train_datashuffled, n_train_sets, anocats, category, stride, window_size, splittype='USDR', EXPERIMENT_PATH=None):
    """Plot training and inference splits with different colors for different anomaly categories.

    Args:
        train_ind_ls (list of lists): List of lists of indices for each training set.
        idx (array-like): Indices of the data.
        train_datashuffled (list of str): List of paths shuffled according to the indices.
        n_train_sets (int): Number of sets to plot.
        anocats (list of str): Anomaly categories to plot.
        category (str): Current category being plotted.
        stride (int): Stride used in splitting the data.
        window_size (int): Window size used in splitting the data.
        splittype (str): Type of split.
        EXPERIMENT_PATH (str): Path to save the plot if specified.
    """

    ano_cols = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']  # Anomaly colors
    colorvec = ['blue'] * len(train_datashuffled)  # All colors to blue (base is normal)

    for i, cat in enumerate(anocats):
        for id_, (col, path) in enumerate(zip(colorvec, train_datashuffled)):
            if cat in path:
                colorvec[id_] = ano_cols[i + 1]

    cats_with_good = ['good'] + anocats

    boolmat = []
    for i in range(n_train_sets):
        boolmat.append(np.isin(idx, train_ind_ls[i]))

    # Creating x and y coordinates for True and False values
    true_points_x = []
    true_points_y = []
    false_points_x = []
    false_points_y = []
    true_color = []
    false_color = []

    for i, row in enumerate(boolmat):
        for j, value in enumerate(row):
            if value:
                true_points_x.append(j)
                true_points_y.append(i)
                true_color.append(colorvec[j])
            else:
                false_points_x.append(j)
                false_points_y.append(i)
                false_color.append(colorvec[j])

    plt.figure(figsize=(25, 6))
    plt.grid(True, alpha=0.2)
    plt.scatter(true_points_x, true_points_y, c=true_color, marker='^', label='Train')
    plt.scatter(false_points_x, false_points_y, c=false_color, marker=".", label='Infer')

    # Add categories to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    for cat, col in zip(cats_with_good, ano_cols):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10))
        labels.append(cat)

    plt.title(f'Type:{splittype}, {n_train_sets} Sets; window size: {window_size}, stride: {stride}, category: {category}')
    plt.xlabel('Sample Nr')
    plt.ylabel('Training Set NR.')

    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if EXPERIMENT_PATH is not None:
        filename = f'Type_{splittype}_{n_train_sets}_Sets_window_size_{window_size}_stride_{stride}_category_{category}.svg'
        save_path = os.path.join(EXPERIMENT_PATH, filename)
        plt.savefig(save_path, format='svg')
        
        filename = f'Type_{splittype}_{n_train_sets}_Sets_window_size_{window_size}_stride_{stride}_category_{category}.png'
        save_path = os.path.join(EXPERIMENT_PATH, filename)
        plt.savefig(save_path, format='png')
        plt.close()
    else:
        plt.show()
        
        
def plot_multiple_training_scores(filepaths, title='Screw 10 Percent contam', log_scale=True):
    dataframes = []
    for filepath in filepaths:
        resdict = read_training_scores(filepath)
        df = pd.DataFrame(resdict)
        if log_scale:
            df['Loss_scale'] = np.log(df['Loss_scale'])
        dataframes.append(df)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    legend_labels = []
    for idx, df in enumerate(dataframes):
        for i, column in enumerate(df.columns):
            axs[i].plot(df.index, df[column], linestyle='-', label=f'Run {idx + 1}')
            legend_labels.append(f'Run {idx + 1}')

    for i, column in enumerate(dataframes[0].columns):
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(column)
        axs[i].set_title(f'{column}')
        axs[i].grid(True)
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(dataframes), fontsize='small')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    
def last_rows_metric(filepaths,row=-1):
    """gets reads each df in filepaths list and conacatenases the last row into a new df"""
    dfs=[]
    for path in filepaths:
        dfs.append(pd.DataFrame(read_training_scores(path)))
    last_rows = [df.iloc[row] for df in dfs]
    last_rows_df = pd.DataFrame(last_rows)
    last_rows_df.reset_index(drop=True, inplace=True)
    return last_rows_df

def all_results_conc(filepaths):
    """cocnatenate all result to one df in filepaths list"""
    dfs=[]
    for path in filepaths:
        df=pd.DataFrame(read_training_scores(path))
        df['epoch'] = df.index
        dfs.append(df)
    rows_df = pd.concat(dfs)
    rows_df.reset_index(drop=True, inplace=True)
    return rows_df

def get_anomaly_dictionaries(category):
    # get markes dicts for plotting anomalies
    base_marker = {'good': '.'}
    base_color = {'good': 'blue'}
    base_color_clean = {'good': 'skyblue'}

    category_anomalies = anomaly_categories.get(category, [])
    
    marker_options = ['x', '^', 's', 'p', 'D', 'v', '<', '>', 'h', 'H', '*', '+', 'X', 'P', 'd']
    color_options = ['green', 'red', 'dodgerblue', 'purple', 'olive', 'brown', 'black', 'darkgreen',
                     'darkred', 'darkblue', 'darkcyan', 'gold', 'pink', 'salmon', 'seagreen', 'teal',
                     'navy', 'silver', 'maroon', 'gray', 'darkviolet', 'lightcoral', 'cyan', 'magenta',
                     'yellow', 'green', 'crimson', 'lavender', 'tomato', 'fuchsia', 'khaki', 'orange',
                     'peru']
    color_clean_options = ['lightgreen', 'orchid', 'aquamarine', 'mediumpurple', 'lime', 'lightsalmon', 'lightskyblue', 
                           'lightcoral', 'lightsteelblue', 'lightseagreen', 'lightcyan', 'lightgoldenrodyellow', 
                           'lightpink', 'lightslategray', 'lightgreen', 'lightblue', 'lightyellow', 'lightgray', 
                           'lightblue', 'lightgreen', 'lightpurple', 'lightcoral', 'lightcyan', 'lightmagenta', 
                           'lightyellow', 'lightgreen', 'lightcrimson', 'lightlavender', 'lighttomato', 'lightfuchsia', 
                           'lightkhaki', 'lightorange', 'lightperu']
    
    marker_dict = base_marker.copy()
    color_dict = base_color.copy()
    color_clean_dict = base_color_clean.copy()
    
    for i, anomaly in enumerate(category_anomalies):
        marker = marker_options[i % len(marker_options)]
        color = color_options[i % len(color_options)]
        color_clean = color_clean_options[i % len(color_clean_options)]
        
        marker_dict[anomaly] = marker
        color_dict[anomaly] = color
        color_clean_dict[anomaly] = color_clean
        
    return marker_dict, color_dict, color_clean_dict

anomaly_categories = {
    "bottle": ["broken_large", "broken_small", "contamination"],
    "cable": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"],
    "capsule": ["crack", "faulty_imprint", "poke", "scratch","squeeze"],
    "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
    "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": ["color", "combined","contamination", "crack", "faulty_imprint", "pill_type","scratch"],
    "screw": ["manipulated_front", "scratch_head", "scratch_neck","thread_side", "thread_top"],
    "tile": ["crack", "glue_strip", "gray_stroke", "oil","rough"],
    "toothbrush": ["defective"],
    "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
    "wood": ["color", "combined", "hole", "liquid", "scratch"],
    "zipper": ["broken_teeth", "combined","fabric_border", "fabric_interior","split_teeth","rough", "squeezed_teeth"]
}

from scipy.stats import ttest_rel ,ttest_ind
def paired_t_test_per_category(df, category_column):
    categories = df[category_column].unique()
    t_test_results = []

    for category in categories:
        print(category)
        #image
        category_data = df[df[category_column] == category]

        t_stat, p_value = ttest_rel(category_data['image_AUC_contaminated'],category_data['image_AUC_clean'],alternative='less')
        t_stat_image_median, p_value_image_median = ttest_ind(category_data['median_image_AUC_contaminated'], category_data['median_image_AUC_clean'], alternative='less')
        t_stat_pixel, p_value_pixel = ttest_ind(category_data['pixel_AUC_contaminated'], category_data['pixel_AUC_clean'], alternative='less')
        t_stat_pixel_median, p_value_pixel_median = ttest_ind(category_data['median_pixel_AUC_contaminated'], category_data['median_pixel_AUC_clean'], alternative='less')
        
        t_test_results.append({
                               
                               'category': category,

                               'image_median_clean':np.median(category_data['image_AUC_clean']),
                               'image_median_contaminated':np.median(category_data['image_AUC_contaminated']),
                               'pixel_median_clean':np.median(category_data['pixel_AUC_clean']),
                               'pixel_median_contaminated':np.median(category_data['pixel_AUC_contaminated']),
                               
                               'image_difference':np.mean(category_data['image_AUC_clean']-category_data['image_AUC_contaminated']),
                               'pixel_difference':np.mean(category_data['pixel_AUC_clean']-category_data['pixel_AUC_contaminated']),
                               
                               't_stat_image': t_stat,
                               'p_value_image': p_value,                            
                               't_stat_pixel': t_stat_pixel,
                               'p_value_pixel': p_value_pixel,
                               
                               'image_median_clean_robust':np.median(category_data['median_image_AUC_clean']),
                               'image_median_contaminated_robust':np.median(category_data['median_image_AUC_contaminated']),
                               'pixel_median_clean_robust':np.median(category_data['median_pixel_AUC_clean']),
                               'pixel_median_contaminated_robust':np.median(category_data['median_pixel_AUC_contaminated']),
                               
                               'image_difference_robust':np.mean(category_data['median_image_AUC_clean']-category_data['median_image_AUC_contaminated']),
                               'pixel_difference_robust':np.mean(category_data['median_pixel_AUC_clean']-category_data['median_pixel_AUC_contaminated']),
                               
                               't_stat_image_robust': t_stat_image_median,
                               'p_value_image_robust': p_value_image_median,                            
                               't_stat_pixel_robust': t_stat_pixel_median,
                               'p_value_pixel_robust': p_value_pixel_median,
                               
                               'var_image_AUC_contaminated':np.median(category_data['var_image_AUC_contaminated']),
                               'var_image_AUC_clean':np.median(category_data['var_image_AUC_clean']),
                               'var_pixel_AUC_contaminated':np.median(category_data['var_pixel_AUC_contaminated']),
                               'var_pixel_AUC_clean':np.median(category_data['var_pixel_AUC_clean'])})
        
    return pd.DataFrame(t_test_results)
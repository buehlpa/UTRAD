import matplotlib.pyplot as plt
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
# UTRAD
UTRAD for nueral networks

Code for the results of UTRAD model, and refinement for the work Unsupervised Anomaly Detection with Transformers Challenges of COntaminated Datasets.

NEEDS Cleanup

## Installation
This repo was tested with Ubuntu 16.04/18.04, Pytorch 1.5.0
## Running 
1. Fetch the Mvtec datasets, and extract to datasets/
2. Run training by using command:
```
python main.py --data_category grid
```
where --data_category is used to specify the catogory.

3. Validate with command:
```
python valid.py --exp_name Exp_15_02_24 --data_category grid --mode mvtec
```
4. Validate with unaligned setting:
```
python valid.py --exp_name Exp_15_02_24  --data_category grid  --mode mvtec --unalign_test
```



# experiments:


python main.py --exp_name ExpNAME --data_category grid --mode mvtec --contamination_rate 0.1

### only training data


trainscores.py # needed for testing if trainings scores are larger for insample scores..

# less data

--data_rate

# Run multiple scripts 

bash [path] /run_multiple.sh

# For differnet dataset
# mvtec loco
python main.py --exp_name Exp_17_02_24_baseline --data_category breakfast_box --mode mvtec_loco --data_set mvtec_loco --contamination_rate 0.0 --data_root /home/bule/projects/datasets/mvtec_loco_anomaly_detection



# development of refinement 

dev_refinement_main



# development of usdr

dev_usdr

# 

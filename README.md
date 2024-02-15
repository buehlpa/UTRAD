# UTRAD
UTRAD for nueral networks
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
Exp0-r18
- original configuration utrad

Exp1
reproduce results with contam schema  with contam = 0.0  , new datalaoder


python main.py --data_category grid --mode mvtec

ExP2 contaminatoion
reproduce results with contam schema  with contam > 0.0  , new datalaoder

python main.py --exp_name Exp2 --data_category grid --mode mvtec --contamination_rate 0.1



# Run multiple scripts 

bash [path] /run_multiple.sh


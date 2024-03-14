#!/bin/bash  Exp_04_03_24
python dev_refinement_main.py --exp_name Exp_14_03_24 --data_category screw --mode mvtec --contamination_rate 0.0 --assumed_contamination_rate 0.1
python dev_refinement_main.py --exp_name Exp_14_03_24 --data_category screw --mode mvtec --contamination_rate 0.1 --assumed_contamination_rate 0.1

python valid.py --exp_name Exp_14_03_24 --data_category screw --mode mvtec --contamination_rate 0.0 
python valid.py --exp_name Exp_14_03_24 --data_category screw --mode mvtec --contamination_rate 0.1 


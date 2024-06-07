#!/bin/bash

# Generate commands for each path
python add_gaussian_res.py --exp_name Exp_04_06_24_run_1 --data_category screw --mode mvtec
python add_gaussian_res.py --exp_name Exp_04_06_24_run_2 --data_category screw --mode mvtec
python add_gaussian_res.py --exp_name Exp_04_06_24_run_3 --data_category screw --mode mvtec
python add_gaussian_res.py --exp_name Exp_04_06_24_run_4 --data_category screw --mode mvtec
python add_gaussian_res.py --exp_name Exp_04_06_24_run_5 --data_category screw --mode mvtec


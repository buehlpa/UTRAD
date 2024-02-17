#!/bin/bash  

python main.py --exp_name Exp_17_02_24_baseline --data_category breakfast_box --mode mvtec_loco --contamination_rate 0.0

python main.py --exp_name Exp_17_02_24_baseline --data_category juice_bottle --mode mvtec_loco --contamination_rate 0.0

python main.py --exp_name Exp_17_02_24_baseline --data_category pushpins --mode mvtec_loco --contamination_rate 0.0

python main.py --exp_name Exp_17_02_24_baseline --data_category screw_bag --mode mvtec_loco --contamination_rate 0.0

python main.py --exp_name Exp_17_02_24_baseline --data_category splicing_connectors --mode mvtec_loco --contamination_rate 0.0

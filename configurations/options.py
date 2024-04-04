import argparse
import os
import torch
import json 

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name',       type=str, default="Exp1", help='the name of the experiment')
        self.parser.add_argument('--epoch_start',    type=int, default=0 , help='epoch to start training from')
        self.parser.add_argument('--epoch_num',      type=int, default=150, help='number of epochs of training')
        self.parser.add_argument('--factor',         type=int, default= 1, help='not implemented yet')
        self.parser.add_argument('--seed',           type=int, default=233, help='random seed')
        self.parser.add_argument('--fixed_seed_bool',type=bool, default=True, help='whether to use fixed seed, if False, seed is set to time.time()')
        self.parser.add_argument('--num_row',        type=int, default=4, help='number of image in a rows for display')
        self.parser.add_argument('--activation',     type=str, default='gelu', help='activation type for transformer')
        self.parser.add_argument('--unalign_test',  action='store_true', default=False, help='whether to valid with unaligned data: \ in this mode, test images are random ratate +-10degree, and randomcrop from 256x256 to 224x224')
        self.parser.add_argument('--data_root', type=str, default='/home/bule/datasets/mvtec_anomaly_detection/', help='dir of the dataset')
        self.parser.add_argument('--data_category', type=str, default="cable", help='category name of the dataset')
        self.parser.add_argument('--data_set', type=str, default="mvtec", help='dataset , mvtec , mvtec_loco, beantec')
        self.parser.add_argument('--batch_size', type=int, default=2, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
        self.parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
        self.parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
        self.parser.add_argument('--results_dir', type=str, default='results', help='the overal result directory')
        self.parser.add_argument('--image_result_dir', type=str, default='result_images', help=' where to save the result images')
        self.parser.add_argument('--model_result_dir', type=str, default='saved_models', help=' where to save the checkpoints')
        self.parser.add_argument('--validation_image_dir', type=str, default='validation_images', help=' where to save the validation image')
        self.parser.add_argument('--contamination_rate', type=float, default=0.0, help='How much the training dataset is contaminated with anomalies from the test dataset max 0.1')
        self.parser.add_argument('--assumed_contamination_rate', type=float, default=0.0, help='How much you assume that the training dataset is contaminated')
        self.parser.add_argument('--mode', type=str, default='utrad_mvtec', help='Which dataloader should be used: one of utrad_mvtec,mvtec, mvtec_loco, beantec,visa')
        self.parser.add_argument('--development', type=bool, default=False, help='used for development mode , some paths are not saved ')


## TODO mode dataset  are equvivalent, remove trhough whole dataset

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        os.makedirs(args.results_dir,exist_ok=True)
        os.makedirs(os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}', args.image_result_dir), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir,args.data_set ,f'contamination_{int(args.contamination_rate*100)}',f'{args.exp_name}-{args.data_category}', args.model_result_dir), exist_ok=True)

        try:
            with open(os.path.join('configurations',f'{args.data_set}.json' ), 'r') as file:
                dataset_parameters = json.load(file)
            setattr(args, 'dataset_parameters', dataset_parameters)
        except FileNotFoundError:
            print(f"Configuration file for {args.data_set} not found. Proceeding with default parameters.")
            setattr(args, 'dataset_parameters', {})
        
        self.args = args
        return self.args

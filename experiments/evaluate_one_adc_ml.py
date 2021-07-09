#!/usr/bin/env python
import os
import mlflow
import math
import random
import pdb
import argparse
from ssltsc.experiments import get_experiment_id

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate parser')
    parser.add_argument('--code_path', type=str, default='/data/beegfs/home/gos/sslts/experiments',
                        help='path to code on cluster')
    # parser.add_argument('--seeds', nargs='+', type=int, default=[91, 2828, 19, 28, 447, 138, 299, 123, 988, 16161],
    parser.add_argument('--seeds', nargs='+', type=int, default=[138, 299, 123, 988, 16161],
                        help='seed for the different unlabeling runs')
    parser.add_argument('--num_labels', nargs='+', type=int, default=[50, 100, 250, 500, 1000],
                        help='amounts of labels')
    parser.add_argument('--model_name', type=str, default='randomforest',
                        help='name of the model')
    parser.add_argument('--mlflow_name', type=str, default=None)
    parser.add_argument('--mlflow_id', type=int, default=None)
    return parser.parse_args()

def run_eval(args):
    # read the best config that matches a) the model and b) the mlflow name which we want to evaluate
    mlflow_id = get_experiment_id(args=args)
    all_paths = os.listdir(f'mlruns/{mlflow_id}/')
    all_paths = [path for path in all_paths if not path.endswith('.png') and not path.endswith('.yaml') and not path.endswith('.csv')]
    for path in all_paths:
        if open(f'mlruns/{mlflow_id}/{path}/params/model').read() == args.model_name:
            pdb.set_trace()
            config_file = f'mlruns/{mlflow_id}/{path}/artifacts/best_config.yaml'
            break

    for num_label in args.num_labels:
        for seed in args.seeds:
            # run call
            # shebang ~/miniconda3/envs/sslts_env/bin/python
            command = f'./run_ml.py --num_labels {num_label} --seed {seed} --config {config_file} --mlflow_id {mlflow_id}'
            print(command)
            os.system(command)

if __name__ =='__main__':
    run_eval(args=parse_args())
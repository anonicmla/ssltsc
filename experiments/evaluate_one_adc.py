# create sbatch scripts to run slurm for
# evaluation of one adc
import os
import mlflow
import math
import random
import argparse
from ssltsc.experiments import get_experiment_id

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate parser')
    parser.add_argument('--code_path', type=str, default='/data/beegfs/home/gos/sslts/experiments',
                        help='path to code on cluster')
    parser.add_argument('--seeds', nargs='+', type=int, default=[138, 299, 123, 988, 16161],
                        help='seed for the different unlabeling runs')
    parser.add_argument('--num_labels', nargs='+', type=int, default=[50, 100, 250, 500, 1000],
                        help='amounts of labels')
    parser.add_argument('--model_name', type=str, default='vat',
                        help='name of the model')
    parser.add_argument('--mlflow_name', type=str, default=None)
    parser.add_argument('--mlflow_id', type=int, default=0)
    return parser.parse_args()


def create_sb_scripts(args):
    commands = []

    # read the best config that matches a) the model and b) the mlflow name which we want to evaluate
    mlflow_id = get_experiment_id(args=args)
    all_paths = os.listdir(f'mlruns/{mlflow_id}/')
    all_paths = [path for path in all_paths if not path.endswith('.png') and not path.endswith('.yaml') and not path.endswith('.csv')]
    for path in all_paths:
        if open(f'mlruns/{mlflow_id}/{path}/params/model').read() == args.model_name:
            config_file = f'mlruns/{mlflow_id}/{path}/artifacts/best_config.yaml'
            break

    for num_label in args.num_labels:
        for seed in args.seeds:
            # run call
            # shebang ~/miniconda3/envs/sslts_env/bin/python
            commands.append(f'run.py --val_steps 1000 --num_labels {num_label} --seed {seed} --config {config_file} --mlflow_id {args.mlflow_id} --early_stopping')

    idx = 0
    for command in commands:
        # exclude 2/8 gpu nodes aka leave 8 GPUs spare
        command_temp = f'sbatch --exclude=hpc-gpu-08,hpc-gpu-01 --partition=gpu --job-name=eval_{args.model_name}_{idx} --output=/data/beegfs/home/gos/out/{args.model_name}_mlflow{mlflow_id}_{idx} --error=/data/beegfs/home/gos/error/{args.model_name}_mlflow{mlflow_id}_{idx} -D {args.code_path} --gres=gpu:tesla:1 {command}'
        os.system(command_temp)
        idx += 1

if __name__ =='__main__':
    create_sb_scripts(args=parse_args())
from ssltsc.postprocessing import visualize_experiment
from ssltsc.experiments import get_experiment_id
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='post process experiment')
    parser.add_argument('--mlflow_id', type=int, default=None,
                        help='number of trials (default: None)')
    parser.add_argument('--mlflow_name', type=str, default=None,
                        help='number of trials (default: None)')

    return parser.parse_args()

def postprocess(args):
    mlflow_id = get_experiment_id(args=args)
    visualize_experiment(mlflow_id=mlflow_id)

if __name__ == "__main__":
    args = parse_args()
    postprocess(args)

import torch
import train_pinn
import train_nn
import numpy as np

import sys
import argparse
import yaml
import init_params as ip

def parse_yaml_from_args():
    parser = argparse.ArgumentParser(description="Parse a YAML file from the command line.")
    parser.add_argument("yaml_file", help="Path to the YAML file")
    args = parser.parse_args()

    try:
        with open(args.yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: YAML file not found: {args.yaml_file}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    return yaml_data


if __name__ == '__main__':
    cfg = parse_yaml_from_args()  # Parse the configuration file from arguments

    # If we have a GPU available, we'll set our device to GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        torch.set_default_device('cuda')
    else:
        device = torch.device("cpu")
        print('No GPU available')

    torch.set_default_dtype(torch.float)  # Set default dtype to float32
    torch.manual_seed(cfg['seed'])  # PyTorch random number generator
    np.random.seed(cfg['seed'])  # Random number generators in other libraries

    'Set battery parameters and loss weights'
    p = ip.InitParams(cfg)
    p.input_size = 2  # number of input for LSTM. [I, Vt]
    p.output_size = cfg['n_r'] - 1  # number of outputs per NN, i.e., [cs(r=1), ..., cs(r=Nr-1)] for anode or cathode


    if cfg['method'] == 'pinn':
        train_pinn.train(cfg, device, p)
    else:
        train_nn.train(cfg, device, p)

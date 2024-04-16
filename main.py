import yaml
import argparse
from datetime import datetime
from utils import *
from utils.sys_utils import extract_configs, sanitize_dict_string
from train_toy_transformer import run_toy_transformer_config
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config','--config', default = 'config/IMDB/signum.yml',type=str) # specify configuration file
    parser.add_argument('--device','--device', default= '[7]') # see free GPU and specify here. for multiple (big models), do e.g. [2,3]
    args = parser.parse_args()

    #get config
    config_file = open(args.config, 'r')
    config = yaml.load(config_file,Loader=yaml.FullLoader)
    config_list = extract_configs(config)
    for config_local in config_list:
        run_name = sanitize_dict_string(config_local)[:-1]
        print(run_name)
        run_toy_transformer_config(config_local, args.device)
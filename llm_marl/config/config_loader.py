#!/usr/bin/env python

# license TBD

import os 

import yaml

def load_config(file_path: str):
    """Load library configuration from a yaml file
    """
    if os.path.exists(file_path) and os.path.isfile(file_path) and file_path.endswith("yaml"):
        config = yaml.safe_load(open(file_path))
        return config
    else:
        raise ValueError("The config file must be a valid yaml file.")

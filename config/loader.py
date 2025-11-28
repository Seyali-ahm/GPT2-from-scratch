import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gpt2_124M.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

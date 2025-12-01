import os
import torch
import urllib.request
from dataloaders.gpt_dataloader import create_dataloader_v1
from config.loader import CONFIG
config_data = CONFIG["data"]
config_model = CONFIG['model']
config_training = CONFIG['training']

def load_text_dataset(config_data):
    file_path = config_data['file_path']
    url = config_data['url']

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()

    # Split
    split = int(config_data["train_ratio"] * len(text_data))
    return text_data[:split], text_data[split:]


train_data, val_data = load_text_dataset(CONFIG["data"])



torch.manual_seed(config_training["seed"])

train_loader = create_dataloader_v1(
    train_data,
    batch_size=config_training["batch_size"],
    max_length=config_model["context_length"],
    stride=config_model["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=config_training["batch_size"],
    max_length=config_model["context_length"],
    stride=config_model["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
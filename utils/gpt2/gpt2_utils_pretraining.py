import os
from utils.gpt2.load_gpt2_tf_weights import download_gpt2

def get_project_root():
    # This file: utils/gpt2/your_file.py
    current = os.path.dirname(os.path.abspath(__file__))
    
    # Move from: utils/gpt2/ → utils/ → project root
    return os.path.dirname(os.path.dirname(current))


def gpt2_weights_exist(model_size="124M", folder="gpt2"):
    project_root = get_project_root()
    path = os.path.join(project_root, folder, model_size)
    return os.path.isdir(path)


def pretrain_gpt2(model_size="124M"):
    project_root = get_project_root()
    folder = os.path.join(project_root, "gpt2")

    if gpt2_weights_exist(model_size, folder="gpt2"):
        print(f"[INFO] GPT-2 {model_size} weights found locally. Loading...")
        settings, params = download_gpt2(model_size)  # loads without redownload
        return settings, params

    else:
        print(f"[INFO] GPT-2 {model_size} weights NOT found.")

        choice = input("Download GPT-2 weights now? (y/n): ")

        if choice.lower() == "y":
            print("[INFO] Downloading GPT-2 weights...")
            settings, params = download_gpt2(model_size)
            print("[INFO] Download complete.")
            return settings, params

        else:
            print("[INFO] Pretraining cancelled (no weights).")
            return None, None

from utils.gpt2.gpt_download3 import download_and_load_gpt2
import os

def download_gpt2(model_size, models_dir="gpt2"):
    """
    Downloads GPT-2 TF checkpoint into project root and returns settings + params.
    """
    # Current file: utils/gpt2/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Project root: GPT2_from_scratch/
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Final absolute folder to save GPT-2 weights
    models_dir = os.path.join(project_root, models_dir)

    # Download + convert weights
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    return settings, params
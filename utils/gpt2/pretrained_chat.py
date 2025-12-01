from utils.gpt2.gpt2_utils_pretraining import pretrain_gpt2   # your loader
from model.gpt_model import GPTModel
from utils.tokenization import text_to_token_ids, token_ids_to_text
from config.loader import CONFIG
import tiktoken
import torch
from utils.gpt2.load_weights import load_weights_into_gpt


def run_pretrained_chat():
    config_model = CONFIG["model"]
    config_gen = CONFIG["generation"]
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load pretrained GPT-2 weights
    settings, params = pretrain_gpt2()

    if settings is None:
        print("[ERROR] Cannot run pretraining without GPT-2 weights.")
        return

    # Build model
    gpt = GPTModel(config_model)
    gpt.eval()

    # Load weights
    load_weights_into_gpt(gpt, params)

    gpt.to(device)

    print("[INFO] GPT-2 pretrained weights loaded successfully!")
    print("\n[INFO] GPT-2 is ready!")
    print("[INFO] Type 'exit' to quit.\n")

    # Start generation loop
    while True:
        user_prompt = input("Enter a prompt: ")

        if user_prompt.lower() in ["exit", "quit", "q"]:
            print("\n[INFO] Exiting generation mode.")
            break

        idx = text_to_token_ids(user_prompt, tokenizer).to(device)

        token_ids = gpt.generate(
            idx=idx,
            max_new_tokens=config_gen["max_new_tokens"],
            top_k=config_gen["top_k"],
            temperature=config_gen["temperature"]
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)

        print("\n======= GENERATED TEXT =======")
        print(generated_text)
        print("================================\n")
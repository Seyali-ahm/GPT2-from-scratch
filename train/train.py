
import torch
import tiktoken
from model.gpt_model import GPTModel
from config.loader import CONFIG
from data.load_text_data import train_loader, val_loader
from train.train_utils import train_model_simple


config_model = CONFIG['model']
config_training = CONFIG["training"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
print(device)

def main():
    import time
    start_time = time.time()

    torch.manual_seed(123)
    model = GPTModel(config_model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config_training["lr"], 
                                  weight_decay=config_training["weight_decay"])

    num_epochs = config_training['num_epochs']
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=config_training["eval_freq"], eval_iter=config_training['eval_iter'],
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

if __name__ == "__main__":
    main()
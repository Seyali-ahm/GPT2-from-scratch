from train.train import train_main
from utils.gpt2.pretrained_chat import run_pretrained_chat


def main():
    print("Choose an option:")
    print("1) Train GPT-2 from scratch")
    print("2) Pretrain GPT-2")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        print("[INFO] Starting training...")
        train_main()   # your train function

    elif choice == "2":
        print("[INFO] Loading pretrained weights...")
        run_pretrained_chat()
if __name__ == "__main__":
    main()

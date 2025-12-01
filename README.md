
# GPT2‑from‑scratch

A self‑contained implementation and training pipeline of a GPT‑2–style language model from scratch (in pure PyTorch / Python).  
This repository provides all necessary modules — data loading, model definition, training scripts, and utilities — to train and experiment with GPT‑2-like models on your own text data.

## 🚀 Why this project

- Educational — to learn how a Transformer / GPT‑2 model works under the hood, without relying on high‑level abstractions or pre-built frameworks.  
- Flexible & Customizable — the configuration and data modules let you adapt vocabulary, dataset, hyperparameters, and training regime.  
- Reproducible — everything from data loading to model architecture to training is in one place, so others can pick up, reproduce, or adapt experiments.

## 📂 Repository structure

```
/
|-- config/             # configuration files and settings
|-- data/               # raw / processed dataset
|-- dataloaders/        # data loading and batching utilities
|-- model/              # model definitions
|-- layers/             # attention, feed‑forward, normalization, etc.
|-- train/              # training scripts and checkpoints
|-- utils/              # helper functions (tokenization, downloading Real GPT2 weights, ... )
|-- Research/           # notes, experiments, logs
|-- README.md
```

## 🔧 Getting Started

### Requirements

- Python 3.12.7  
- PyTorch  
- Other dependencies (if any) listed in `requirements.txt`

Install requirements:

```bash
pip install -r requirements.txt
```

### Data Preparation
If you want to put your text data, and train the model, do as follows:
1. Inside config folder, find gpt2_124M file.  
2. Put any text that can be downloaded from the internet in url part.  
3. Run the main function and choose 1.
By default, it downloads the book named "the verdict".
it is proper for educational purposes

### Training

Run:

```bash
python main.py
```

### Text Generation

```bash
python main.py
choose 2 and then wait to download the pretrained weights of GPT2-124M params.
After that it is possible to interactively send the text to produce next tokens.
you can change the value for top_k sampling and temperature scaling in gpt_124M yaml file```

## ⭐ Features

- Fully implemented GPT‑2‑style architecture  
- Modular components for easy customization  
- Config‑driven training setup  
- Sampling / generation utilities

## 📚 Learnings

- Tokenization  
- Embeddings & positional encodings  
- Multi‑head self‑attention  
- Transformer blocks  
- Language‑model training loops  
- Sampling strategies
- Loading pretrained weights and assigning to the model

## 📝 Contribution

1. Fork the repository  
2. Create a feature branch  
3. Commit changes  
4. Open a pull request  




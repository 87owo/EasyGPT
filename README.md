# EasyGPT

Easily train and inference on your personal computer, no need for large scale clusters!

<img width="1397" height="703" alt="image" src="https://github.com/user-attachments/assets/718aed82-5a84-4d39-ad4d-eb8b2fb031e8" />

## Requirements

You need to select the appropriate command according to the system and CUDA version.

```
pip install bitsandbytes
pip install safetensors
pip install torch
pip install tqdm
```

## File Information

The following lists the storage locations of all relevant code and other related documents.

```
EasyGPT/
├── data/                    # Training text dataset
│
├── model/
│   ├── stage_epoch_*/       # Model training save location
│   └── ...
│
├── train.py                 # Model training complete code
├── chat.py                  # Model dialogue complete code
└── ...                      # Other supplementary folders and files
```

## Chat With EasyGPT

<img width="1396" height="538" alt="image" src="https://github.com/user-attachments/assets/5b0850fa-a1e2-48ff-aa82-c6153bd899c7" />

## Dataset Format

Dataset Example Download: https://github.com/87owo/EasyGPT/releases

```
<|user|>Hello!<|assistant|>Hello! I am EasyGPT, an AI assistant. How can I help you?<|end|>
<|user|>Can you introduce yourself?<|assistant|>Sure! I am EasyGPT, an AI assistant.<|end|>
<|user|>Who are you?<|assistant|>Hello, My name is EasyGPT, an AI training by 87owo.<|end|>
```

## Model Config

8GB gpu memory configuration, if you have more gpu memory, you can increase batch_size appropriately

| Params | hidden_size | ffn_hidden_size | block_count | num_heads | vocab_size | batch_size |
|--------|-------------|-----------------|-------------|-----------|------------|------------|
| 421M   | 1024        | 4096            | 24          | 16        | 32000      | 2          |
| 182M   | 768         | 3072            | 16          | 12        | 32000      | 4          |
| 77M    | 512         | 2048            | 12          | 8         | 32000      | 8          |

## Official Website

https://github.com/87owo/EasyGPT

## Project License

https://github.com/87owo/EasyGPT/blob/main/LICENSE.md

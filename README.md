# EasyGPT

Easily train and inference on your personal computer, no need for large scale clusters!

It is recommended to have at least 10,000 pieces of training data and 10 training rounds.

## Requirements

You need to select the appropriate command according to the system and CUDA version.

```
pip install bitsandbytes
pip install safetensors
pip install torch
pip install tqdm
```

## File Information

```
EasyGPT/
├── data/ 
│   └── dataset.txt (Training format example)
│
├── model/ 
│   └── ... (Model training save location)
│
├── train.py (Model training complete code)
├── chat.py (Model dialogue complete code)
└── ...
```

## Dataset Format

Dataset Example Download: https://github.com/87owo/EasyGPT/releases

```
<|user|>Hello!<|assistant|>Hello! I am EasyGPT, an AI assistant. How can I help you?<|end|>
<|user|>Can you introduce yourself?<|assistant|>Sure! I am EasyGPT, an AI assistant.<|end|>
...
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

## MIT license

https://github.com/87owo/EasyGPT/blob/main/LICENSE

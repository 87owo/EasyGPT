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

```
<|user|>Hello!<|assistant|>Hello! I am EasyGPT, an AI assistant. How can I help you?<|end|>
```

## Dataset Example

https://github.com/87owo/EasyGPT/releases

## Model Config

Optimal model size configuration table for 8GB memory

| Model Name | params | hidden_size | ffn_hidden_size | block_count | num_heads | vocab_size | batch_size |
|------------|--------|-------------|-----------------|-------------|-----------|------------|------------|
| EasyGPT-M  | 400M   | 1024        | 4096            | 24          | 16        | 32000      | 2          |
| EasyGPT-S  | 70M    | 512         | 2048            | 12          | 8         | 32000      | 8          |

## Official Website

https://github.com/87owo/EasyGPT

## MIT license

https://github.com/87owo/EasyGPT/blob/main/LICENSE

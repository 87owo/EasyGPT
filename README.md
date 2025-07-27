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

## Dataset Example

https://github.com/87owo/EasyGPT/releases

```
<|user|>Hello!<|assistant|>Hello! I am EasyGPT, an AI assistant. How can I help you?<|end|>
```

## Official Website

https://github.com/87owo/EasyGPT

## MIT license

https://github.com/87owo/EasyGPT/blob/main/LICENSE

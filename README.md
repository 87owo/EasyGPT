# EasyGPT

Easily train and inference on your personal computer, no need for large scale clusters!
It is recommended to have at least 10,000 pieces of training data and 20 training rounds.

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
│   └── dialogues.txt (Training format example)
│
├── model/ 
│   └── ... (Model training save location)
│
├── train.py (Model training complete code)
├── chat.py (Model dialogue complete code)
└── ...
```

## Official Website

https://github.com/87owo/EasyGPT

## MIT license

https://github.com/87owo/EasyGPT/blob/main/LICENSE

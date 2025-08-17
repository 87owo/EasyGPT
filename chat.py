import os, json, torch
from safetensors.torch import load_file
from train import *
from collections import OrderedDict
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

# ================================================

def sample_next_token(logits, generated_tokens, repetition_penalty, presence_penalty, temperature):
    for token in set(generated_tokens):
        if logits[token] < 0:
            logits[token] *= repetition_penalty
        else:
            logits[token] /= repetition_penalty
    vocab_size = logits.size(0)
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    mask[list(set(generated_tokens))] = True
    logits[mask] += presence_penalty
    probs = torch.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item(), probs

# ================================================

def generate_response(model, tokenizer, prompt, device, config, max_length=512, temperature=0.3, repetition_penalty=1.0, presence_penalty=-1.5):
    encoded = tokenizer(f"<|user|>{prompt}<|assistant|>", update=False)
    generated = encoded["input_ids"].unsqueeze(0).to(device)
    unknown_id = tokenizer.split_tokens.get("<|unknown|>")
    end_id = tokenizer.split_tokens.get("<|end|>")
    newline_id = tokenizer.split_tokens.get("\\n")

    print(Fore.GREEN + "Assistant:" + Style.RESET_ALL, end=" ", flush=True)

    with torch.no_grad():
        for _ in range(max_length):
            if generated.size(1) > config["max_seq_length"]:
                current_input = generated[:, -config["max_seq_length"] :]
                pos_offset = generated.size(1) - config["max_seq_length"]
            else:
                current_input = generated
                pos_offset = 0

            outputs = model(current_input, pos_offset=pos_offset)
            logits = outputs["logits"][0, -1, :].clone()
            gen_tokens = generated[0].tolist()
            token_id, probs = sample_next_token(logits, gen_tokens, repetition_penalty, presence_penalty, temperature)

            if token_id == unknown_id and probs.sum() > 0:
                probs[unknown_id] = 0.0
                probs = probs / probs.sum()
                token_id = torch.multinomial(probs, num_samples=1).item()

            generated = torch.cat((generated, torch.tensor([[token_id]], device=generated.device)), dim=1)
            if token_id == end_id:
                break
            token_str = tokenizer.decode([token_id])
            if token_id == newline_id:
                print()
            else:
                print(token_str, end="", flush=True)
    print()

# ================================================

def load_chat_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        token_dict = json.load(f)
    tokenizer = ChatTokenizer(config)
    tokenizer.split_tokens = OrderedDict(token_dict)
    model = ChatModel(config).to(device)
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()

    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return model, tokenizer, config

# ================================================

if __name__ == "__main__":
    print("EasyGPT Beta V1.5 Torch Inference (Dev)")
    model_dir = "./model/Fine-tuning_epoch_12"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_chat_model(model_dir, device)
    while True:
        print("=" * 50)
        prompt = input(Fore.CYAN + "User:" + Style.RESET_ALL + " ")
        if prompt.strip().lower() in ["exit", "quit"]:
            break
        generate_response(model, tokenizer, prompt, device, config)
